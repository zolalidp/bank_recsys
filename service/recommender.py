"""Ядро рекомендаций: модель + бандл фичей + витрина.

Отделено от FastAPI, чтобы логику можно было тестировать и переиспользовать
(например, в батч-скоринге) без веб-слоя.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostClassifier

from scripts import serving_store
from scripts.artifacts import FeatureArtifacts
from scripts.features import build_long
from scripts.train import predict_scores, rank_topk
from service.monitoring import PREDICTION_SECONDS

TOP_K = 7


@dataclass
class Recommendation:
    product: str
    score: float | None  # None у выдачи холодного старта: у популярности нет вероятности


@dataclass
class RecommendResult:
    ncodpers: int
    source: str  # "model" | "popular"
    stale_months: int | None  # на сколько месяцев устарел снапшот клиента; None для popular
    items: list[Recommendation]


class Recommender:
    def __init__(self, model: CatBoostClassifier, bundle: FeatureArtifacts):
        bundle.validate()
        self.model = model
        self.bundle = bundle

    @classmethod
    def load(cls, model_path: str | Path, bundle_dir: str | Path) -> "Recommender":
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return cls(model, FeatureArtifacts.load(bundle_dir))

    def popular(self, ncodpers: int = 0) -> RecommendResult:
        return RecommendResult(
            ncodpers=ncodpers,
            source="popular",
            stale_months=None,
            items=[Recommendation(p, None) for p in self.bundle.pop_order[:TOP_K]],
        )

    def recommend(self, ncodpers: int) -> RecommendResult | None:
        """None — клиента нет в витрине, решение об ответе за эндпоинтом."""
        rows = serving_store.fetch_clients([ncodpers], self.bundle.products)
        if not len(rows):
            return None

        candidates = build_long(
            rows, self.bundle.products, self.bundle.pop_rate, self.bundle.affinity,
            with_labels=False,
        )
        # Клиент, владеющий всем каталогом: рекомендовать нечего, но 404
        # был бы неверен — он в витрине есть.
        if not len(candidates):
            return RecommendResult(
                ncodpers=ncodpers, source="model",
                stale_months=self._staleness(rows), items=[],
            )

        t0 = time.perf_counter()
        scores = predict_scores(self.model, candidates)
        PREDICTION_SECONDS.observe(time.perf_counter() - t0)
        top = rank_topk(candidates, scores, k=TOP_K)[ncodpers]

        by_product = dict(
            zip(candidates["product"].astype(str).to_numpy(), scores)
        )
        return RecommendResult(
            ncodpers=ncodpers,
            source="model",
            stale_months=self._staleness(rows),
            items=[Recommendation(p, round(float(by_product[p]), 6)) for p in top],
        )

    @staticmethod
    def _staleness(rows) -> int:
        return int(serving_store.latest_month_id() - rows["month_id"].iloc[0])
