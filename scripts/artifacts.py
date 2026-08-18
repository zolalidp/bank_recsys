"""Бандл артефактов, без которого обученную модель нельзя применить.

Модель использует фичи product_pop_rate и affinity_score, которые считаются
по обучающей выборке (compute_product_stats). Если сохранить только model.cb,
эти две фичи в проде невоспроизводимы и модель бесполезна. Поэтому статистики
сохраняются рядом с моделью и версионируются вместе с ней.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUNDLE_NAME = "feature_artifacts"


@dataclass
class FeatureArtifacts:
    """Всё, что нужно, чтобы построить фичи для инференса.

    products: порядок продуктов; задаёт индексацию pop_rate и affinity.
    pop_rate: базовая частота покупки продукта среди валидных кандидатов.
    affinity: affinity[i, j] = P(купит j | уже владеет i).
    pop_order: продукты по убыванию популярности — выдача холодного старта.
    feature_cols: порядок колонок, в котором модель ждёт признаки.
    """

    products: list[str]
    pop_rate: np.ndarray
    affinity: np.ndarray
    pop_order: list[str]
    feature_cols: list[str]

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.savez(
            directory / f"{BUNDLE_NAME}.npz",
            pop_rate=self.pop_rate,
            affinity=self.affinity,
        )
        (directory / f"{BUNDLE_NAME}.json").write_text(
            json.dumps(
                {
                    "products": self.products,
                    "pop_order": self.pop_order,
                    "feature_cols": self.feature_cols,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "FeatureArtifacts":
        directory = Path(directory)
        arrays = np.load(directory / f"{BUNDLE_NAME}.npz")
        meta = json.loads((directory / f"{BUNDLE_NAME}.json").read_text())
        return cls(
            products=meta["products"],
            pop_rate=arrays["pop_rate"],
            affinity=arrays["affinity"],
            pop_order=meta["pop_order"],
            feature_cols=meta["feature_cols"],
        )

    def validate(self) -> None:
        """Ловит рассинхрон между списком продуктов и матрицами — иначе
        affinity[i, j] молча начнёт указывать не на тот продукт."""
        n = len(self.products)
        if self.pop_rate.shape != (n,):
            raise ValueError(f"pop_rate {self.pop_rate.shape}, ожидалось ({n},)")
        if self.affinity.shape != (n, n):
            raise ValueError(f"affinity {self.affinity.shape}, ожидалось ({n}, {n})")
        if sorted(self.pop_order) != sorted(self.products):
            raise ValueError("pop_order и products содержат разные продукты")


def make_artifacts(
    products: list[str], pop_rate: np.ndarray, affinity: np.ndarray, feature_cols: list[str]
) -> FeatureArtifacts:
    """Собирает бандл, выводя pop_order из pop_rate."""
    order = [products[i] for i in np.argsort(-pop_rate)]
    bundle = FeatureArtifacts(
        products=list(products),
        pop_rate=np.asarray(pop_rate, dtype="float32"),
        affinity=np.asarray(affinity, dtype="float32"),
        pop_order=order,
        feature_cols=list(feature_cols),
    )
    bundle.validate()
    return bundle
