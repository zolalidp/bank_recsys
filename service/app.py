"""HTTP-слой сервиса рекомендаций.

Эндпоинты:
- GET /recommend/{ncodpers} — известный клиент: ранжирование моделью;
  неизвестный: топ-7 популярных (холодный старт) с пометкой source=popular.
- GET /recommend/popular — топ-7 популярных явно.
- GET /health — прод-проверка: модель загружена и витрина отвечает.

Модель и бандл грузятся один раз при старте из локальных файлов (кладутся в
образ на этапе сборки, см. Dockerfile) — сервис не зависит от MLflow в
рантайме, только от Postgres с витриной.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts import serving_store
from service.logging_setup import get_logger, setup_logging
from service.monitoring import init_counters, instrument, record_result
from service.recommender import Recommender

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.cb")
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "models/feature_artifacts")

setup_logging()
log = get_logger("service")


class RecommendationOut(BaseModel):
    product: str
    score: float | None


class RecommendResponse(BaseModel):
    ncodpers: int
    source: str
    stale_months: int | None
    recommendations: list[RecommendationOut]


@asynccontextmanager
async def lifespan(app: FastAPI):
    rec = Recommender.load(MODEL_PATH, BUNDLE_DIR)
    app.state.recommender = rec
    init_counters(rec.bundle.products)
    log.info(
        "сервис готов: модель %s деревьев, %s продуктов, top-%s",
        rec.model.tree_count_, len(rec.bundle.products), 7,
    )
    yield
    log.info("сервис останавливается")


app = FastAPI(title="bank_recsys", lifespan=lifespan)
instrument(app)


def _to_response(result) -> RecommendResponse:
    return RecommendResponse(
        ncodpers=result.ncodpers,
        source=result.source,
        stale_months=result.stale_months,
        recommendations=[
            RecommendationOut(product=r.product, score=r.score) for r in result.items
        ],
    )


@app.get("/health")
def health():
    r: Recommender = app.state.recommender
    try:
        latest = serving_store.latest_month_id()
    except Exception as exc:
        log.error("health: витрина недоступна: %s", exc)
        raise HTTPException(status_code=503, detail=f"витрина недоступна: {exc}")
    return {
        "status": "ok",
        "model_trees": r.model.tree_count_,
        "products": len(r.bundle.products),
        "store_latest_month_id": latest,
    }


@app.get("/recommend/popular", response_model=RecommendResponse)
def recommend_popular():
    r: Recommender = app.state.recommender
    result = r.popular()
    record_result(result)
    return _to_response(result)


@app.get("/recommend/{ncodpers}", response_model=RecommendResponse)
def recommend(ncodpers: int):
    r: Recommender = app.state.recommender
    result = r.recommend(ncodpers)
    if result is None:
        # Новый клиент: отдаём популярное, а не 404 — так фронту не нужно
        # обрабатывать два сценария; source=popular говорит, что это фолбэк.
        log.info("recommend ncodpers=%s: не найден в витрине, холодный старт", ncodpers)
        result = r.popular(ncodpers)
    else:
        log.info(
            "recommend ncodpers=%s: source=%s stale=%s top=%s",
            ncodpers, result.source, result.stale_months,
            result.items[0].product if result.items else "-",
        )
    record_result(result)
    return _to_response(result)
