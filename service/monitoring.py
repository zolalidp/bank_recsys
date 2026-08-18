"""Prometheus-метрики сервиса. Описание каждой метрики — в Monitoring.md.

Дефолтные коллекторы prometheus_client (process_*, python_gc_*) дают CPU и
память процесса бесплатно — их не отключаем, дашборд на них опирается.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Время обработки HTTP-запроса, от получения до ответа",
    ["method", "handler", "status"],
    # Плотнее в районе ожидаемых ~0.1-0.3 с (поход в Postgres + инференс).
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0),
)

RECOMMENDATIONS = Counter(
    "recommendations_total",
    "Сколько раз продукт попал в выдачу top-7",
    ["product", "source"],
)

RECOMMEND_REQUESTS = Counter(
    "recommend_requests_total",
    "Запросы рекомендаций по источнику выдачи",
    ["source"],  # model — персональное ранжирование, popular — холодный старт
)

CLIENT_STALENESS = Histogram(
    "client_staleness_months",
    "На сколько месяцев устарел снапшот клиента в витрине",
    buckets=(0, 1, 2, 3, 6, 12, 24),
)

PREDICTION_SECONDS = Histogram(
    "prediction_seconds",
    "Чистое время скоринга кандидатов моделью (без Postgres и HTTP)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
)


def instrument(app: FastAPI) -> None:
    """Вешает middleware тайминга и публикует /metrics."""

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        REQUEST_DURATION.labels(
            method=request.method,
            # шаблон пути (/recommend/{ncodpers}), а не конкретный URL —
            # иначе кардинальность метрики растёт с каждым новым клиентом
            handler=route.path if route else "unmatched",
            status=str(response.status_code),
        ).observe(time.perf_counter() - start)
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def init_counters(products: list[str]) -> None:
    """Регистрирует все комбинации лейблов нулями при старте сервиса.

    Без этого счётчик продукта рождается сразу с ненулевым значением при
    первом попадании в выдачу, и increase() в Prometheus не видит этого
    прироста (серия для него — прямая линия): первые события каждого
    продукта после деплоя молча теряются на дашборде.
    """
    for source in ("model", "popular"):
        RECOMMEND_REQUESTS.labels(source=source)
        for product in products:
            RECOMMENDATIONS.labels(product=product, source=source)


def record_result(result) -> None:
    """Учитывает выдачу рекомендаций (RecommendResult) в счётчиках."""
    RECOMMEND_REQUESTS.labels(source=result.source).inc()
    if result.stale_months is not None:
        CLIENT_STALENESS.observe(result.stale_months)
    for item in result.items:
        RECOMMENDATIONS.labels(product=item.product, source=result.source).inc()
