"""Построение и заливка витрины фичей в Postgres.

    python -m scripts.build_serving_table

Берёт data/interim/monthly_with_buys.parquet, оставляет по последнему
известному месяцу на клиента (build_serving_snapshot) и перезаписывает
таблицу bank_recsys_serving_features. Запускать после каждой свежей выгрузки
данных — иначе сервис отвечает по устаревшим снапшотам (у клиентов растёт
stale_months, метрика client_staleness_months это показывает).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scripts import features as F
from scripts import serving_store


def main() -> None:
    meta = F.load_meta()
    products = F.get_products(meta)

    print("читаю interim-таблицу...")
    df = F.load_base()

    print("строю витрину (последний месяц каждого клиента)...")
    serving = F.build_serving_snapshot(df, products)

    print(f"заливаю {len(serving):,} строк в {serving_store.TABLE}...")
    n = serving_store.write_serving_table(serving, products)
    print(f"готово: {n:,} строк, свежайший month_id = {serving_store.latest_month_id()}")


if __name__ == "__main__":
    main()
