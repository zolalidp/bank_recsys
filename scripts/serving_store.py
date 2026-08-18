"""Витрина фичей для сервиса: одна строка на клиента в Postgres.

Сервис получает ncodpers и должен собрать ровно те же признаки, что видела
модель на обучении. Поэтому здесь фиксируются не только данные, но и типы:
после round-trip через Postgres pandas по умолчанию вернёт float64 и object
вместо float32 и category, а make_pool различает их (float приводится к
строке, category — нет), и признаки тихо разъедутся.
"""

from __future__ import annotations

import io
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psycopg2

from scripts.features import DYNAMIC_COLS, STATIC_COLS

TABLE = "bank_recsys_serving_features"

# Колонки, которые модель считает категориальными, хранятся текстом,
# кроме cod_prov_prev: он числовой в данных и приводится к строке уже
# внутри make_pool — если сохранить его текстом здесь, строковое
# представление разойдётся с обучением ("28" против "28.0").
_FLOAT_COLS = [
    "age_prev",
    "antiguedad_prev",
    "indrel_prev",
    "ind_actividad_cliente_prev",
    "renta_prev",
    "cod_prov_prev",
]
_TEXT_COLS = STATIC_COLS + [
    f"{c}_prev"
    for c in DYNAMIC_COLS
    if c not in ("age", "antiguedad", "indrel", "ind_actividad_cliente", "renta", "cod_prov")
]


@contextmanager
def connect():
    conn = psycopg2.connect(
        host=os.environ["DB_DESTINATION_HOST"],
        port=os.environ["DB_DESTINATION_PORT"],
        dbname=os.environ["DB_DESTINATION_NAME"],
        user=os.environ["DB_DESTINATION_USER"],
        password=os.environ["DB_DESTINATION_PASSWORD"],
    )
    try:
        yield conn
    finally:
        conn.close()


def _column_types(serving: pd.DataFrame, products: list[str]) -> dict[str, str]:
    types = {"ncodpers": "integer", "month_id": "integer", "month": "smallint",
             "n_owned_prev": "smallint"}
    for p in products:
        types[f"{p}_owned_prev"] = "smallint"
    for c in _FLOAT_COLS:
        types[c] = "real"
    for c in _TEXT_COLS:
        types[c] = "text"
    missing = set(serving.columns) - set(types)
    if missing:
        raise ValueError(f"не задан тип для колонок: {sorted(missing)}")
    return types


def write_serving_table(serving: pd.DataFrame, products: list[str]) -> int:
    """Пересоздаёт таблицу и грузит витрину целиком через COPY."""
    types = _column_types(serving, products)
    cols = list(serving.columns)
    ddl_cols = ",\n  ".join(
        f"{c} {types[c]}" + (" primary key" if c == "ncodpers" else "") for c in cols
    )

    buf = io.StringIO()
    serving.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"drop table if exists {TABLE}")
            cur.execute(f"create table {TABLE} (\n  {ddl_cols}\n)")
            cur.copy_expert(
                f"copy {TABLE} ({', '.join(cols)}) from stdin with (format csv, null '\\N')",
                buf,
            )
            cur.execute(f"select count(*) from {TABLE}")
            n = cur.fetchone()[0]
        conn.commit()
    return n


def _restore_dtypes(df: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """Возвращает типы к тем, что были на обучении (см. докстринг модуля)."""
    for c in _FLOAT_COLS:
        df[c] = df[c].astype("float32")
    for c in _TEXT_COLS:
        df[c] = df[c].astype("category")
    for p in products:
        df[f"{p}_owned_prev"] = df[f"{p}_owned_prev"].astype("int8")
    df["n_owned_prev"] = df["n_owned_prev"].astype("int8")
    df["month"] = df["month"].astype("int8")
    df["ncodpers"] = df["ncodpers"].astype("int32")
    df["month_id"] = df["month_id"].astype("int32")
    return df


def fetch_clients(ncodpers: list[int], products: list[str]) -> pd.DataFrame:
    """Достаёт снапшоты клиентов. Отсутствующие id просто не попадут в ответ —
    их обрабатывает эндпоинт холодного старта."""
    with connect() as conn:
        df = pd.read_sql(
            f"select * from {TABLE} where ncodpers = any(%(ids)s)",
            conn,
            params={"ids": list(ncodpers)},
        )
    return _restore_dtypes(df, products) if len(df) else df


_LATEST_CACHE: dict[str, tuple[float, int]] = {}
_LATEST_TTL_SEC = 3600  # витрина обновляется раз в месяц, часа TTL достаточно


def latest_month_id() -> int:
    """Самый свежий месяц в витрине — по нему определяется устаревание
    данных конкретного клиента. Кешируется: значение меняется только при
    перезаливке витрины, а запрос дёргается на каждый recommend."""
    import time

    hit = _LATEST_CACHE.get(TABLE)
    if hit and time.monotonic() - hit[0] < _LATEST_TTL_SEC:
        return hit[1]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"select max(month_id) from {TABLE}")
        value = cur.fetchone()[0]
    _LATEST_CACHE[TABLE] = (time.monotonic(), value)
    return value
