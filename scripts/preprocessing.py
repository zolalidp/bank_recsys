"""Приведение сырой выгрузки к чистому виду.

Логика перенесена из EDA.ipynb без изменений: витрина для сервиса должна
строиться тем же кодом, что и обучающая выборка, иначе прод и обучение
разъедутся в мелочах вроде заглушки -999999 в antiguedad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "train_ver2.csv.zip"

# Почти полностью пустые (99.99% и 99.8% пропусков) и константная tipodom.
DROP_COLS = ["conyuemp", "ult_fec_cli_1t", "tipodom"]

# Приходят строками или с заглушками, приводим к числам явно.
NUMERIC_COLS = ["ind_nuevo", "indrel", "ind_actividad_cliente", "cod_prov"]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Типы, пропуски, month_id. Не трогает таргеты — их считает add_buy_targets."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    target_cols = [c for c in df.columns if c.endswith("_ult1")]
    df[target_cols] = df[target_cols].fillna(0).astype("int8")

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df["ncodpers"] = df["ncodpers"].astype("int32")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("float32")

    # -999999 это заглушка "стаж неизвестен", а не отрицательный стаж.
    antiguedad = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["antiguedad"] = antiguedad.mask(antiguedad < 0).astype("float32")

    df["renta"] = df["renta"].astype("float32")

    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"])
    df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], errors="coerce")

    # indrel_1mes приходит вперемешку как 1 / 1.0 / "P", схлопываем к одному виду.
    s = df["indrel_1mes"].astype(str).str.strip()
    df["indrel_1mes"] = s.replace(
        {"1.0": "1", "2.0": "2", "3.0": "3", "4.0": "4", "nan": None}
    )

    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")

    df["month_id"] = df["fecha_dato"].dt.year * 12 + df["fecha_dato"].dt.month
    return df


def add_buy_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Целевое событие: переход владения продуктом из 0 в 1 между соседними
    месяцами. Клиенты без предыдущего месяца получают 0 (у них таргет не
    определён) и отсеиваются позже на этапе build_snapshot."""
    target_cols = [c for c in df.columns if c.endswith("_ult1")]

    prev = df[["ncodpers", "month_id"] + target_cols].copy()
    prev["month_id"] += 1
    prev = prev.rename(columns={c: f"{c}_prev" for c in target_cols})

    out = df.merge(prev, on=["ncodpers", "month_id"], how="left")

    has_prev = out[f"{target_cols[0]}_prev"].notna()
    for c in target_cols:
        out[f"{c}_buy"] = ((out[c] == 1) & (out[f"{c}_prev"] == 0)).astype("int8")
    out.loc[~has_prev, [f"{c}_buy" for c in target_cols]] = 0

    return out.drop(columns=[c for c in out.columns if c.endswith("_prev")])


def build_interim(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Полный путь: сырой csv -> чистая таблица с таргетами."""
    return add_buy_targets(clean(load_raw(path)))
