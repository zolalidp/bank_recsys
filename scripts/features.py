"""Построение обучающего датасета для рекомендации банковских продуктов.

Формат: long-таблица (ncodpers, month_id, product) -> label, где label = 1,
если клиент купил product в этом месяце (переход владения 0 -> 1), а
кандидатами являются только продукты, которых у клиента не было в
предыдущем месяце (см. README: "продукты, которых у него ещё нет").

Все динамические признаки (активность, сегмент, доход и т.д.) берутся из
ПРЕДЫДУЩЕГО месяца, чтобы не утекала информация, синхронная с покупкой
(см. вывод EDA: "активность нужно брать в лаговой форме").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

# Статичные признаки клиента: не зависят от месяца, берутся из текущей строки.
STATIC_COLS = ["sexo", "pais_residencia", "ind_empleado"]

# Динамические признаки: берутся из предыдущего месяца (суффикс _prev).
DYNAMIC_COLS = [
    "age",
    "antiguedad",
    "indrel",
    "indrel_1mes",
    "tiprel_1mes",
    "indresi",
    "indext",
    "canal_entrada",
    "indfall",
    "cod_prov",
    "ind_actividad_cliente",
    "renta",
    "segmento",
]

CATEGORICAL_FEATURES = [
    "sexo",
    "pais_residencia",
    "ind_empleado",
    "indrel_1mes_prev",
    "tiprel_1mes_prev",
    "indresi_prev",
    "indext_prev",
    "canal_entrada_prev",
    "indfall_prev",
    "cod_prov_prev",
    "segmento_prev",
    "product",
    "month",
]

NUMERIC_FEATURES = [
    "age_prev",
    "antiguedad_prev",
    "indrel_prev",
    "ind_actividad_cliente_prev",
    "renta_prev",
    "n_owned_prev",
    "product_pop_rate",
    "affinity_score",
]


def load_meta() -> dict:
    return json.loads((DATA_DIR / "meta.json").read_text())


def get_products(meta: dict) -> list[str]:
    return [c for c in meta["target_cols"] if c not in meta["rare"]]


def load_base() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "monthly_with_buys.parquet")


def _owned_feature_names(products: list[str]) -> list[str]:
    return [f"{p}_owned_prev" for p in products]


def build_snapshot(df: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """Собирает по одной строке на (ncodpers, month_id) с лаговыми фичами,
    вектором владения продуктами на конец прошлого месяца и таргетами
    текущего месяца. Строки без валидного предыдущего месяца отбрасываются
    (inner join)."""

    prev_cols = ["ncodpers", "month_id"] + DYNAMIC_COLS + products
    prev = df[prev_cols].copy()
    prev["month_id"] += 1
    prev = prev.rename(
        columns={c: f"{c}_prev" for c in DYNAMIC_COLS}
        | {p: f"{p}_owned_prev" for p in products}
    )

    buy_cols = [f"{p}_buy" for p in products]
    current = df[["ncodpers", "month_id", "fecha_dato"] + STATIC_COLS + buy_cols]

    snap = current.merge(prev, on=["ncodpers", "month_id"], how="inner")
    snap["month"] = snap["fecha_dato"].dt.month.astype("int8")
    snap["n_owned_prev"] = snap[_owned_feature_names(products)].sum(axis=1).astype("int8")
    return snap


def build_serving_snapshot(df: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """Витрина для инференса: по одной строке на клиента с его САМЫМ СВЕЖИМ
    известным месяцем.

    Отличие от build_snapshot: там строка месяца M содержит фичи из M-1 и
    таргеты из M (мы знаем, что произошло). Здесь мы предсказываем следующий
    месяц, которого ещё нет в данных, поэтому фичи последнего известного
    месяца M становятся `_prev`, а `month` — календарный месяц M+1. Таргетов
    нет по определению.
    """
    last_idx = df.groupby("ncodpers", observed=True)["month_id"].idxmax()
    last = df.loc[last_idx]

    cols = ["ncodpers", "month_id", "fecha_dato"] + STATIC_COLS + DYNAMIC_COLS + products
    snap = last[cols].rename(
        columns={c: f"{c}_prev" for c in DYNAMIC_COLS}
        | {p: f"{p}_owned_prev" for p in products}
    )

    # предсказываем месяц, следующий за последним известным
    next_month = snap["fecha_dato"] + pd.offsets.MonthBegin(1)
    snap["month"] = next_month.dt.month.astype("int8")
    snap["month_id"] = snap["month_id"] + 1
    snap["n_owned_prev"] = snap[_owned_feature_names(products)].sum(axis=1).astype("int8")
    return snap.drop(columns=["fecha_dato"])


def compute_product_stats(
    train_snap: pd.DataFrame, products: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Считает ТОЛЬКО по train (никогда не по val — иначе утечка):

    - pop_rate[j]: базовая частота покупки продукта j СРЕДИ ВАЛИДНЫХ
      кандидатов на j (owned_prev_j == 0). Именно этот срез использует
      build_long, поэтому знаменатель должен быть тем же.
    - affinity[i, j]: P(купит j | уже владеет i), тоже только среди валидных
      кандидатов на j — матрица совместных покупок из EDA, переведённая в
      фичу. Единственный признак в датасете, который варьируется МЕЖДУ
      кандидатами одного клиента — без него различить кандидатов внутри
      клиента почти нечем (см. разбор просадки MAP@7 первой версии модели).

      Важно: если считать P(купит j | владеет i) по ВСЕМ строкам с owns_i
      (а не только по кандидатам на j), метрика искажается клиентами, которые
      уже владеют и i, и j — для них buy_j тривиально 0 не потому что не
      купили бы, а потому что уже есть. Пример на реальных данных: 100%
      владельцев nomina уже владеют nom_pens, из-за чего affinity[nomina,
      nom_pens] без этой поправки получался 0.0 вместо ожидаемого сильного
      совместного паттерна (см. EDA: пики покупок nomina/nom_pens совпадают).
    """
    n = len(products)
    owned = train_snap[_owned_feature_names(products)].to_numpy()
    buys = train_snap[[f"{p}_buy" for p in products]].to_numpy(dtype="float32")

    pop_rate = np.zeros(n, dtype="float32")
    affinity = np.zeros((n, n), dtype="float32")

    for j in range(n):
        cand_j = owned[:, j] == 0  # только те, кому j можно предложить
        owned_cand = owned[cand_j]
        buys_cand_j = buys[cand_j, j]

        pop_rate[j] = buys_cand_j.mean() if cand_j.any() else 0.0

        counts_i = owned_cand.sum(axis=0)  # (n,) владеют i среди кандидатов на j
        sums_i = owned_cand.T @ buys_cand_j  # (n,) из них купили j
        affinity[:, j] = np.divide(
            sums_i, counts_i, out=np.full(n, pop_rate[j], dtype="float32"), where=counts_i > 0
        )

    return pop_rate, affinity


def build_long(
    snap: pd.DataFrame,
    products: list[str],
    pop_rate: np.ndarray,
    affinity: np.ndarray,
    month_ids: list[int] | None = None,
    negative_sample_rate: float | None = None,
    random_state: int = 42,
    with_labels: bool = True,
) -> pd.DataFrame:
    """Разворачивает снапшот в long-формат: одна строка на кандидата
    (ncodpers, month_id, product), только для продуктов, которых у клиента
    не было в прошлом месяце.

    with_labels=False — режим инференса: снапшот без колонок `_buy`, колонка
    label не создаётся, сэмплирование не применяется. Это намеренно тот же
    код, что и на обучении: любое расхождение в построении фичей между
    train и проливом даёт тихую деградацию качества.

    pop_rate/affinity: статистики из compute_product_stats, посчитанные
    ТОЛЬКО на train (даже при сборке val_long) — иначе утечка из будущего.

    negative_sample_rate: доля негативов, которую оставляем, ОДИНАКОВАЯ для
    всех продуктов (позитивы сохраняются все). Именно равномерность
    принципиальна: раньше здесь сэмплировалось фиксированное соотношение
    negative:positive ВНУТРИ каждого продукта, из-за чего доля покупок у всех
    22 продуктов становилась одинаковой (6.25%), хотя в реальности она
    различается в 3000 раз (ind_cco 1.66% против ind_viv 0.0005%). Это
    стирало приор продукта — ровно тот сигнал, которым выигрывает baseline
    по популярности, и объясняло MAP@7 ниже baseline при высоком AUC.

    На валидации сэмплирование не использовать — там нужен полный набор
    кандидатов для честного ранжирования top-7.
    """

    if month_ids is not None:
        snap = snap[snap["month_id"].isin(month_ids)]

    owned_cols = _owned_feature_names(products)
    owned_matrix = snap[owned_cols].to_numpy()
    n_owned = snap["n_owned_prev"].to_numpy()

    shared_cols = [c for c in snap.columns if c not in owned_cols]
    keep_cols = [c for c in shared_cols if not c.endswith("_buy")] + owned_cols

    rng = np.random.default_rng(random_state)

    # Для каждого продукта сэмплируем негативы СРАЗУ, до конкатенации —
    # иначе на 16 месяцах x 22 продукта промежуточная таблица кандидатов
    # разрастается до ~200M+ строк и не помещается в память (проверено:
    # OOM без этого шага).
    parts = []
    for pi, p in enumerate(products):
        mask = (snap[f"{p}_owned_prev"] == 0).to_numpy()
        label = snap[f"{p}_buy"].astype("int8").to_numpy()[mask] if with_labels else None

        # affinity_score: средняя P(купит p | владеет q) по продуктам q,
        # которые уже есть у клиента; для клиентов без единого продукта
        # (n_owned_prev=0) — фолбэк на базовую популярность p.
        affinity_sum = owned_matrix[mask] @ affinity[:, pi]
        n_owned_masked = n_owned[mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            affinity_score = affinity_sum / n_owned_masked
        affinity_score = np.where(n_owned_masked > 0, affinity_score, pop_rate[pi]).astype("float32")

        if negative_sample_rate is not None:
            if not with_labels:
                raise ValueError("сэмплирование негативов требует with_labels=True")
            # Доля негативов одна и та же для всех продуктов, поэтому
            # относительные приоры продуктов сохраняются точно.
            neg_idx = np.flatnonzero(label == 0)
            n_neg = int(round(len(neg_idx) * negative_sample_rate))
            keep_idx = np.concatenate(
                [np.flatnonzero(label == 1), rng.choice(neg_idx, size=n_neg, replace=False)]
            )
        else:
            keep_idx = np.arange(int(mask.sum()))

        part = snap.loc[mask, keep_cols].iloc[keep_idx].copy()
        part["product"] = p
        if with_labels:
            part["label"] = label[keep_idx]
        part["product_pop_rate"] = np.float32(pop_rate[pi])
        part["affinity_score"] = affinity_score[keep_idx]
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)
    long_df["product"] = long_df["product"].astype("category")
    long_df["month"] = long_df["month"].astype("category")
    return long_df


def sqrt_class_weights(
    long_df: pd.DataFrame, negative_sample_rate: float | None = None
) -> list[float]:
    """Вес позитивного класса как КОРЕНЬ из натурального соотношения классов.

    Полный балансирующий вес (neg/pos, что делает auto_class_weights="Balanced")
    при соотношении 1:459 раздувает редкие позитивы настолько, что модель
    переобучается на шуме; корень — компромисс между этим и полным
    игнорированием дисбаланса.

    Поправка на сэмплирование: если оставлена доля f негативов, фактическое
    соотношение в данных равно R*f, где R — натуральное. Чтобы эффективное
    соотношение в лоссе стало sqrt(R), вес позитивов = f*sqrt(R); при f=1
    (без сэмплирования) это ровно sqrt(R).
    """
    f = 1.0 if negative_sample_rate is None else negative_sample_rate
    n_pos = int((long_df["label"] == 1).sum())
    n_neg = len(long_df) - n_pos
    if n_pos == 0:
        return [1.0, 1.0]
    natural_ratio = (n_neg / f) / n_pos
    return [1.0, float(f * np.sqrt(natural_ratio))]


def actual_purchases(snap: pd.DataFrame, products: list[str], month_id: int) -> dict[int, list[str]]:
    """Для честной оценки MAP@7: клиент -> список реально купленных продуктов
    в заданном месяце (по _buy колонкам, до разворота в long)."""
    month_df = snap[snap["month_id"] == month_id]
    buy_cols = [f"{p}_buy" for p in products]
    result: dict[int, list[str]] = {}
    values = month_df[["ncodpers"] + buy_cols].to_numpy()
    for row in values:
        ncodpers = int(row[0])
        bought = [products[i] for i, v in enumerate(row[1:]) if v == 1]
        result[ncodpers] = bought
    return result
