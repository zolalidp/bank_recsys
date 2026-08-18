"""Обучение и инференс модели рекомендации банковских продуктов (CatBoost)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from scripts.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_pool(df: pd.DataFrame, label: bool = True) -> Pool:
    x = df[FEATURE_COLS].copy()
    for c in CATEGORICAL_FEATURES:
        col = x[c]
        if pd.api.types.is_float_dtype(col):
            # CatBoost требует cat_features как int/str; float с NaN -> строка
            # (например cod_prov_prev), конвертация неизбежна.
            x[c] = col.astype(str)
        elif isinstance(col.dtype, pd.CategoricalDtype):
            # category dtype CatBoost понимает нативно (без Python-стрификации
            # каждого элемента) — но NaN как категория не поддерживается.
            if col.isna().any():
                x[c] = col.cat.add_categories(["missing"]).fillna("missing")
        else:
            x[c] = col.fillna("missing")
    y = df["label"].to_numpy() if label else None
    return Pool(x, label=y, cat_features=CATEGORICAL_FEATURES)


def train_catboost(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    params: dict | None = None,
    random_state: int = 42,
) -> CatBoostClassifier:
    """random_state должен приходить из meta.json, а не задаваться здесь заново —
    иначе обучение молча разъедется с остальным пайплайном при его смене."""
    default_params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        early_stopping_rounds=50,
        verbose=100,
    )
    if params:
        default_params.update(params)

    model = CatBoostClassifier(**default_params)
    train_pool = make_pool(train_df)
    eval_pool = make_pool(eval_df) if eval_df is not None else None
    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
    return model


def predict_scores(
    model: CatBoostClassifier,
    candidates_df: pd.DataFrame,
    ntree_end: int = 0,
    pool: Pool | None = None,
) -> np.ndarray:
    """ntree_end: сколько деревьев использовать (0 = все). Позволяет оценить
    MAP@7 на разном числе итераций без переобучения — нужно потому, что
    early stopping в CatBoost идёт по AUC, а он с MAP@7 не согласован
    (наблюдали рост AUC при падении MAP@7).

    pool: готовый Pool, чтобы не пересобирать его на каждый срез (сборка на
    19M строк валидации заметно дороже самого предсказания).
    """
    if pool is None:
        pool = make_pool(candidates_df, label=False)
    return model.predict_proba(pool, ntree_end=ntree_end)[:, 1]


def rank_topk(candidates_df: pd.DataFrame, scores: np.ndarray, k: int = 7) -> dict[int, list[str]]:
    """ncodpers -> топ-k продуктов по убыванию score."""
    tmp = pd.DataFrame(
        {
            "ncodpers": candidates_df["ncodpers"].to_numpy(),
            "product": candidates_df["product"].astype(str).to_numpy(),
            "score": scores,
        }
    )
    tmp = tmp.sort_values(["ncodpers", "score"], ascending=[True, False])

    result: dict[int, list[str]] = defaultdict(list)
    for ncodpers, product in zip(tmp["ncodpers"].to_numpy(), tmp["product"].to_numpy()):
        bucket = result[ncodpers]
        if len(bucket) < k:
            bucket.append(product)
    return dict(result)
