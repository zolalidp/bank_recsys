"""MAP@K метрика для ранжирующей рекомендации (см. README: MAP@7)."""

from __future__ import annotations

from typing import Sequence


def apk(actual: Sequence, predicted: Sequence, k: int = 7) -> float:
    """Average Precision@K для одного клиента.

    actual — список реально купленных продуктов в целевом месяце.
    predicted — ранжированный список рекомендованных продуктов (лучший первым).
    """
    if not actual:
        return 0.0

    predicted = predicted[:k]
    actual_set = set(actual)

    hits = 0
    score = 0.0
    for i, p in enumerate(predicted, start=1):
        if p in actual_set and p not in predicted[: i - 1]:
            hits += 1
            score += hits / i

    return score / min(len(actual_set), k)


def mapk(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 7) -> float:
    """MAP@K по всем клиентам. actual/predicted — списки списков одной длины."""
    if len(actual) != len(predicted):
        raise ValueError("actual и predicted должны быть одной длины")
    if not actual:
        return 0.0
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)
