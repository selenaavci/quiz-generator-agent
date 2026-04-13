from __future__ import annotations

import hashlib
from typing import Dict, Tuple, Union


DIFFICULTY_BANDS: Dict[str, Tuple[int, int]] = {
    "kolay": (1, 2), "easy": (1, 2),
    "orta": (2, 4), "medium": (2, 4), "normal": (2, 4),
    "zor": (4, 5), "hard": (4, 5),
}


def _normalize_label(label: Union[str, int, None]) -> str:
    if label is None:
        return "orta"
    return str(label).strip().lower()


def _hash_pick(label: str, idx: int, span: int) -> int:
    h = hashlib.md5(f"{label}:{idx}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(span, 1)


def _band(label: str) -> Tuple[int, int]:
    return DIFFICULTY_BANDS.get(_normalize_label(label), (2, 4))


def _coerce_int(value: Union[int, str]) -> int:
    if isinstance(value, int):
        return max(1, min(5, value))
    try:
        return max(1, min(5, int(str(value).strip())))
    except Exception as e:
        raise ValueError("not an int") from e


def difficulty_for(value: Union[int, str], idx: int) -> int:
    try:
        return _coerce_int(value)
    except ValueError:
        label = _normalize_label(value)
        lo, hi = _band(label)
        return lo + _hash_pick(label, idx, hi - lo + 1)


def difficulty_balanced(value: Union[int, str], idx: int, metrics: Dict) -> int:
    try:
        return _coerce_int(value)
    except ValueError:
        pass

    label = _normalize_label(value)
    lo, hi = _band(label)
    candidates = list(range(lo, hi + 1))

    def used(d: int) -> int:
        return int(metrics.get(f"difficulty_count_{d}", 0)) if isinstance(metrics, dict) else 0

    counts = [(d, used(d)) for d in candidates]
    min_count = min(c for _, c in counts)
    best = [d for d, c in counts if c == min_count]
    return best[_hash_pick(label, idx, len(best))]
