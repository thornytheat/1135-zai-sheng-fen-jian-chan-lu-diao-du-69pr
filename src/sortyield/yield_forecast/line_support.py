"""Candidate-line historical operating-condition support checks."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from math import expm1, log1p
from typing import Any, Iterable, Mapping


def _model_vector(*args):
    from .engine import model_vector

    return model_vector(*args)


def _get(row: Any, *names: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, Mapping):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return default
    for name in names:
        if hasattr(row, name):
            value = getattr(row, name)
            if value is not None:
                return value
    return default


def cutoff_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def record_line(row: Any) -> str:
    line = _get(row, "line", "production_line", "line_name")
    if line is not None:
        return str(line)
    features = _get(row, "features")
    if features:
        return "涡电流线" if float(features[3]) >= 0.5 else "光选线"
    raise ValueError("无法识别历史产线")


def record_material(row: Any) -> str:
    value = _get(row, "material", "polymer", "fit_group", "feed")
    if not value:
        raise ValueError("历史工况缺少材质")
    return str(value)


def record_conditions(row: Any) -> tuple[float, float, float]:
    kg = _get(row, "kg", "handled_kg", "net_kg", "weight_kg")
    impurity = _get(row, "impurity", "impurity_rate")
    moisture = _get(row, "moisture", "moisture_rate")
    features = _get(row, "features")
    if kg is None and features:
        kg = expm1(float(features[0]))
    if impurity is None and features:
        impurity = features[1]
    if moisture is None and features:
        moisture = features[2]
    return float(kg), float(impurity), float(moisture)


def valid_history(history: Iterable[Any], cutoff: Any = None) -> tuple[Any, ...]:
    """Return records closed no later than the shift cutoff."""

    cut = cutoff_date(cutoff)
    result = []
    for row in history:
        closed = cutoff_date(_get(row, "closed", "closed_at", "close_date"))
        if cut is not None and closed is not None and closed > cut:
            continue
        result.append(row)
    return tuple(result)


filter_closed_history = valid_history
history_before_cutoff = valid_history


def line_windows(
    history: Iterable[Any], cutoff: Any = None
) -> dict[tuple[str, str], tuple[tuple[float, float], ...]]:
    """Strict min/max model-coordinate envelopes by material and line."""

    bounds: dict[tuple[str, str], list[list[float]]] = {}
    for row in valid_history(history, cutoff):
        key = (record_material(row), record_line(row))
        kg, impurity, moisture = record_conditions(row)
        values = (log1p(max(0.0, kg)), impurity, moisture)
        window = bounds.setdefault(
            key, [[float("inf"), -float("inf")] for _ in range(3)]
        )
        for index, value in enumerate(values):
            window[index][0] = min(window[index][0], value)
            window[index][1] = max(window[index][1], value)
    return {
        key: tuple((low, high) for low, high in window)
        for key, window in bounds.items()
    }


operating_windows = line_windows
line_envelopes = line_windows
historical_windows = line_windows


def line_limits(
    history: Iterable[Any] | None = None,
    material: str | None = None,
    line: str | None = None,
    cutoff: Any = None,
    *,
    windows: Mapping[tuple[str, str], Any] | None = None,
):
    if windows is None:
        if history is None:
            raise ValueError("需要历史工况或已计算区间")
        windows = line_windows(history, cutoff)
    if material is None or line is None:
        return windows
    window = windows.get((material, line))
    if window is None:
        return None
    return _get(window, "limits", default=window)


def smart_support(*args: Any, **kwargs: Any) -> bool:
    """Tolerant support predicate for several plausible call signatures."""

    lot = kwargs.get("lot")
    audit = kwargs.get("audit")
    history = kwargs.get("history")
    cutoff = kwargs.get("cutoff")
    material = kwargs.get("material")
    line = kwargs.get("line")
    kg = kwargs.get("kg")
    impurity = kwargs.get("impurity")
    moisture = kwargs.get("moisture")
    window = kwargs.get("window", kwargs.get("limits"))

    positional = list(args)
    # Common explicit form: within_line(lot, audit, window)
    if positional and (_get(positional[0], "lot", "kg", "net_kg") is not None):
        lot = lot or positional.pop(0)
    if positional and (
        hasattr(positional[0], "approved_samples")
        or _get(positional[0], "impurity") is not None
    ):
        audit = audit or positional.pop(0)
    if positional and isinstance(positional[0], tuple):
        window = window or positional.pop(0)
    if positional and isinstance(positional[0], str):
        line = line or positional.pop(0)
    if positional and not isinstance(positional[0], str):
        candidate = positional.pop(0)
        if history is None and isinstance(candidate, (list, tuple)):
            history = candidate
        elif window is None:
            window = candidate
    if positional and isinstance(positional[0], (date, datetime, str)):
        cutoff = cutoff or positional.pop(0)

    material = material or _get(lot, "material", "feed", "polymer")
    line = line or _get(lot, "line", "candidate_line", "destination_line")
    kg = kg if kg is not None else _get(lot, "kg", "net_kg", "input_kg")
    if impurity is None and audit is not None:
        impurity = _get(audit, "impurity")
    if moisture is None and audit is not None:
        moisture = _get(audit, "moisture")
    impurity = impurity if impurity is not None else _get(lot, "impurity")
    moisture = moisture if moisture is not None else _get(lot, "moisture")

    if window is None and history is not None:
        window = line_limits(history, material, line, cutoff)
    if window is None or kg is None:
        return False

    limits = _get(window, "limits", default=window)
    if limits is None:
        return False
    try:
        kg_bounds, impurity_bounds, moisture_bounds = limits
        if not (kg_bounds[0] <= log1p(float(kg)) <= kg_bounds[1]):
            return False
        if impurity is not None and not (
            impurity_bounds[0] <= float(impurity) <= impurity_bounds[1]
        ):
            return False
        if moisture is not None and not (
            moisture_bounds[0] <= float(moisture) <= moisture_bounds[1]
        ):
            return False
        return True
    except (TypeError, ValueError, IndexError):
        return False


within_line = smart_support
line_supported = smart_support
within_operating_range = smart_support
within_historical_conditions = smart_support
candidate_supported = smart_support
in_line_support = smart_support


def out_of_line_support(*args: Any, **kwargs: Any) -> bool:
    return not smart_support(*args, **kwargs)
