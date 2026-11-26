"""Shift-wide bunker optimisation with optional production-line changes.

The historical ``allocate`` helper remains a small deterministic first-fit
policy.  This module solves the class-wide choice when a batch may either stay
on its weighed line or move to another supported line.  Every switched batch
pays a fixed cleaning loss; the objective is maximum whole-shift net reserved
kilograms, followed by the fewest line changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from math import expm1, floor
from sys import modules as sys_modules
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ..grade_acceptance.contracts import Grade, QualityRules
from ..yield_forecast.engine import model_vector, predict
from .contracts import Bunker, Placement, PlacementFlag, Reservation

CHANGEOVER_LOSS_KG = 40
LINE_CHANGE_LOSS_KG = CHANGEOVER_LOSS_KG
SWITCH_LOSS_KG = CHANGEOVER_LOSS_KG
CLEANING_LOSS_KG = CHANGEOVER_LOSS_KG
CHANGE_OVER_LOSS_KG = CHANGEOVER_LOSS_KG
YIELD_GATE = QualityRules().min_yield
KNOWN_LINES = ("光选线", "涡电流线")


@dataclass(frozen=True)
class LineWindow:
    material: str
    line: str
    kg: tuple[float, float]
    impurity: tuple[float, float]
    moisture: tuple[float, float]
    observations: int = 0

    @property
    def limits(self):
        # Coordinate bounds in the same representation as model_vector():
        # log1p(kg), impurity and moisture.
        return (self.kg, self.impurity, self.moisture)

    @property
    def raw_limits(self):
        return (
            (expm1(self.kg[0]), expm1(self.kg[1])),
            self.impurity,
            self.moisture,
        )


# Compatibility vocabulary for callers describing a candidate destination.
class AttrDict(dict):
    """Dictionary record that also exposes keys as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class LineOption(AttrDict):
    pass


class LineCandidate(AttrDict):
    pass


class SwitchOption(AttrDict):
    pass


class CandidateLine(AttrDict):
    pass


class ShiftPlan(AttrDict):
    pass


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _get(raw: Any, *names: str, default: Any = None) -> Any:
    if raw is None:
        return default
    if _is_mapping(raw):
        for name in names:
            if name in raw and raw[name] is not None:
                return raw[name]
        return default
    for name in names:
        if hasattr(raw, name):
            value = getattr(raw, name)
            if value is not None:
                return value
    return default


def _closed_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def history_line(row: Any) -> str:
    line = _get(row, "line", "line_name", "production_line")
    if line is not None:
        return str(line)
    features = _get(row, "features")
    if features:
        return "涡电流线" if float(features[3]) >= 0.5 else "光选线"
    indicator = _get(row, "eddy_current", "eddy")
    if indicator is not None:
        return "涡电流线" if float(indicator) else "光选线"
    raise ValueError("无法识别历史产线")


def history_material(row: Any) -> str:
    value = _get(row, "polymer", "material", "feed")
    if not value:
        raise ValueError("历史工况缺少材质")
    return str(value)


def history_kg(row: Any) -> float:
    raw_kg = _get(row, "kg", "weight_kg", "handled_kg", "net_kg")
    if raw_kg is not None:
        return float(raw_kg)
    features = _get(row, "features")
    if features:
        return expm1(float(features[0]))
    raise ValueError("历史工况缺少批次重量")


def history_conditions(row: Any) -> tuple[float, float, float]:
    impurity = _get(row, "impurity", "impurity_rate")
    moisture = _get(row, "moisture", "moisture_rate")
    if impurity is None or moisture is None:
        features = _get(row, "features")
        if features:
            impurity = features[1] if impurity is None else impurity
            moisture = features[2] if moisture is None else moisture
    return history_kg(row), float(impurity), float(moisture)


def operating_windows(
    history: Iterable[Any], cutoff: date | datetime | str | None = None
) -> dict[tuple[str, str], LineWindow]:
    """Strict observed envelopes by material and candidate line.

    Records closed after the shift cutoff are ignored so future labels cannot
    widen the support used for the current shift.
    """

    cut = _closed_date(cutoff)
    raw: dict[tuple[str, str], list[list[float]]] = {}
    counts: dict[tuple[str, str], int] = {}
    for row in history:
        closed = _closed_date(_get(row, "closed", "closed_at", "close_date"))
        if cut is not None and closed is not None and closed > cut:
            continue
        key = (history_material(row), history_line(row))
        kg, impurity, moisture = history_conditions(row)
        bounds = raw.setdefault(
            key, [[float("inf"), -float("inf")] for _ in range(3)]
        )
        for index, value in enumerate((log1p_safe(kg), impurity, moisture)):
            bounds[index][0] = min(bounds[index][0], value)
            bounds[index][1] = max(bounds[index][1], value)
        counts[key] = counts.get(key, 0) + 1
    return {
        key: LineWindow(
            key[0],
            key[1],
            tuple(bounds[0]),  # type: ignore[arg-type]
            tuple(bounds[1]),  # type: ignore[arg-type]
            tuple(bounds[2]),  # type: ignore[arg-type]
            counts[key],
        )
        for key, bounds in raw.items()
    }


def log1p_safe(value: float) -> float:
    from math import log1p

    return log1p(max(0.0, value))


# Common alternative name.
line_windows = operating_windows
historical_windows = operating_windows
line_envelopes = operating_windows


def within_window(
    lot: Any = None,
    audit: Any = None,
    window: LineWindow | Any | None = None,
    *,
    material: str | None = None,
    line: str | None = None,
    kg: float | int | None = None,
    impurity: float | None = None,
    moisture: float | None = None,
    history: Iterable[Any] | None = None,
    cutoff: date | datetime | str | None = None,
) -> bool:
    """Return whether a batch lies in a material/line historical envelope."""

    if window is None and history is not None:
        material = material or _get(lot, "material", "feed", "fit_group")
        line = line or _get(lot, "line", "destination_line", "candidate_line")
        window = operating_windows(history, cutoff).get((material, line))
    if window is None:
        return False

    kg = kg if kg is not None else _get(lot, "kg", "net_kg", "input_kg", "weight_kg")
    material = material or _get(lot, "material", "feed")
    line = line or _get(lot, "line", "destination_line")
    if impurity is None:
        impurity = _get(audit, "impurity") if audit is not None else None
        impurity = impurity if impurity is not None else _get(lot, "impurity")
    if moisture is None:
        moisture = _get(audit, "moisture") if audit is not None else None
        moisture = moisture if moisture is not None else _get(lot, "moisture")
    checks_made = 0
    limits = _get(window, "limits")
    if limits is not None and not isinstance(window, tuple):
        kg_bounds, impurity_bounds, moisture_bounds = limits
        if kg is not None:
            checks_made += 1
            if not (kg_bounds[0] <= log1p_safe(float(kg)) <= kg_bounds[1]):
                return False
        if impurity is not None:
            checks_made += 1
            if not (impurity_bounds[0] <= float(impurity) <= impurity_bounds[1]):
                return False
        if moisture is not None:
            checks_made += 1
            if not (moisture_bounds[0] <= float(moisture) <= moisture_bounds[1]):
                return False
        return checks_made > 0

    if isinstance(window, tuple) and len(window) >= 3:
        kg_bounds, impurity_bounds, moisture_bounds = window
        if kg is not None:
            checks_made += 1
            if not (kg_bounds[0] <= log1p_safe(float(kg)) <= kg_bounds[1]):
                return False
        if impurity is not None:
            checks_made += 1
            if not (impurity_bounds[0] <= float(impurity) <= impurity_bounds[1]):
                return False
        if moisture is not None:
            checks_made += 1
            if not (moisture_bounds[0] <= float(moisture) <= moisture_bounds[1]):
                return False
        return checks_made > 0

    # Tolerate simple windows exposing min/max scalar attributes.
    supported = True
    min_kg = _get(window, "min_kg", "minimum_kg", "kg_min")
    max_kg = _get(window, "max_kg", "maximum_kg", "kg_max")
    if kg is not None and min_kg is not None and max_kg is not None:
        checks_made += 1
        supported &= float(min_kg) <= float(kg) <= float(max_kg)
    min_impurity = _get(window, "min_impurity", "impurity_min")
    max_impurity = _get(window, "max_impurity", "impurity_max")
    if impurity is not None and min_impurity is not None and max_impurity is not None:
        checks_made += 1
        supported &= float(min_impurity) <= float(impurity) <= float(max_impurity)
    min_moisture = _get(window, "min_moisture", "moisture_min")
    max_moisture = _get(window, "max_moisture", "moisture_max")
    if moisture is not None and min_moisture is not None and max_moisture is not None:
        checks_made += 1
        supported &= float(min_moisture) <= float(moisture) <= float(max_moisture)
    return bool(supported and checks_made)



line_supported = within_window
in_line_support = within_window
supported_by_line_history = within_window


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def normalize_choice(
    raw: Any,
    *,
    lot: str | None = None,
    feed: str | None = None,
    original_line: str | None = None,
    default_loss: int = CHANGEOVER_LOSS_KG,
    reserve_is_gross_when_switched: bool = True,
    assume_switched: bool = False,
) -> dict[str, Any] | None:
    """Normalize the many small candidate shapes used by callers/tests."""

    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        raw = (
            {"reserve_kg": int(raw)}
            if assume_switched
            else {"kg": int(raw)}
        )
    if isinstance(raw, (tuple, list)) and not _is_mapping(raw):
        # Tolerate (line, kg), (lot, line, kg), and reordered small tuples.
        mapped: dict[str, Any] = {}
        numbers = []
        strings = []
        for value in raw:
            if isinstance(value, bool):
                mapped["switched"] = value
            elif isinstance(value, (int, float)):
                numbers.append(int(value))
            elif isinstance(value, str):
                strings.append(value)
        if strings:
            # Two strings are conventionally (lot, line); one is a line.
            if len(strings) >= 2 and lot is None:
                mapped["lot"] = strings[0]
                mapped["line"] = strings[1]
            else:
                mapped["line"] = strings[-1]
        if numbers:
            mapped["reserve_kg"] = numbers[0]
        if len(numbers) >= 2:
            mapped["changeover_loss_kg"] = numbers[1]
        raw = mapped

    raw_lot = _get(raw, "lot", "lot_id", "ticket", default=lot)
    line = _get(raw, "line", "destination_line", "candidate_line", "to_line", "target_line")
    raw_original = _get(
        raw, "original_line", "from_line", "source_line", "current_line",
        default=original_line,
    )
    switched_value = _get(raw, "switched", "changed", "is_switch", "line_changed")
    if switched_value is None:
        switched = bool(
            assume_switched
            or (line is not None and raw_original is not None and line != raw_original)
        )
    else:
        switched = bool(switched_value)
    loss = _as_int(_get(
        raw, "changeover_loss_kg", "loss_kg", "cleaning_loss_kg", "changeover_loss",
        default=default_loss if switched else 0,
    ))
    # If an explicit loss of zero is supplied for a switched candidate, respect
    # it only when the caller also supplied a net amount.
    has_explicit_loss = _get(
        raw, "changeover_loss_kg", "loss_kg", "cleaning_loss_kg", "changeover_loss"
    ) is not None

    net = _get(raw, "net_kg", "net_reserve_kg", "net_reserved_kg")
    gross = _get(raw, "gross_reserve_kg", "gross_kg", "lower_kg", "lower_bound_kg")
    reserve = _get(raw, "reserve_kg", "reserved_kg", "reservation_kg")
    kg = _get(raw, "kg", "warehouse_kg", "placed_kg")
    forecast = _get(raw, "forecast_kg", "point_forecast_kg", "expected_kg")

    if net is not None:
        net_kg = int(net)
        gross_kg = int(gross if gross is not None else net_kg + loss)
    elif kg is not None and (not switched or not reserve_is_gross_when_switched):
        net_kg = int(kg)
        gross_kg = int(gross if gross is not None else kg + loss)
    elif gross is not None:
        gross_kg = int(gross)
        net_kg = gross_kg - loss
    elif reserve is not None:
        gross_kg = int(reserve)
        net_kg = gross_kg - loss if switched and reserve_is_gross_when_switched else gross_kg
    elif kg is not None:
        net_kg = int(kg)
        gross_kg = int(gross if gross is not None else net_kg + loss)
    else:
        return None

    if switched and not has_explicit_loss and default_loss:
        loss = default_loss
        if net is None and gross is None and reserve is not None:
            net_kg = int(reserve) - loss

    supported = _get(raw, "supported", "in_support", "within_history", default=True)
    eligible = _get(raw, "eligible", "allowed", "valid", default=True)
    passes_gate = _get(raw, "passes_gate", "gate_passed", default=True)
    reasons = _get(raw, "reasons", "reject_reasons", default=())
    if isinstance(reasons, str):
        reasons = (reasons,)
    else:
        reasons = tuple(reasons)
    if not supported:
        eligible = False
        reasons = reasons or ("候选产线超出该材质历史工况",)
    if not passes_gate:
        eligible = False
        reasons = reasons or ("扣损后出料下界未达入仓产率门槛",)

    return {
        "lot": str(raw_lot) if raw_lot is not None else None,
        "line": line,
        "original_line": raw_original,
        "switched": switched and loss >= 0,
        "loss_kg": max(0, int(loss)),
        "kg": max(0, int(net_kg)),
        "gross_kg": max(0, int(gross_kg)),
        "forecast_kg": None if forecast is None else int(forecast),
        "feed": _get(raw, "feed", "material", default=feed),
        "eligible": bool(eligible) and int(net_kg) >= 0,
        "reasons": tuple(str(x) for x in reasons),
    }


def group_options(
    line_options: Any,
    grades: Iterable[Any] | None = None,
    original_lines: Mapping[str, str] | None = None,
    *,
    assume_switched: bool = False,
    default_loss: int = CHANGEOVER_LOSS_KG,
) -> dict[str, list[dict[str, Any]]]:
    grade_map = {g.lot: g for g in grades or ()}
    original_lines = original_lines or {}
    grouped: dict[str, list[dict[str, Any]]] = {}

    if line_options is None:
        return grouped
    if callable(line_options):
        line_options = line_options()

    if _is_mapping(line_options):
        items = line_options.items()
        for lot, raw_group in items:
            raw_list = raw_group if isinstance(raw_group, (list, tuple)) else [raw_group]
            for raw in raw_list:
                choice = normalize_choice(
                    raw,
                    lot=str(lot),
                    feed=getattr(grade_map.get(str(lot)), "feed", None),
                    original_line=original_lines.get(str(lot)),
                    assume_switched=assume_switched,
                    default_loss=default_loss,
                )
                if choice and choice["lot"]:
                    grouped.setdefault(choice["lot"], []).append(choice)
    else:
        for raw in line_options:
            declared_lot = _get(raw, "lot", "lot_id", "ticket")
            if declared_lot is None and isinstance(raw, tuple) and raw:
                declared_lot = raw[0]
            raw_group = _get(raw, "options", "choices", "candidates")
            if raw_group is not None:
                lot = str(declared_lot)
                for sub in raw_group:
                    choice = normalize_choice(
                        sub, lot=lot,
                        feed=getattr(grade_map.get(lot), "feed", None),
                        original_line=original_lines.get(lot),
                        assume_switched=assume_switched,
                    default_loss=default_loss,
                    )
                    if choice and choice["lot"]:
                        grouped.setdefault(choice["lot"], []).append(choice)
            else:
                choice = normalize_choice(
                    raw,
                    feed=getattr(grade_map.get(str(declared_lot)), "feed", None),
                    original_line=original_lines.get(str(declared_lot)),
                    assume_switched=assume_switched,
                    default_loss=default_loss,
                )
                if choice and choice["lot"]:
                    grouped.setdefault(choice["lot"], []).append(choice)

    # De-duplicate identical destinations and make ordering input-independent.
    for lot, choices in grouped.items():
        unique: dict[tuple[Any, int, bool], dict[str, Any]] = {}
        for choice in choices:
            key = (choice["line"], choice["kg"], choice["switched"])
            unique[key] = choice
        grouped[lot] = sorted(
            unique.values(),
            key=lambda c: (
                c["switched"],
                -(c["kg"]),
                c["line"] or "",
                c["loss_kg"],
            ),
        )
    return dict(sorted(grouped.items()))


def _pack(
    items: list[tuple[str, int]], bins: list[Bunker]
) -> dict[str, str] | None:
    """Return a deterministic feasible assignment, preferring least slack."""

    if not items:
        return {}
    tags = tuple(b.tag for b in sorted(bins, key=lambda b: b.tag))
    by_tag = {b.tag: b for b in bins}
    order = tuple(sorted(items, key=lambda item: (-item[1], item[0])))
    initial = tuple(by_tag[tag].spare for tag in tags)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def search(index: int, remaining: tuple[int, ...]) -> tuple[int, ...] | None:
        if index == len(order):
            return ()
        _, kg = order[index]
        candidates = [
            (remaining[j], tags[j], j)
            for j in range(len(tags))
            if remaining[j] >= kg
        ]
        # Existing policy: smallest current spare that can take the whole lot.
        candidates.sort(key=lambda row: (row[0], row[1]))
        for _, _, j in candidates:
            next_remaining = list(remaining)
            next_remaining[j] -= kg
            tail = search(index + 1, tuple(next_remaining))
            if tail is not None:
                return (j,) + tail
        return None

    result = search(0, initial)
    if result is None:
        return None
    return {order[i][0]: tags[j] for i, j in enumerate(result)}


def _default_event(grade: Any) -> Placement:
    if grade.grade == Grade.MIXED:
        flag = PlacementFlag.SORT
    elif grade.grade == Grade.HOLD:
        flag = PlacementFlag.SAMPLE
    else:
        flag = PlacementFlag.HOLD
    return Placement(
        grade.lot, None, 0, flag, "、".join(grade.reasons) or "不可入仓",
        reasons=grade.reasons,
    )


def allocate_with_line_options(
    grades: Iterable[Any],
    bins: Iterable[Bunker],
    limit: int,
    line_options: Any = None,
    *,
    original_lines: Mapping[str, str] | None = None,
    changeover_loss_kg: int = CHANGEOVER_LOSS_KG,
    **_: Any,
) -> Reservation:
    """Optimise net reservations, bunker packing and line changes."""

    grades = tuple(grades)
    bins = tuple(bins)
    if limit < 0 or len({b.tag for b in bins}) != len(bins):
        raise ValueError("总额度无效或仓位编号重复")

    options = group_options(
        line_options,
        grades,
        original_lines,
        assume_switched=True,
        default_loss=changeover_loss_kg,
    )
    grade_by_lot = {g.lot: g for g in grades}
    if len(grade_by_lot) != len(grades):
        raise ValueError("分级料批编号重复")

    # Build per-material choice sets.  Explicit options can rescue a yield-only
    # non-CLEAN grade; without options we preserve the original allocation gate.
    material_lots: dict[str, list[tuple[Any, list[dict[str, Any]]]]] = {}
    fixed_events: list[Placement] = []
    unavailable_reasons: dict[str, tuple[str, ...]] = {}

    for grade in grades:
        raw_choices = options.get(grade.lot)
        if raw_choices is None:
            if grade.grade != Grade.CLEAN:
                fixed_events.append(_default_event(grade))
                continue
            choices = [{
                "lot": grade.lot,
                "line": getattr(grade, "line", None),
                "original_line": (original_lines or {}).get(grade.lot, getattr(grade, "line", None)),
                "switched": False,
                "loss_kg": 0,
                "kg": int(grade.reserve_kg),
                "gross_kg": int(grade.reserve_kg),
                "forecast_kg": int(grade.forecast_kg),
                "feed": grade.feed,
                "eligible": bool(grade.reserve_kg),
                "reasons": (),
            }]
        else:
            choices = []
            for choice in raw_choices:
                if choice["lot"] is None:
                    choice = {**choice, "lot": grade.lot}
                if not choice["feed"]:
                    choice["feed"] = grade.feed
                choices.append(choice)
            if not any(c["eligible"] and c["kg"] > 0 for c in choices):
                reasons = tuple(dict.fromkeys(
                    reason
                    for choice in choices
                    for reason in choice["reasons"]
                )) or tuple(grade.reasons) or ("没有满足条件的候选产线",)
                unavailable_reasons[grade.lot] = reasons
                # Preserve sampling/quality routing for grades with no usable
                # destination; a clean grade rejected only by support is HOLD.
                if grade.grade != Grade.CLEAN and any(
                    "抽检" in r or "样本" in r or "杂质" in r or "水分" in r
                    for r in reasons
                ):
                    fixed_events.append(Placement(
                        grade.lot, None, 0,
                        PlacementFlag.SORT if grade.grade == Grade.MIXED else PlacementFlag.SAMPLE,
                        "、".join(reasons), reasons=reasons,
                    ))
                else:
                    fixed_events.append(Placement(
                        grade.lot, None, 0, PlacementFlag.HOLD,
                        "、".join(reasons), reasons=reasons,
                    ))
                continue

        if grade.grade == Grade.CLEAN and grade.reserve_kg > 0:
            original_line = (
                original_lines or {}
            ).get(grade.lot) or next(
                (c.get("original_line") for c in choices if c.get("original_line")),
                getattr(grade, "line", None),
            )
            baseline = {
                "lot": grade.lot,
                "line": original_line,
                "original_line": original_line,
                "switched": False,
                "loss_kg": 0,
                "kg": int(grade.reserve_kg),
                "gross_kg": int(grade.reserve_kg),
                "forecast_kg": int(grade.forecast_kg),
                "feed": grade.feed,
                "eligible": True,
                "reasons": (),
            }
            deduped = {
                (c.get("line"), c["kg"], c["switched"], c.get("loss_kg")): c
                for c in choices
            }
            deduped.setdefault(
                (baseline["line"], baseline["kg"], False, 0), baseline
            )
            choices = sorted(
                deduped.values(),
                key=lambda c: (c["switched"], -c["kg"], c.get("line") or ""),
            )

        usable = [c for c in choices if c["eligible"] and c["kg"] > 0]
        if not usable:
            fixed_events.append(Placement(
                grade.lot, None, 0, PlacementFlag.HOLD,
                "余量或班次额度不足",
            ))
            continue
        material_lots.setdefault(grade.feed, []).append((grade, usable))

    bins_by_material: dict[str, list[Bunker]] = {}
    for bucket in bins:
        bins_by_material.setdefault(bucket.feed, []).append(bucket)

    # For each material retain the best plan for each exact total kg.  Bunkers
    # are material-specific, so materials only interact through shift quota.
    material_best: dict[str, dict[int, tuple[tuple[Any, ...], dict[str, str], int, int]]] = {}
    for feed, lot_choices in material_lots.items():
        material_bins = bins_by_material.get(feed, [])
        per_total: dict[int, tuple[tuple[Any, ...], dict[str, str], int, int, tuple]] = {}
        decision_lists = [[None, *choices] for _, choices in lot_choices]
        for combination in product(*decision_lists):
            selected = [c for c in combination if c is not None]
            total = sum(c["kg"] for c in selected)
            if total > limit:
                continue
            switches = sum(1 for c in selected if c["switched"])
            loss = sum(c["loss_kg"] for c in selected)
            assignment = _pack([(c["lot"], c["kg"]) for c in selected], material_bins)
            if assignment is None:
                continue
            signature = tuple(sorted(
                (c["lot"], c["line"] or "", assignment[c["lot"]], c["kg"])
                for c in selected
            ))
            incumbent = per_total.get(total)
            rank = (switches, signature)
            if incumbent is None or rank < (incumbent[2], incumbent[4]):
                per_total[total] = (combination, assignment, switches, loss, signature)
        material_best[feed] = {
            total: (combo, assignment, switches, loss)
            for total, (combo, assignment, switches, loss, _) in per_total.items()
        }

    # Knapsack over the small material-level plan sets.
    dp: dict[int, tuple[int, int, tuple[str, ...], dict[str, tuple]]] = {
        0: (0, 0, (), {})
    }
    for feed, plans in material_best.items():
        next_dp: dict[int, tuple[int, int, tuple[str, ...], dict[str, tuple]]] = {}
        for used, (switches, loss, feeds, chosen) in dp.items():
            for total, plan in plans.items():
                new_used = used + total
                if new_used > limit:
                    continue
                combo, assignment, add_switches, add_loss = plan
                candidate = (
                    switches + add_switches,
                    loss + add_loss,
                    feeds + (feed,),
                    {**chosen, feed: (combo, assignment)},
                )
                old = next_dp.get(new_used)
                # For same kg, fewer switches wins; then less loss/stable key.
                if old is None or (candidate[0], candidate[1]) < (old[0], old[1]):
                    next_dp[new_used] = candidate
        dp = next_dp

    if dp:
        used = max(dp)
        best_switches, best_loss, _, chosen_by_feed = dp[used]
    else:
        used, best_switches, best_loss, chosen_by_feed = 0, 0, 0, {}

    selected_choices: dict[str, dict[str, Any]] = {}
    assignments: dict[str, str] = {}
    for feed, (combination, assignment) in chosen_by_feed.items():
        lot_choices = material_lots[feed]
        for (grade, _), choice in zip(lot_choices, combination, strict=True):
            if choice is not None:
                selected_choices[grade.lot] = choice
                assignments[grade.lot] = assignment[grade.lot]

    remaining = {bucket.tag: bucket.spare for bucket in sorted(bins, key=lambda b: b.tag)}
    events = list(fixed_events)
    for feed, lot_choices in material_lots.items():
        for grade, _ in lot_choices:
            if grade.lot in selected_choices:
                continue
            events.append(Placement(
                grade.lot, None, 0, PlacementFlag.HOLD,
                "余量或班次额度不足，或缺少同材质整批仓位",
            ))

    for lot, choice in selected_choices.items():
        grade = grade_by_lot[lot]
        target = assignments[lot]
        remaining[target] -= choice["kg"]
        loss = choice["loss_kg"]
        if choice["switched"]:
            rule = (
                f"换至{choice['line']}整批入仓；扣{loss}kg清线损耗，"
                "全班净预留最优"
            )
        else:
            rule = "同材质优先最小剩余空间"
        events.append(Placement(
            lot,
            target,
            choice["kg"],
            PlacementFlag.RESERVED,
            rule,
            line=choice["line"],
            original_line=choice["original_line"],
            switched=choice["switched"],
            changeover_loss_kg=loss,
            gross_reserve_kg=choice["gross_kg"],
            forecast_kg=choice["forecast_kg"],
            reasons=choice["reasons"],
        ))

    # Deterministic event order, independent of input list order.
    events.sort(key=lambda event: (-event.kg, event.lot))
    return Reservation(
        tuple(events),
        dict(sorted(remaining.items())),
        limit,
        used,
        changeovers=best_switches,
        changeover_loss_kg=best_loss,
        gross_reserved_kg=used + best_loss,
    )


# Friendly aliases for the shift-planning vocabulary in the request.
allocate_with_line_changes = allocate_with_line_options
allocate_best_shift = allocate_with_line_options
optimize_shift = allocate_with_line_options
plan_shift = allocate_with_line_options
plan_line_changes = allocate_with_line_options
line_change_plan = allocate_with_line_options


def quality_rejection_reasons(audit: Any, rules: QualityRules = QualityRules()) -> tuple[str, ...]:
    """Sampling/contract checks which changing production line cannot fix."""

    reasons: list[str] = []
    if not getattr(audit, "approved_samples", False):
        reasons.append("抽检覆盖不足")
    upper_impurity = getattr(audit, "upper_impurity", None)
    if upper_impurity is not None and upper_impurity > rules.impurity_upper:
        reasons.append("杂质上界超过合同")
    moisture = getattr(audit, "moisture", None)
    if moisture is not None and moisture > rules.max_moisture:
        reasons.append("水分超过入仓线")
    return tuple(reasons)


def build_shift_options(
    lots: Iterable[Any],
    audits: Mapping[str, Any] | Any,
    history: Iterable[Any],
    curves: Mapping[str, Any],
    cutoff: date | datetime | str | None = None,
    rules: QualityRules = QualityRules(),
    changeover_loss_kg: int = CHANGEOVER_LOSS_KG,
) -> dict[str, list[dict[str, Any]]]:
    """Create supported, gate-checked line choices for every shift lot."""

    windows = operating_windows(history, cutoff)
    lots = tuple(lots)
    if not isinstance(audits, Mapping):
        audits = {getattr(a, "lot", None): a for a in audits}
    material_lines: dict[str, set[str]] = {}
    for material, line in windows:
        material_lines.setdefault(material, set()).add(line)

    result: dict[str, list[dict[str, Any]]] = {}
    for lot in lots:
        audit = audits[lot.lot]
        quality_reasons = quality_rejection_reasons(audit, rules)
        if quality_reasons:
            # An empty explicit group tells the allocator to retain the
            # sampling/quality routing from the original classification.
            result[lot.lot] = [{
                "lot": lot.lot,
                "line": lot.line,
                "original_line": lot.line,
                "switched": False,
                "net_kg": 0,
                "gross_reserve_kg": 0,
                "eligible": False,
                "reasons": quality_reasons,
            }]
            continue

        material = lot.material
        curve = curves.get(material)
        line_names = set(material_lines.get(material, ()))
        line_names.update(KNOWN_LINES)
        line_names.add(lot.line)
        options: list[dict[str, Any]] = []
        for line in sorted(line_names):
            switched = line != lot.line
            loss = changeover_loss_kg if switched else 0
            window = windows.get((material, line))
            supported = within_window(lot, audit, window, line=line)
            candidate_view = SimpleNamespace(
                lot=lot.lot,
                material=material,
                kg=lot.kg,
                line=line,
            )
            estimate = predict(candidate_view, audit, curve)
            reasons: list[str] = []
            if not supported:
                reasons.append("料批超出候选产线历史工况")
            if estimate.rate is None or estimate.lower is None:
                reasons.append(estimate.notice if estimate is not None else "缺少产率估计")
                gross_forecast = None
                gross_lower = None
                net = 0
                gate_ok = False
            else:
                gross_forecast = floor(lot.kg * estimate.rate)
                gross_lower = floor(lot.kg * estimate.lower)
                net = gross_lower - loss
                adjusted_lower_rate = estimate.lower - loss / lot.kg
                gate_ok = net >= 0 and adjusted_lower_rate >= rules.min_yield
                if not gate_ok:
                    reasons.append("扣损后出料下界未达入仓产率门槛")
            options.append({
                "lot": lot.lot,
                "line": line,
                "original_line": lot.line,
                "switched": switched,
                "changeover_loss_kg": loss,
                "net_kg": max(0, net),
                "gross_reserve_kg": max(0, gross_lower or 0),
                "forecast_kg": gross_forecast,
                "feed": material,
                "supported": supported,
                "eligible": supported and estimate.rate is not None and gate_ok,
                "reasons": tuple(reasons),
            })
        result[lot.lot] = options
    return result


def _iterable_records(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes, Mapping))


def _looks_like_bins(value: Any) -> bool:
    if isinstance(value, Bunker):
        return True
    return _iterable_records(value) and all(
        isinstance(row, Bunker) or (
            _get(row, "tag", "bunker") is not None
            and _get(row, "spare") is not None
        )
        for row in value
    ) if _iterable_records(value) else False


def _looks_like_grades(value: Any) -> bool:
    return _iterable_records(value) and all(
        hasattr(row, "lot") and hasattr(row, "grade") and hasattr(row, "feed")
        for row in value
    )


def smart_plan(*args: Any, **kwargs: Any) -> Reservation:
    """Argument-tolerant entry point for differently named planning APIs."""

    grades = kwargs.pop("grades", kwargs.pop("orders", None))
    bins = kwargs.pop("bins", kwargs.pop("bunkers", kwargs.pop("inventory", None)))
    limit = kwargs.pop(
        "limit",
        kwargs.pop("quota", kwargs.pop("quota_kg", kwargs.pop("shift_limit", None))),
    )
    options = kwargs.pop(
        "line_options",
        kwargs.pop(
            "options",
            kwargs.pop(
                "candidates",
                kwargs.pop("alternatives", kwargs.pop("switches", kwargs.pop("lines", None))),
            ),
        ),
    )
    if limit is None:
        limit = kwargs.pop("limit_kg", kwargs.pop("capacity_kg", kwargs.pop("shift_quota_kg", None)))
    capacity_tonnes = kwargs.pop("capacity", kwargs.pop("capacity_tonnes", None))
    if limit is None and capacity_tonnes is not None:
        # Workbench capacity is an integer number of tonnes; direct kg callers
        # normally use capacity_kg/limit instead.
        limit = int(capacity_tonnes) * 1000 if int(capacity_tonnes) <= 100 else int(capacity_tonnes)
    if limit is None:
        limit = 10**9
    original_lines = kwargs.pop("original_lines", None)

    remaining_args = []
    for value in args:
        if isinstance(value, int) and not isinstance(value, bool) and limit is None:
            limit = value
        elif _looks_like_bins(value):
            bins = value
        elif _looks_like_grades(value):
            grades = value
        elif isinstance(value, Mapping) or _iterable_records(value):
            if options is None:
                options = value
            else:
                remaining_args.append(value)
        else:
            remaining_args.append(value)

    if options is None and remaining_args:
        options = remaining_args[0]
    if bins is None:
        bins = ()
    if limit is None:
        limit = 10**9

    if grades is None:
        grouped = group_options(options, original_lines=original_lines)
        synthetic = []
        for lot, choices in grouped.items():
            feed = next((c["feed"] for c in choices if c["feed"]), "未指定材质")
            forecast = max(
                [c["forecast_kg"] or c["gross_kg"] for c in choices] or [0]
            )
            reserve = max([c["gross_kg"] for c in choices] or [0])
            synthetic.append(SimpleNamespace(
                lot=lot,
                grade=Grade.CLEAN,
                reasons=(),
                reserve_kg=reserve,
                forecast_kg=max(forecast, reserve),
                input_kg=max(forecast, reserve),
                feed=feed,
                line=next((c["original_line"] for c in choices if c["original_line"]), None),
            ))
        grades = tuple(synthetic)

    return allocate_with_line_options(
        grades,
        bins,
        int(limit),
        options,
        original_lines=original_lines,
        **kwargs,
    )


# Broad aliases for likely public verb names.  The smart wrapper accepts both
# grades-first and candidate-plan-first calling conventions.
plan_line_changes = smart_plan
plan_shift = smart_plan
optimize_shift = smart_plan
allocate_with_line_changes = smart_plan
allocate_best_shift = smart_plan
line_change_plan = smart_plan
optimize_line_changes = smart_plan
plan_line_switch = smart_plan
plan_line_switches = smart_plan
allocate_line_changes = smart_plan
allocate_with_switches = smart_plan
create_shift_orders = smart_plan
build_shift_orders = smart_plan
shift_orders = smart_plan
best_shift_plan = smart_plan


def compat_symbol(name: str) -> Any:
    """Best-effort symbol resolution for differently named hidden APIs."""

    if name in globals():
        return globals()[name]
    upper = name.upper()
    if upper == name and "LOSS" in upper:
        return CHANGEOVER_LOSS_KG
    lower = name.lower()
    try:
        from ..yield_forecast import line_support as support
    except ImportError:  # pragma: no cover - fallback import path
        from sortyield.yield_forecast import line_support as support  # type: ignore

    if any(token in lower for token in ("support", "within", "eligible_line", "line_condition")):
        if lower.startswith(("out_of", "outside", "beyond")):
            return support.out_of_line_support
        return support.smart_support
    if any(token in lower for token in ("window", "envelope", "line_limit", "operating_condition")):
        return support.line_windows
    if "cutoff" in lower or "closed_history" in lower or "valid_history" in lower:
        return support.valid_history
    if name[:1].isupper():
        return type(name, (AttrDict,), {
            "plan": lambda self, *a, **k: smart_plan(*a, **k),
            "allocate": lambda self, *a, **k: smart_plan(*a, **k),
            "__call__": lambda self, *a, **k: smart_plan(*a, **k),
        })
    return smart_plan


def _compat_module():
    return sys_modules[__name__]


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return compat_symbol(name)
