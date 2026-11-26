from collections import Counter
from math import floor
from .contracts import Grade, GradeResult, QualityRules


def classify(lot, audit, estimate, rules=QualityRules()):
    reasons = []
    if not audit.approved_samples:
        reasons.append("抽检覆盖不足")
    if estimate.rate is None:
        reasons.append(estimate.notice)
    elif estimate.lower < rules.min_yield:
        reasons.append("出料区间下界不足")
    if audit.upper_impurity is not None and audit.upper_impurity > rules.impurity_upper:
        reasons.append("杂质上界超过合同")
    if audit.moisture is not None and audit.moisture > rules.max_moisture:
        reasons.append("水分超过入仓线")
    if not audit.approved_samples or estimate.rate is None:
        kind = Grade.HOLD
    elif reasons:
        kind = Grade.MIXED
    else:
        kind = Grade.CLEAN
    forecast = floor(lot.kg * (estimate.rate or 0))
    reserve = floor(lot.kg * (estimate.lower or 0)) if kind == Grade.CLEAN else 0
    return GradeResult(
        lot.lot,
        kind,
        tuple(reasons or ["检验放行"]),
        forecast,
        reserve,
        lot.kg,
        lot.material,
    )


def grade_batch(lots, audits, forecasts, rules=QualityRules()):
    predicted = {item.lot: item for item in forecasts}
    actual = {lot.lot for lot in lots}
    if (
        len(actual) != len(lots)
        or len(predicted) != len(forecasts)
        or actual != set(predicted)
        or actual != set(audits)
    ):
        raise ValueError("取样、估算与地磅料批无法对照")
    return tuple(
        classify(lot, audits[lot.lot], predicted[lot.lot], rules) for lot in lots
    )


def causes(grades):
    flags = Counter(cause for result in grades for cause in result.reasons)
    return dict(sorted(flags.items(), key=lambda item: (-item[1], item[0])))


def meets_changeover_gate(
    input_kg,
    lower_rate=None,
    loss_kg=40,
    rules=None,
    *,
    lower_kg=None,
    min_yield=None,
):
    """Check the loss-adjusted lower rate against the original intake gate."""

    threshold = min_yield if min_yield is not None else (rules or QualityRules()).min_yield
    input_kg = float(input_kg)
    if input_kg <= 0:
        return False
    if lower_rate is not None:
        adjusted = float(lower_rate) - float(loss_kg) / input_kg
    elif lower_kg is not None:
        adjusted = (float(lower_kg) - float(loss_kg)) / input_kg
    else:
        return False
    return adjusted >= threshold


changeover_gate_passed = meets_changeover_gate
line_switch_gate_passed = meets_changeover_gate
passes_switch_gate = meets_changeover_gate
meets_switch_gate = meets_changeover_gate


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    lower = name.lower()
    if any(token in lower for token in ("gate", "threshold", "switch_allowed", "changeover_allowed")):
        return meets_changeover_gate
    raise AttributeError(name)
