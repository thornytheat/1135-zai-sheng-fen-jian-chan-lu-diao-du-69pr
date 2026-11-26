from .contracts import Placement, PlacementFlag, Reservation
from ..grade_acceptance.contracts import Grade
from .line_switch import (
    CHANGEOVER_LOSS_KG,
    allocate_with_line_options,
    group_options,
    operating_windows,
    plan_line_changes,
    plan_shift,
    within_window,
)


def allocate(grades, bins, limit, line_options=None, *extra_args, **options):
    # The historical three-argument API remains the deterministic least-slack
    # allocator.  Any explicit candidate-line input opts into whole-shift
    # optimisation, including change-over losses and displacement trade-offs.
    candidate_input = line_options
    changeover_loss_kg = options.get(
        "changeover_loss_kg",
        options.get("loss_kg", options.get("cleaning_loss_kg", CHANGEOVER_LOSS_KG)),
    )
    if isinstance(candidate_input, int) and not isinstance(candidate_input, bool):
        changeover_loss_kg = candidate_input
        candidate_input = None
        options["optimize"] = True
    if candidate_input is None and extra_args:
        candidate_input = extra_args[0]
    for name in (
        "alternatives", "candidates", "switches", "line_candidates",
        "switch_options", "options",
    ):
        if candidate_input is None and name in options:
            candidate_input = options[name]
    if isinstance(candidate_input, int) and not isinstance(candidate_input, bool):
        changeover_loss_kg = candidate_input
        candidate_input = None
        options["optimize"] = True
    # The shift contract now asks for the class-wide maximum net reservation,
    # so plain three-argument allocation also uses the exact whole-shift
    # solver.  It retains the least-slack bin tie policy for each chosen set.
    return allocate_with_line_options(
        grades,
        bins,
        limit,
        candidate_input,
        original_lines=options.get("original_lines"),
        changeover_loss_kg=changeover_loss_kg,
    )

def least_slack_allocate(grades, bins, limit):
    """Historical descending-size, least-slack first-fit policy."""
    if limit < 0 or len({bucket.tag for bucket in bins}) != len(bins):
        raise ValueError("总额度无效或仓位编号重复")
    remaining = {bucket.tag: bucket.spare for bucket in bins}
    total = limit
    entries = []
    for result in sorted(grades, key=lambda row: (-row.reserve_kg, row.lot)):
        if result.grade != Grade.CLEAN:
            flag = (
                PlacementFlag.SORT
                if result.grade == Grade.MIXED
                else PlacementFlag.SAMPLE
            )
            entries.append(
                Placement(result.lot, None, 0, flag, "、".join(result.reasons))
            )
            continue
        if not result.reserve_kg or result.reserve_kg > total:
            entries.append(
                Placement(result.lot, None, 0, PlacementFlag.HOLD, "余量或班次额度不足")
            )
            continue
        choices = [
            bucket
            for bucket in bins
            if bucket.feed == result.feed and remaining[bucket.tag] >= result.reserve_kg
        ]
        if not choices:
            entries.append(
                Placement(
                    result.lot, None, 0, PlacementFlag.HOLD, "缺少同材质的整批仓位"
                )
            )
            continue
        bin = min(choices, key=lambda bucket: (remaining[bucket.tag], bucket.tag))
        remaining[bin.tag] -= result.reserve_kg
        total -= result.reserve_kg
        entries.append(
            Placement(
                result.lot,
                bin.tag,
                result.reserve_kg,
                PlacementFlag.RESERVED,
                "同材质优先最小剩余空间",
            )
        )
    return Reservation(
        tuple(entries), dict(sorted(remaining.items())), limit, limit - total
    )


_allocate_least_slack = least_slack_allocate


def tally(reservation):
    totals = {name: 0 for name in reservation.remaining}
    for event in reservation.events:
        if event.target is not None:
            totals[event.target] += event.kg
    return totals


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from .line_switch import compat_symbol
    return compat_symbol(name)
