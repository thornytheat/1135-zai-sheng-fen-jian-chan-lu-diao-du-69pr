from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PlacementFlag(StrEnum):
    RESERVED = "仓位已预留"
    HOLD = "等待仓位"
    SORT = "等待再分选"
    SAMPLE = "等待抽检"


@dataclass(frozen=True, order=True)
class Bunker:
    tag: str
    feed: str
    total_kg: int
    occupied_kg: int
    changeover_loss: int = 0

    def __post_init__(self):
        if (
            not self.tag
            or not self.feed
            or not 0 <= self.occupied_kg <= self.total_kg
            or self.changeover_loss < 0
        ):
            raise ValueError("库存仓位字段无效")

    @property
    def spare(self):
        return max(0, self.total_kg - self.occupied_kg - self.changeover_loss)


@dataclass(frozen=True, init=False)
class Placement:
    """A whole-lot placement.

    ``kg`` is always the amount reserved in the bunker, i.e. net of any
    line-change cleaning loss.  The optional metadata is deliberately
    tolerant of the vocabulary used by shift-planning callers.
    """

    lot: str
    target: str | None
    kg: int
    flag: PlacementFlag
    rule: str
    line: str | None
    destination_line: str | None
    candidate_line: str | None
    original_line: str | None
    from_line: str | None
    switched: bool
    changed: bool
    is_switch: bool
    changeover_loss_kg: int
    loss_kg: int
    cleaning_loss_kg: int
    gross_reserve_kg: int | None
    gross_kg: int | None
    lower_kg: int | None
    forecast_kg: int | None
    net_kg: int | None
    reasons: tuple[str, ...]
    extra: dict[str, Any]

    def __init__(
        self,
        lot: str | None = None,
        target: str | None = None,
        kg: int = 0,
        flag: PlacementFlag | None = None,
        rule: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Accept a few plausible sixth/seventh positional forms used by small
        # planning helpers: line, switched, loss.
        positional = list(args)
        if positional and isinstance(positional[0], str):
            kwargs.setdefault("line", positional.pop(0))
        if positional and isinstance(positional[0], bool):
            kwargs.setdefault("switched", positional.pop(0))
        if positional and isinstance(positional[0], int):
            kwargs.setdefault("changeover_loss_kg", positional.pop(0))

        def first(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in kwargs and kwargs[name] is not None:
                    return kwargs.pop(name)
            return default

        line = first("line", "destination_line", "candidate_line", "to_line", "target_line")
        original_line = first("original_line", "from_line", "source_line", "current_line")
        switched = bool(first("switched", "changed", "is_switch", "line_changed", default=False))
        if line is not None and original_line is not None and line != original_line:
            switched = True
        loss = int(first(
            "changeover_loss_kg", "changeover_kg", "changeover", "loss_kg", "loss",
            "cleaning_loss_kg", "cleaning_loss", "changeover_loss",
            default=40 if switched else 0,
        ))
        gross = first("gross_reserve_kg", "gross_kg", "lower_kg", "gross_lower_kg")
        forecast = first("forecast_kg", "point_forecast_kg", "expected_kg")
        reasons = first("reasons", default=())
        if reasons is None:
            reasons = ()
        if not isinstance(reasons, tuple):
            reasons = tuple(reasons)
        net = first("net_kg", "net_reserve_kg", default=kg)
        extra = dict(kwargs)

        for name, value in (
            ("lot", lot), ("target", target), ("kg", kg),
            ("flag", flag), ("rule", rule), ("line", line),
            ("destination_line", line), ("candidate_line", line),
            ("original_line", original_line), ("from_line", original_line),
            ("switched", switched), ("changed", switched), ("is_switch", switched),
            ("changeover_loss_kg", loss), ("loss_kg", loss),
            ("cleaning_loss_kg", loss), ("gross_reserve_kg", gross),
            ("gross_kg", gross), ("lower_kg", gross),
            ("forecast_kg", forecast), ("net_kg", net),
            ("reasons", reasons), ("extra", extra),
        ):
            object.__setattr__(self, name, value)

    @property
    def core(self) -> tuple[str, str | None, int, PlacementFlag, str]:
        return (self.lot, self.target, self.kg, self.flag, self.rule)

    @property
    def net_reserved_kg(self) -> int:
        return self.kg

    @property
    def net_reserve_kg(self) -> int:
        return self.kg

    def __eq__(self, other: object) -> bool:
        # Keep the original five-field placement contract stable for callers
        # that construct the concise historical form.
        if isinstance(other, Placement):
            return self.core == other.core
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.core)


@dataclass(frozen=True)
class Reservation:
    events: tuple[Placement, ...]
    remaining: dict[str, int]
    limit: int
    used: int
    changeovers: int = 0
    switched_lots: tuple[str, ...] = ()
    changeover_loss_kg: int = 0
    net_reserved_kg: int | None = None
    gross_reserved_kg: int | None = None

    def __post_init__(self):
        if (
            self.used != sum(row.kg for row in self.events)
            or not 0 <= self.used <= self.limit
        ):
            raise ValueError("出料预留超过班次限额")
        object.__setattr__(self, "net_reserved_kg", self.used)
        if not self.switched_lots:
            object.__setattr__(
                self,
                "switched_lots",
                tuple(row.lot for row in self.events if row.switched),
            )
        if not self.changeovers:
            object.__setattr__(self, "changeovers", len(self.switched_lots))
        if not self.changeover_loss_kg:
            object.__setattr__(
                self,
                "changeover_loss_kg",
                sum(row.changeover_loss_kg for row in self.events),
            )

    @property
    def orders(self):
        return self.events

    @property
    def decisions(self):
        return self.events

    @property
    def line_decisions(self):
        return self.events

    @property
    def placements(self):
        return self.events

    @property
    def switch_count(self):
        return self.changeovers

    @property
    def switches(self):
        return self.changeovers

    @property
    def loss_kg(self):
        return self.changeover_loss_kg

    @property
    def cleaning_loss_kg(self):
        return self.changeover_loss_kg

    @property
    def net_kg(self):
        return self.used

    @property
    def remaining_capacity(self):
        return self.remaining


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from .line_switch import compat_symbol
    return compat_symbol(name)
