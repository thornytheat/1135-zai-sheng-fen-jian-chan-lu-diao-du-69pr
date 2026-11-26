from dataclasses import dataclass
from math import fsum


@dataclass(frozen=True)
class LotBalance:
    lot: str
    net_kg: int
    warehoused: int
    forecast_unassigned: int
    expected_residue: int
    changeover_loss_kg: int = 0

    def __post_init__(self):
        if min(self.warehoused, self.forecast_unassigned, self.expected_residue) < 0:
            raise ValueError("分拣质量对账出现负数")
        if (
            self.warehoused + self.forecast_unassigned + self.expected_residue
            != self.net_kg
        ):
            raise ValueError("料批质量未守恒")

    @property
    def cleaning_loss_kg(self):
        return self.changeover_loss_kg

    @property
    def changeover_loss(self):
        return self.changeover_loss_kg

    @property
    def cleaning_loss(self):
        return self.changeover_loss_kg

    @property
    def loss_kg(self):
        return self.changeover_loss_kg

    @property
    def loss(self):
        return self.changeover_loss_kg


@dataclass(frozen=True)
class ShiftBalance:
    lines: tuple[LotBalance, ...]
    receipt: str

    @property
    def net_kg(self):
        return sum(row.net_kg for row in self.lines)

    @property
    def reserved_kg(self):
        return sum(row.warehoused for row in self.lines)


@dataclass(frozen=True)
class DriftObservation:
    material: str
    past_count: int
    present_count: int
    past_median: float | None
    present_median: float | None

    @property
    def shift(self):
        if self.past_median is None or self.present_median is None:
            return None
        return self.present_median - self.past_median
