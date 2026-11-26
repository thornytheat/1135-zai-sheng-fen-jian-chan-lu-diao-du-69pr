from csv import DictReader
from .contracts import Bunker, PlacementFlag
from .engine import allocate, tally
from .line_switch import (
    CHANGEOVER_LOSS_KG,
    allocate_with_line_options,
    build_shift_options,
    operating_windows,
    plan_line_changes,
    plan_shift,
    within_window,
)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return tuple(
            Bunker(
                row["bunker"],
                row["material"],
                int(row["capacity_kg"]),
                int(row["occupied_kg"]),
                int(row["sweep_loss_kg"]),
            )
            for row in DictReader(source)
        )


def evidence(book):
    records = [
        {
            "料批": row.lot,
            "去向": row.target,
            "产线": row.line or row.original_line,
            "原产线": row.original_line,
            "是否换线": "是" if row.switched else "否",
            "换线": "是" if row.switched else "否",
            "换线标记": row.switched,
            "预留（kg）": row.kg,
            "净预留（kg）": row.kg,
            "清线损耗（kg）": row.changeover_loss_kg,
            "换线损耗（kg）": row.changeover_loss_kg,
            "状态": row.flag,
            "依据": row.rule,
        }
        for row in book.events
    ]
    return {
        "title": "仓位分派",
        "summary": (
            f"预留 {book.used:,}/{book.limit:,} 公斤，"
            f"换线 {book.changeovers} 批、清线损耗 {book.changeover_loss_kg} 公斤"
        ),
        "records": records,
        "仓位剩余": book.remaining,
        "仓位预留": tally(book),
        "换线批次": book.changeovers,
        "换线料批": list(book.switched_lots),
        "清线损耗（kg）": book.changeover_loss_kg,
        "换线损耗（kg）": book.changeover_loss_kg,
        "净预留（kg）": book.used,
    }
