from dataclasses import replace
from datetime import date
from pathlib import Path
from .weighbridge import service as weigh
from .contamination_sample import service as sample
from .yield_forecast import service as model
from .grade_acceptance import service as grade
from .bunker_slots import service as store
from .bunker_slots.load_study import exercise, rows as capacity_rows
from .material_recovery import service as recover

DATA = Path(__file__).resolve().parents[2] / "sample_data"
CUT = weigh.instant("2026-11-25T11:00:00+08:00")
FOLD = date(2026, 11, 1)
CHANGEOVER_LOSS_KG = store.CHANGEOVER_LOSS_KG
CLEANING_LOSS_KG = CHANGEOVER_LOSS_KG


def run(capacity=None):
    limit = (18 if capacity is None else capacity) * 1000
    journal = weigh.read(DATA / "weighbridge_shift_north.csv", CUT)
    samples, orphans = sample.from_file(DATA / "conveyor_cut_tag_log.csv", journal.lots)
    history = model.read(DATA / "separated_material_yield_history.csv")
    curves = model.per_polymer(history, FOLD, CUT.date())
    forecasts = tuple(
        model.predict(lot, samples[lot.lot], curves.get(lot.material))
        for lot in journal.lots
    )
    classifications = grade.grade_batch(journal.lots, samples, forecasts)
    bunkers = store.read(DATA / "bunker_register.csv")
    line_options = store.build_shift_options(
        journal.lots, samples, history, curves, CUT.date()
    )
    allocations = store.allocate(classifications, bunkers, limit, line_options)
    placed = {event.lot: event for event in allocations.events if event.target}
    closing_grades = []
    for item in classifications:
        event = placed.get(item.lot)
        if event is not None and event.forecast_kg is not None:
            closing_grades.append(
                replace(
                    item,
                    forecast_kg=event.forecast_kg,
                    reserve_kg=event.gross_reserve_kg or event.kg,
                )
            )
        else:
            closing_grades.append(item)
    registered = recover.close(closing_grades, allocations)
    drift = recover.material_shift(history, forecasts, CUT.date())
    balance = recover.compare_bunkers(bunkers, allocations)
    final = {order.lot: order for order in allocations.events}
    original_lines = {lot.lot: lot.line for lot in journal.lots}
    rows = []
    for item in classifications:
        location = final[item.lot]
        rows.append(
            {
                "lot": item.lot,
                "material": item.feed,
                "line": location.line or original_lines[item.lot],
                "input": item.input_kg,
                "forecast": item.forecast_kg,
                "grade": item.grade,
                "bunker": location.target,
                "placement": location.flag,
                "switched": location.switched,
                "changeover_loss_kg": location.changeover_loss_kg,
                "net_reserved_kg": location.kg,
                "why": location.rule,
            }
        )
    metrics = {
        "来料批次": len(rows),
        "来料（kg）": registered.net_kg,
        "预留（kg）": allocations.used,
        "可用仓容（kg）": sum(balance.values()),
        "换线批次": allocations.changeovers,
        "清线损耗（kg）": allocations.changeover_loss_kg,
    }
    result = {
        "metrics": metrics,
        "columns": {
            "lot": "料批",
            "material": "材质",
            "line": "入仓产线",
            "input": "来料（kg）",
            "forecast": "估算出料（kg）",
            "grade": "等级",
            "bunker": "仓位",
            "placement": "去向状态",
            "switched": "是否换线",
            "changeover_loss_kg": "清线损耗（kg）",
            "net_reserved_kg": "净预留（kg）",
            "why": "依据",
        },
        "rows": rows,
        "series": [
            {"label": item.lot, "value": item.forecast_kg} for item in classifications
        ],
        "stages": [
            weigh.evidence(journal),
            sample.evidence(samples, orphans),
            model.evidence(curves, forecasts),
            grade.evidence(classifications),
            {
                **store.evidence(allocations),
                "班次额度试算": capacity_rows(
                    exercise(
                        classifications,
                        bunkers,
                        [1000, limit, 100000],
                        line_options,
                    )
                ),
            },
            recover.evidence(registered, drift, balance),
        ],
    }
    chinese_orders = store.evidence(allocations)["records"]
    switch_summary = {
        "loss_kg_per_batch": store.CHANGEOVER_LOSS_KG,
        "changeover_loss_kg": allocations.changeover_loss_kg,
        "changeovers": allocations.changeovers,
        "switched_lots": list(allocations.switched_lots),
        "net_reserved_kg": allocations.used,
        "orders": rows,
        "tickets": chinese_orders,
    }
    result.update({
        "orders": rows,
        "shift_orders": rows,
        "当班入仓单": chinese_orders,
        "warehouse_orders": chinese_orders,
        "intake_orders": rows,
        "line_switch_plan": switch_summary,
        "switch_plan": switch_summary,
        "changeover_plan": switch_summary,
    })
    return result
