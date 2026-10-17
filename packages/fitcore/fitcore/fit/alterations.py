"""Alterations for the recommended size. Easy zones only; hard zones are never touched."""

import math

from fitcore.fit.config import FitConfig
from fitcore.schemas import Alteration, Category, FitWarning, SizeFit, Verdict


def _fmt_cm(v: float) -> str:
    return f"{v:g}"


def suggest_alterations(rec: SizeFit, category: Category, cfg: FitConfig) -> tuple[list[Alteration], list[FitWarning]]:
    hard = set(cfg.hard_zones[category])
    step = cfg.rounding_step_cm
    out: list[Alteration] = []
    warnings: list[FitWarning] = []

    for zone, z in rec.zones.items():
        # Hard zones are never altered; zones without a configured action are not alterable either.
        if zone in hard or zone not in cfg.easy_zones or zone not in cfg.actions:
            continue
        if z.verdict == Verdict.ideal:
            continue  # nothing to fix now; any wash effect is reported as a warning
        acts = cfg.actions[zone]
        need = z.target_ideal_cm - z.ease_now_cm  # positive: needs more length/room
        if need > 0:
            act, cap = acts.increase, acts.increase.max_cm
        else:
            act = acts.decrease
            # Do not remove so much that washing pushes the zone below its minimum.
            keep = max(0.0, z.ease_after_wash_cm - z.target_min_cm)
            cap = min(act.max_cm, keep) if z.verdict in (Verdict.long, Verdict.loose) else act.max_cm
        amount = min(abs(need), cap)
        amount = math.floor(amount / step + 1e-9) * step
        if amount < cfg.min_alteration_cm:
            continue
        out.append(Alteration(zone=zone, action=act.action, cm=amount, text=act.text.format(cm=_fmt_cm(amount))))
        if abs(need) - amount > step:
            warnings.append(FitWarning(
                kind="size", zone=zone,
                message=f"A tailor can change the {zone.value.replace('_', ' ')} by about {_fmt_cm(amount)} cm at most; {abs(need) - amount:.1f} cm of the gap would remain.",
            ))
    return out, warnings
