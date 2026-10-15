"""Aggregate results across training seeds (Section 15.3).

    python -m dacc.aggregate runs/pilot_cifar100 --out runs/pilot_cifar100/aggregate.json

Reports mean and standard deviation across seeds for every method, paired
per-seed differences P1 - X with a two-sided 95% Student-t interval, and the
across-seed gate decisions. Never selects the best seed. Per-image bootstrap
intervals would condition on trained models and are not used here.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import read_json, write_json

# two-sided 95% Student-t critical values, df = 1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def t95(df: int) -> float:
    if df < 1:
        return float("nan")
    return _T95[df - 1] if df <= 30 else 1.96


def paired_interval(diffs: List[float]) -> Dict[str, Any]:
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "ci95": None, "diffs": d.tolist(),
                "note": "fewer than two seeds: no interval"}
    sd = float(d.std(ddof=1))
    half = t95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half], "diffs": d.tolist()}


def aggregate(run_root: Path, metric: str = "accuracy") -> Dict[str, Any]:
    seeds = sorted(p for p in run_root.glob("seed*") if (p / "metrics" / "results.json").exists())
    if not seeds:
        raise SystemExit(f"no seed*/metrics/results.json under {run_root}")
    per_seed = {p.name: read_json(p / "metrics" / "results.json") for p in seeds}
    methods = sorted(set().union(*[set(r["methods"].keys()) for r in per_seed.values()]))
    table: Dict[str, Any] = {}
    for m in methods:
        vals = [r["methods"][m][metric] for r in per_seed.values() if m in r["methods"]]
        losses = [r["methods"][m].get("mean_loss") for r in per_seed.values() if m in r["methods"]]
        table[m] = {"n_seeds": len(vals), f"{metric}_mean": float(np.mean(vals)),
                    f"{metric}_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    "mean_loss_mean": float(np.mean([l for l in losses if l is not None])) if any(l is not None for l in losses) else None,
                    "per_seed": vals}
    paired: Dict[str, Any] = {}
    for m in methods:
        if m == "P1_joint":
            continue
        diffs = [r["methods"]["P1_joint"][metric] - r["methods"][m][metric]
                 for r in per_seed.values() if m in r["methods"] and "P1_joint" in r["methods"]]
        if diffs:
            paired[f"P1_joint - {m}"] = paired_interval(diffs)

    simpler = [m for m in ("B4_hard_routing", "B5_additive", "B5_additive_wide", "B6_sequential",
                           "B6_sequential_wide") if m in table]
    gates: Dict[str, Any] = {}
    if simpler and "P1_joint" in table:
        strongest = max(simpler, key=lambda m: table[m][f"{metric}_mean"])
        pi = paired.get(f"P1_joint - {strongest}", {})
        crit = list(per_seed.values())[0].get("gates", {}).get("coordination", {}).get("criterion_pp", 0.3)
        gates["coordination"] = {
            "strongest_simpler": strongest, "mean_advantage_pp": pi.get("mean"), "ci95": pi.get("ci95"),
            "criterion_pp": crit,
            "pass": bool(pi.get("mean") is not None and pi["mean"] >= crit and pi.get("ci95") is not None
                         and pi["ci95"][0] > 0),
            "on_fail": "drop the joint interaction claim"}
    if "B2_best_fixed" in table and "P1_joint" in table:
        wins = [r["methods"]["P1_joint"][metric] > r["methods"]["B2_best_fixed"][metric] for r in per_seed.values()]
        gates["predictability"] = {"P1_beats_B2_every_seed": all(wins), "per_seed": wins,
                                   "on_fail": "revise descriptor or stop"}
    hd = [r.get("gates", {}).get("headroom", {}).get("value_pp") for r in per_seed.values()]
    hd = [h for h in hd if h is not None]
    if hd:
        gates["headroom"] = {"per_seed_pp": hd, "mean_pp": float(np.mean(hd)),
                             "pass_every_seed": all(r.get("gates", {}).get("headroom", {}).get("pass", False)
                                                    for r in per_seed.values())}
    return {"run_root": str(run_root), "seeds": list(per_seed.keys()), "metric": metric, "methods": table,
            "paired_P1_differences": paired, "gates": gates,
            "statistics": "mean and SD across training seeds; paired t interval over seeds; no best-seed selection"}


def print_table(agg: Dict[str, Any]) -> None:
    m = agg["metric"]
    print(f"{'method':28s} {'seeds':>5s} {m + ' mean':>14s} {'sd':>8s} {'loss':>8s}")
    for k, v in sorted(agg["methods"].items(), key=lambda kv: -kv[1][f"{m}_mean"]):
        sd = v[f"{m}_sd"]
        loss = v["mean_loss_mean"]
        print(f"{k:28s} {v['n_seeds']:5d} {v[f'{m}_mean']:14.3f} {('%.3f' % sd) if sd is not None else '-':>8s} "
              f"{('%.4f' % loss) if loss is not None else '-':>8s}")
    for k, v in agg["paired_P1_differences"].items():
        ci = v["ci95"]
        print(f"{k:40s} mean {v['mean']:+.3f} pp  ci95 {('[%+.3f, %+.3f]' % tuple(ci)) if ci else '-'}")
    for k, v in agg["gates"].items():
        print(f"gate {k}: {v}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "mean_loss", "top5_accuracy"])
    args = ap.parse_args(argv)
    agg = aggregate(Path(args.run_root), args.metric)
    print_table(agg)
    write_json(Path(args.out) if args.out else Path(args.run_root) / "aggregate.json", agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
