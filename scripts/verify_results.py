"""Check that a finished seed directory contains every required result."""

import json
import sys
from pathlib import Path

REQUIRED_METHODS = ["B0_static", "B1_capacity", "B2_best_fixed", "B3_soft_dynamic", "B4_hard_routing",
                    "B5_additive", "B5_additive_wide", "B6_sequential", "B6_sequential_wide",
                    "P1_joint", "P2_bilinear", "P1_alternated", "B4_extra_budget", "B0_extra_budget"]
REQUIRED_FILES = ["config.json", "versions.json", "splits.json", "run_info.json",
                  "metrics/static.json", "metrics/b1.json", "metrics/bank.json", "metrics/collect_check.json",
                  "metrics/headroom_r0.json", "metrics/controllers.json", "metrics/routers.json",
                  "metrics/results.json", "metrics/alternate.json", "metrics/profile.json",
                  "metrics/timings.json", "targets/route_r0.pt", "targets/val_r0.pt",
                  "deploy/bundle.pt", "deploy/model.onnx", "deploy/model_ts.pt", "deploy/export_check.json",
                  "logs/metrics.jsonl"]


def main(seed_dir: str) -> int:
    d = Path(seed_dir)
    missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
    res = json.loads((d / "metrics/results.json").read_text())
    missing += [f"method:{m}" for m in REQUIRED_METHODS if m not in res["methods"]]
    for k in ("D0_loss_oracle", "D0_accuracy_oracle", "recovery_P1_vs_B4"):
        if k not in res["diagnostics"]:
            missing.append(f"diagnostic:{k}")
    for g in ("headroom", "predictability", "coordination", "shared_path"):
        if g not in res["gates"]:
            missing.append(f"gate:{g}")
    for k in ("learned", "best_fixed", "uniform_random", "shuffled_between_images"):
        if k not in res["interventions_P1"]:
            missing.append(f"intervention:{k}")
    prof = json.loads((d / "metrics/profile.json").read_text())
    for k in ("params", "macs", "latency", "target_collection"):
        if k not in prof:
            missing.append(f"profile:{k}")
    if not all(res["real_vs_table_actions_agree"].values()):
        missing.append("real-vs-table disagreement")
    if missing:
        print("MISSING:", missing)
        return 1
    print("ALL_KEYS_PRESENT")
    for m in REQUIRED_METHODS:
        v = res["methods"][m]
        print(f"  {m:22s} acc {v['accuracy']:7.2f}  loss {v['mean_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
