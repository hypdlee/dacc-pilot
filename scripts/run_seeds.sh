#!/usr/bin/env bash
# Run the full pipeline for several seeds, then aggregate.
#   bash scripts/run_seeds.sh configs/pilot_cifar100.yaml 11 22 33
set -euo pipefail
CFG="$1"; shift
SEEDS="${*:-11 22 33}"
for s in $SEEDS; do
  python -m dacc.pipeline --config "$CFG" --seed "$s" --stages all
done
NAME=$(python -c "import sys; from dacc.config import load_config; print(load_config(sys.argv[1]).run_name)" "$CFG")
ROOT=$(python -c "import sys; from dacc.config import load_config; print(load_config(sys.argv[1]).out_root)" "$CFG")
python -m dacc.aggregate "$ROOT/$NAME" --out "$ROOT/$NAME/aggregate.json"
