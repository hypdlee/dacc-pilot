# GATES: DA-CNN coordinated-corrections pilot codebase

Task: end-to-end, runnable, deployable implementation of the proposal
"Learning coordinated convolutional corrections" (WRN-28-4, CIFAR-100 pilot).
Run from the repository root. Ledger: 9 of 9 checked.

- [x] G1 Package imports cleanly
  CHECK: python -c "import dacc, dacc.config, dacc.data, dacc.model, dacc.controller, dacc.targets, dacc.analysis, dacc.train, dacc.routers, dacc.evaluate, dacc.profiling, dacc.deploy, dacc.pipeline, dacc.aggregate; print('ok')"
  EXPECT: ok
  EVIDENCE: printed "ok"; pyflakes on dacc/ scripts/ tests/ reports no issues

- [x] G2 All correctness tests pass (Section 14 checks + arithmetic)
  CHECK: python -m pytest -q tests 2>&1 | tail -1
  EXPECT: passed
  EVIDENCE: "28 passed in 7.17s"

- [x] G3 Appendix B defaults match the proposal exactly
  CHECK: python -m pytest -q tests/test_config.py 2>&1 | tail -1
  EXPECT: passed
  EVIDENCE: "3 passed in 0.06s" (also loads every shipped configs/*.yaml)

- [x] G4 Full pipeline (every stage) runs end to end on the smoke config
  CHECK: python -m dacc.pipeline --config configs/smoke.yaml --seed 11 --stages all --force > /tmp/smoke.log 2>&1; echo exit=$?
  EXPECT: exit=0
  EVIDENCE: exit=0 (11 stages). Ablation run with every option on (cutmix, EMA, best_val, label smoothing,
            specialize, untied, BN recalibrate, linear eps, 2 rounds, KL+MSE, meanstd descriptor, final_test)
            also exit=0. Two runs of seed 11 gave identical results.

- [x] G5 Smoke run produced results for every method B0-B6, P1, P2, D0 plus gates and profile
  CHECK: python scripts/verify_results.py runs/smoke/seed11
  EXPECT: ALL_KEYS_PRESENT
  EVIDENCE: ALL_KEYS_PRESENT; real_vs_table_actions_agree all true

- [x] G6 Multi-seed aggregation with paired CIs works
  CHECK: python -m dacc.pipeline --config configs/smoke.yaml --seed 22 --stages all --force > /tmp/smoke22.log 2>&1 && python -m dacc.aggregate runs/smoke --out runs/smoke/aggregate.json && echo agg_ok
  EXPECT: agg_ok
  EVIDENCE: agg_ok; table with mean/SD, paired P1 differences with t intervals, three gates

- [x] G7 Deployment bundle: ONNX + TorchScript export match grouped execution, predict CLI works on an image file
  CHECK: python scripts/deploy_smoke.py runs/smoke/seed11
  EXPECT: DEPLOY_OK
  EVIDENCE: DEPLOY_OK; dense_vs_grouped 0.0, torchscript 0.0, onnx 7.2e-07 max abs diff; actions identical

- [x] G8 No placeholders left in code
  CHECK: grep -rnE "TODO|FIXME|NotImplementedError|pass  # stub|\.\.\.  #" dacc scripts tests | wc -l
  EXPECT: 0
  EVIDENCE: 0

- [x] G9 README documents install, every stage, configs, outputs, and what is NOT verified here (manual)
  EVIDENCE: README.md sections "Install", "Run", "Stages and outputs" (11 rows), "Method keys",
            "Deployment", "Configs", "Implementation choices the proposal leaves open" (8 items),
            "Verified here vs not verified", "Not included"
