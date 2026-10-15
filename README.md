# Coordinated convolutional corrections: pilot implementation

Code for the research proposal *Learning coordinated convolutional corrections* (DA-CNN proposal, 8 Oct 2026).
It implements the CIFAR-100 mechanism pilot end to end (work packages WP2 to WP7 for one seed, plus multi-seed
aggregation and a deployable inference bundle).

**Backbone:** a plain Wide ResNet-28-4. The original DA-CNN code is not used to run anything here
(see "Not included" below).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install the CUDA build of torch for your GPU first if needed
python -m pytest -q tests                # 28 correctness tests, about 10 s on CPU
```

## Run

```bash
# 1. Smoke test: every stage on synthetic data, CPU, about 1 minute
python -m dacc.pipeline --config configs/smoke.yaml --seed 11 --stages all
python scripts/verify_results.py runs/smoke/seed11

# 2. Real pilot, one seed (downloads CIFAR-100 to ./data)
python -m dacc.pipeline --config configs/pilot_cifar100.yaml --seed 11 --stages all

# 3. Screening seeds + aggregation (mean, SD, paired t intervals, gates)
bash scripts/run_seeds.sh configs/pilot_cifar100.yaml 11 22 33

# 4. Finalists
bash scripts/run_seeds.sh configs/pilot_cifar100.yaml 11 22 33 44 55
```

Stages are resumable: a stage is skipped when its output exists. Use `--force` to recompute,
`--stages bank,collect,analyze` to run a subset, and `--override section.key=value` for one-off changes
(a changed setting needs its own `run_name`, per Appendix B).

### The decision point

Run `--stages static,bank,collect,analyze` first and open `metrics/headroom_r0.json`.
If `gate_headroom.pass` is false (accuracy-oracle headroom over the best fixed pair below 0.5 pp),
the controller is not worth building yet: revise the bank (ranks, sites, `model.untied`) or stop.

## Stages and outputs

All outputs go to `runs/<run_name>/seed<seed>/`.

| Stage | What it does | Main output |
|---|---|---|
| `static` | B0: WRN-28-4 on the 40k bank split, SGD 0.1, 200 ep, warm-up 5, cosine | `ckpt/static.pt`, `metrics/static.json` |
| `b1` | B1 capacity control: one always-on correction per site, ranks scaled so extra params match bank + joint controller, trained from scratch for 200+10+30 epochs | `ckpt/b1.pt` |
| `bank` | 10 ep frozen backbone with uniform random joint actions, then 30 ep joint refinement (shared lr = corr lr / 10), anchor eta 0.25 | `ckpt/bank.pt` |
| `collect` | 25-action float32 loss tables on route and val (test only if `eval.final_test`); verified against full forward passes | `targets/*_r0.pt`, `metrics/collect_check.json` |
| `analyze` | H1: oracles, headroom gate, utilities, interactions, decision changes, coverage, redundancy, leave-one-expert-out | `metrics/headroom_r0.json` |
| `controllers` | P1 joint, P2 bilinear, B5 additive, B6 sequential, plus parameter-matched wide B5/B6, all on the frozen bank | `ckpt/ctrl_*.pt`, `metrics/controllers.json` |
| `routers` | B3 soft dynamic conv and B4 hard routing (straight-through Gumbel), trained end to end, frozen bank | `ckpt/router_*.pt` |
| `evaluate` | Real conditional execution for every method, D0 diagnostics, recovery, intervention controls, confusion matrices, gates | `metrics/results.json` |
| `alternate` | Up to 3 rounds: bank fine-tune under rho = (1-eps) pi + eps/25, recollect, retrain; matched-budget B4 and static controls; target drift | `metrics/alternate.json` |
| `profile` | Params, MACs (1 MAC = 1 multiply-add), batch-1 and batch-128 latency with sync, target-collection cost, stage timings | `metrics/profile.json` |
| `export` | Deployment bundle, ONNX and TorchScript, numerical equivalence check | `deploy/` |

Every run stores the resolved config, software versions, split hashes, normalisation statistics,
seed, checkpoint hashes and a per-epoch log (`logs/metrics.jsonl`). Target files carry the source
checkpoint hash, action mapping, transform, normalisation mode and collection throughput.

## Method keys in `results.json`

| Key | Proposal ID | Notes |
|---|---|---|
| `B0_static` | B0 | static checkpoint; `B0_bank_zero_path` is the (0,0) path after bank refinement |
| `B1_capacity` | B1 | extra params and epochs recorded |
| `B2_best_fixed` | B2 | pair chosen on the route split, evaluated on val |
| `B3_soft_dynamic` | B3 | continuous mixture; can beat every discrete action (flagged in diagnostics) |
| `B4_hard_routing` | B4 | same bank, ordinary routing |
| `B5_additive`, `B5_additive_wide` | B5 | wide = parameter-matched to P1 |
| `B6_sequential`, `B6_sequential_wide` | B6 | sees B's real input after executing A |
| `P1_joint` | P1 | proposed controller |
| `P2_bilinear` | P2 | rank-2 interaction |
| `diagnostics.D0_*` | D0 | label-informed oracles, never ranked |
| `P1_alternated`, `B4_extra_budget`, `B0_extra_budget` | Section 10.2 | added by the `alternate` stage |

Each controller entry also has table metrics: regret (mean, p90, max), top-1 match with the loss oracle,
Spearman rank correlation, centred error mean and p99, and the Proposition 1 bound check.
`real_vs_table_actions_agree` confirms real execution picks exactly the actions the table predicts.

## Deployment

```bash
python -m dacc.predict --bundle runs/pilot_cifar100/seed11/deploy/bundle.pt --images a.jpg b.png --topk 5
```

Input: any RGB image, resized to 32x32; normalisation is built in; no label is used.
Output: top-k classes and the executed correction action (i, j).

- `bundle.pt`: model + joint controller + config + normalisation + class names. Loaded with
  `dacc.deploy.load_bundle`; runs the **grouped** path (only the selected correction executes).
- `model.onnx`, `model_ts.pt`: the **dense** path (all four experts computed, one-hot mixed) because
  exporters need control-flow-free graphs. Outputs match the grouped path (checked in
  `deploy/export_check.json`), but this path is not the efficient one: use grouped execution for latency claims.
- `bundle_alternated.pt`: the refined pair from the last alternating round.

## Configs

| File | Purpose |
|---|---|
| `pilot_cifar100.yaml` | Appendix B pilot (all defaults written out) |
| `cifar10.yaml` | transfer to CIFAR-10 |
| `recipe_strong.yaml` | 300 ep, CutMix, EMA recipe experiment |
| `ablation_*.yaml` | KL + MSE objective, untied maps, mean-only descriptor, specialisation (Eq. 20), BN recalibration |
| `smoke.yaml` | tiny synthetic run that exercises every stage |

## Implementation choices the proposal leaves open

These are declared here so they can be reported and, if needed, changed:

1. **Descriptor standardisation.** Controller and router inputs are standardised with statistics fitted on the
   route split only (fixed buffers, no parameters). The P1 MLP still has exactly 101,657 parameters.
2. **Sequential head A target** is min over j of L_ij (the best achievable after choosing i). Head B is trained on
   all five executed A states, so it always sees states it meets at deployment, and also receives one-hot(i).
3. **B4 hard routing** uses straight-through Gumbel-softmax (tau 1.0) and a Switch-style balance loss (0.01).
   B3 softmax includes the zero option, so it can switch corrections off.
4. **Bank phase 1** keeps BN running statistics frozen along with the shared weights; phase 2 updates them.
   Omega is applied only as optimizer weight decay on correction maps (gains undecayed), never twice.
5. **Specialisation (Eq. 20)** samples one (site, expert) per step and weights it against the other experts at that
   site, at a frozen checkpoint taken at the start of each bank phase.
6. **Selection.** Controller and router epochs are selected on validation, so validation numbers are mildly
   optimistic. Clean numbers come from the official test set, enabled only with `eval.final_test: true` after all
   design choices are frozen.
7. **Static extra-budget control** continues the static model for rounds x bank_epochs epochs at lr 0.01 with
   cosine decay; it is a fixed-epoch comparison and P1 additionally spends target collection.
8. **Static checkpoint rule** defaults to the last epoch (`static.checkpoint_rule: last`).

## Verified here vs not verified

Verified in this workspace (CPU, synthetic data): the 28 tests in `tests/` (zero-action equivalence, collector vs
full forward, batch-order invariance, grouped vs looped, kernel fusion vs output mixing, gradient reaching every
expert, BN statistics untouched by collection, no labels or unselected future features at deployment, action index
5i + j, tie rule, disjoint stratified splits, Proposition 1, interaction identity and the worked example,
parameter counts 9,344 / 35,072 / 101,657, MAC counts 2,170,880 / 2,134,016 / 101,504, a small overfit test);
complete runs of every stage for two seeds plus an ablation run with every option switched on; bit-identical
repeat runs; ONNX / TorchScript outputs matching grouped execution.

**Not verified here:** no GPU and no CIFAR download were available, so the real pilot has not been run and there are
no accuracy, latency or GPU-hour numbers. The smoke results are meaningless as performance evidence.

## Not included

- **WP1 (recover and audit the DA-CNN v3 notebook):** needs your original code and weights. Once recovered, the
  correction sites can be added to that model as the second backbone required by H4.
- Tiny ImageNet / ImageNet / detection transfer (WP6 onward) depend on the CIFAR gates passing.
