"""Run configuration.

Every default below is the Appendix B "initial pilot value" of the proposal,
or the value stated in the body text where Appendix B is silent. Settings are
stored with every run; changing one requires a new run id (Appendix B).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataCfg:
    dataset: str = "cifar100"            # cifar100 | cifar10 | synthetic
    root: str = "./data"
    download: bool = True
    # Section 10.1: fixed stratified 40,000 / 5,000 / 5,000 split of the
    # official 50,000 training images. Official test set untouched.
    bank_size: int = 40000
    route_size: int = 5000
    val_size: int = 5000
    split_seed: int = 0                  # split is identical across all seeds/methods
    # synthetic dataset (only for smoke tests on machines without CIFAR)
    synthetic_train: int = 2000
    synthetic_test: int = 400
    synthetic_classes: int = 10
    data_on_device: bool = True          # keep uint8 images on the GPU, augment there


@dataclass
class ModelCfg:
    depth: int = 28                      # WRN depth 28
    widen: int = 4                       # widening factor 4 -> 64/128/256 channels
    stem_channels: int = 16
    dropout: float = 0.0
    num_experts: int = 4                 # four experts + zero correction per site
    rank_divisor: int = 4                # r = C / 4 -> 32 at A, 64 at B
    gain_init: float = 0.05              # trainable shared gain per site
    untied: bool = False                 # ablation: expert-specific A and P maps


@dataclass
class StaticTrainCfg:
    # Section 9.1 screening recipe
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    nesterov: bool = False
    label_smoothing: float = 0.0
    mix: str = "none"                    # none | mixup | cutmix (recipe experiment only)
    mix_alpha: float = 1.0
    optimizer: str = "sgd"               # sgd | adamw (recipe comparison)
    adamw_lr: float = 1e-3
    adamw_wd: float = 0.05
    ema_decay: float = 0.0               # optional moving-average weights (0 = off)
    clean_train_eval_size: int = 5000    # clean training accuracy subset (no aug, eval mode)
    checkpoint_rule: str = "last"        # last | best_val  (declared selection rule)


@dataclass
class BankTrainCfg:
    # Section 9.2 + Appendix B
    warm_epochs: int = 10                # frozen backbone
    warm_lr: float = 0.01
    refine_epochs: int = 30
    refine_corr_lr: float = 0.01
    refine_shared_lr: float = 0.001      # <= 1/10 of correction lr
    batch_size: int = 128
    momentum: float = 0.9
    anchor_eta: float = 0.25             # static-path anchor weight
    corr_weight_decay: float = 5e-4      # omega, implemented ONLY as optimizer weight decay
    shared_weight_decay: float = 5e-4
    # Optional unresolved-example specialisation (Eq. 20), ablation only
    specialize: bool = False
    spec_kappa: float = 0.5
    spec_wmax: float = 2.0
    spec_weight: float = 1.0


@dataclass
class ControllerCfg:
    hidden: int = 128
    descriptor: str = "full"             # full (mean,std,2x2 pool) | mean | meanstd
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 30
    score_loss: str = "huber"            # huber | mse
    huber_delta: float = 0.1
    beta_kl: float = 0.0                 # optional KL ablation uses 0.1
    tau: float = 0.1
    bilinear_rank: int = 2
    match_capacity: bool = True          # also train parameter-matched wider additive / sequential
    select_metric: str = "regret"        # validation metric used to pick the controller epoch


@dataclass
class RouterCfg:
    # Ordinary end-to-end routing baselines B3 (soft) and B4 (hard)
    hidden: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 30
    gumbel_tau: float = 1.0
    balance_coef: float = 0.01           # Switch-style load-balance auxiliary loss
    augment: bool = False                # same clean images the supervised controllers see


@dataclass
class AlternateCfg:
    rounds: int = 3
    bank_epochs: int = 10
    corr_lr: float = 0.01
    shared_lr: float = 0.001
    eps: float = 0.2
    eps_final: float = 0.05
    eps_schedule: str = "fixed"          # fixed | linear (to eps_final)
    policy_tau: float = 0.1              # pi_phi(a|h) = softmax(-q / tau)
    bn_mode: str = "frozen"              # frozen | recalibrate
    recalib_batches: int = 200
    drift_subset: int = 1000             # monitored target subset size
    controller_epochs: int = 30
    baselines: bool = True               # B4 and static receive the same extra budget


@dataclass
class EvalCfg:
    headroom_gate_pp: float = 0.5
    predictability_gate: str = "joint_beats_best_fixed"
    coordination_gate_pp: float = 0.3
    reference_static_acc: Optional[float] = None   # set to your reproduced baseline (%)
    reference_tolerance_pp: float = 1.0
    coverage_thresholds: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1])
    cost_lambda: float = 0.0             # accuracy track: zero
    final_test: bool = False             # official test set only when explicitly enabled


@dataclass
class ProfileCfg:
    latency_images: int = 1000
    warmup_iters: int = 50
    batch_sizes: List[int] = field(default_factory=lambda: [1, 128])
    precision: str = "fp32"


@dataclass
class Config:
    run_name: str = "pilot_cifar100"
    out_root: str = "./runs"
    device: str = "auto"
    amp: bool = False                    # mixed precision for training; losses always float32
    deterministic: bool = True
    seeds_screening: List[int] = field(default_factory=lambda: [11, 22, 33])
    seeds_finalist: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    static: StaticTrainCfg = field(default_factory=StaticTrainCfg)
    bank: BankTrainCfg = field(default_factory=BankTrainCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
    router: RouterCfg = field(default_factory=RouterCfg)
    alternate: AlternateCfg = field(default_factory=AlternateCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    profile: ProfileCfg = field(default_factory=ProfileCfg)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _merge(dc: Any, updates: Dict[str, Any], path: str = "") -> None:
    names = {f.name: f for f in fields(dc)}
    for key, val in updates.items():
        if key not in names:
            raise KeyError(f"unknown config key '{path}{key}'")
        cur = getattr(dc, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise TypeError(f"config section '{path}{key}' must be a mapping")
            _merge(cur, val, path=f"{path}{key}.")
        else:
            setattr(dc, key, val)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load defaults, then a YAML file, then ``a.b=value`` overrides."""
    cfg = Config()
    if path:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        nested: Dict[str, Any] = {}
        cursor = nested
        for p in parts[:-1]:
            cursor[p] = {}
            cursor = cursor[p]
        cursor[parts[-1]] = _parse_scalar(raw)
        _merge(cfg, nested)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    m = cfg.model
    if (m.depth - 4) % 6 != 0:
        raise ValueError("WRN depth must satisfy (depth - 4) % 6 == 0")
    if (m.depth - 4) // 6 < 2:
        raise ValueError("need at least 2 blocks per stage so the site block is stride one")
    if cfg.bank.refine_shared_lr > cfg.bank.refine_corr_lr / 10 + 1e-12:
        raise ValueError("shared-path lr must be no larger than 1/10 of the correction lr")
    if cfg.data.dataset not in {"cifar100", "cifar10", "synthetic"}:
        raise ValueError("dataset must be cifar100, cifar10 or synthetic")
    if cfg.controller.descriptor not in {"full", "mean", "meanstd"}:
        raise ValueError("descriptor must be full, mean or meanstd")
    if cfg.alternate.bn_mode not in {"frozen", "recalibrate"}:
        raise ValueError("alternate.bn_mode must be frozen or recalibrate")


def config_from_dict(d: Dict[str, Any]) -> Config:
    cfg = Config()
    _merge(cfg, d)
    validate(cfg)
    return cfg


def clone(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def save_config(cfg: Config, path: Path) -> None:
    path.write_text(cfg.to_json())
