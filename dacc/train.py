"""Training stages (Section 9) and cached-target controller fitting (Section 8).

* ``train_static``   B0 static backbone (also reused for B1 and the static
                     extra-budget control).
* ``train_bank``     correction bank: frozen shared path with uniform random
                     joint actions, then joint fine-tuning with a smaller
                     shared learning rate, plus the static-path anchor (Eq. 19).
                     Optional unresolved-example specialisation (Eq. 20).
* ``fit_controller`` joint / additive / bilinear / sequential controllers on
                     a frozen bank from cached descriptors and loss tables.
"""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .analysis import controller_metrics
from .config import Config
from .controller import (SequentialController, build_controller, decide, hidden_for_params,
                         router_objective, sequential_targets)
from .data import DataBundle, mix_batch
from .evaluate import eval_fixed_action
from .model import CorrectedWRN, set_bn_eval
from .utils import JsonlLogger, cosine_warmup_lr, make_generator


def _autocast(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _scaler(device: torch.device, amp: bool):
    return torch.amp.GradScaler("cuda", enabled=bool(amp and device.type == "cuda"))


def soft_ce(logits: torch.Tensor, soft_targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing > 0:
        k = soft_targets.size(1)
        soft_targets = soft_targets * (1 - smoothing) + smoothing / k
    return -(soft_targets * F.log_softmax(logits.float(), 1)).sum(1).mean()


def clean_train_subset(cfg: Config, bundle: DataBundle) -> np.ndarray:
    bank = bundle.splits["bank"]
    k = min(cfg.static.clean_train_eval_size, len(bank))
    rng = np.random.default_rng(cfg.data.split_seed + 7)
    return np.sort(rng.choice(bank, size=k, replace=False))


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


# ===================================================================== static
def train_static(cfg: Config, model: CorrectedWRN, bundle: DataBundle, device: torch.device, seed: int,
                 logger: JsonlLogger, tag: str = "static", epochs: Optional[int] = None,
                 lr: Optional[float] = None, warmup: Optional[int] = None,
                 fixed_action: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
    """SGD (or AdamW) with warm-up + cosine. Trains on the bank split only.

    ``fixed_action`` other than (0, 0) trains the corrections too (B1 control).
    """
    s = cfg.static
    epochs = s.epochs if epochs is None else epochs
    warmup = s.warmup_epochs if warmup is None else warmup
    model.to(device).train()
    train_corr = tuple(fixed_action) != (0, 0)

    groups = model.shared_param_groups(s.weight_decay if s.optimizer == "sgd" else s.adamw_wd)
    if train_corr:
        groups.append({"params": model.correction_parameters(include_gain=False),
                       "weight_decay": cfg.bank.corr_weight_decay})
        groups.append({"params": model.gain_parameters(), "weight_decay": 0.0})
    else:
        for p in model.correction_parameters():
            p.requires_grad_(False)
    if s.optimizer == "sgd":
        base_lr = s.lr if lr is None else lr
        opt = torch.optim.SGD(groups, lr=base_lr, momentum=s.momentum, nesterov=s.nesterov)
    elif s.optimizer == "adamw":
        base_lr = s.adamw_lr if lr is None else lr
        opt = torch.optim.AdamW(groups, lr=base_lr)
    else:
        raise ValueError(f"unknown optimizer {s.optimizer}")

    gen = make_generator(seed, bundle.train.device)
    bank = bundle.splits["bank"]
    bs = s.batch_size
    steps_per_epoch = len(bank) // bs
    total = max(1, epochs * steps_per_epoch)
    warm_steps = warmup * steps_per_epoch
    scaler = _scaler(device, cfg.amp)
    ema = EMA(model, s.ema_decay) if s.ema_decay > 0 else None
    clean_idx = clean_train_subset(cfg, bundle)
    val_data, val_idx = bundle.source("val")
    action_t = torch.tensor(fixed_action, device=device)

    best = {"acc": -1.0, "state": None, "epoch": -1}
    history: List[Dict[str, Any]] = []
    step = 0
    for ep in range(epochs):
        model.train()
        if train_corr:
            for p in model.correction_parameters():
                p.requires_grad_(True)
        run_loss, seen = 0.0, 0
        for x, y, _ in bundle.train.batches(bank, bs, shuffle=True, augment=True, gen=gen, drop_last=True):
            for g in opt.param_groups:
                g["lr"] = cosine_warmup_lr(step, total, warm_steps, base_lr)
            xm, ys = mix_batch(x, y, bundle.num_classes, s.mix, s.mix_alpha, gen)
            with _autocast(device, cfg.amp):
                a = action_t.repeat(x.size(0), 1) if train_corr else None
                z = model(xm, a)
            loss = soft_ce(z, ys, s.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            run_loss += float(loss) * x.size(0)
            seen += x.size(0)
            step += 1
        evm = ema.shadow if ema is not None else model
        val = eval_fixed_action(evm, val_data, val_idx, 500, fixed_action)
        clean = eval_fixed_action(evm, bundle.train, clean_idx, 500, fixed_action)
        rec = {"stage": tag, "epoch": ep + 1, "train_objective": run_loss / max(seen, 1),
               "objective": f"{'mixed-label ' if s.mix != 'none' else ''}CE, smoothing={s.label_smoothing}",
               "clean_train_acc": clean["accuracy"], "clean_train_loss": clean["mean_loss"],
               "val_acc": val["accuracy"], "val_loss_unsmoothed": val["mean_loss"],
               "lr": opt.param_groups[0]["lr"]}
        logger.log(rec)
        history.append(rec)
        if val["accuracy"] > best["acc"]:
            best = {"acc": val["accuracy"], "state": copy.deepcopy(evm.state_dict()), "epoch": ep + 1}

    final_model = ema.shadow if ema is not None else model
    if s.checkpoint_rule == "best_val" and best["state"] is not None:
        model.load_state_dict(best["state"])
        selected = best["epoch"]
    else:
        if ema is not None:
            model.load_state_dict(final_model.state_dict())
        selected = epochs
    for p in model.parameters():
        p.requires_grad_(True)
    return {"history": history, "selected_epoch": selected, "checkpoint_rule": s.checkpoint_rule,
            "best_val_epoch": best["epoch"], "best_val_acc": best["acc"],
            "final_val": eval_fixed_action(model, val_data, val_idx, 500, fixed_action),
            "final_clean_train": eval_fixed_action(model, bundle.train, clean_idx, 500, fixed_action)}


def b1_site_ranks(cfg: Config, target_extra: int, widths: List[int]) -> Tuple[int, int]:
    """Ranks for a single always-on correction per site matching ``target_extra`` params."""
    ca, cb = widths[1], widths[2]
    ra0, rb0 = ca // cfg.model.rank_divisor, cb // cfg.model.rank_divisor
    per_unit = ra0 * (2 * ca + 9) + rb0 * (2 * cb + 9)
    scale = target_extra / per_unit
    return max(1, int(round(ra0 * scale))), max(1, int(round(rb0 * scale)))


# ===================================================================== bank
def _sample_uniform_actions(b: int, n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randint(0, n, (b, 2), generator=gen, device=gen.device).to(device)


def _spec_loss(model: CorrectedWRN, frozen: CorrectedWRN, x: torch.Tensor, h: torch.Tensor, y: torch.Tensor,
               cfg: Config, gen: torch.Generator) -> torch.Tensor:
    """One-site unresolved-example weighting (Eq. 20), one random (site, expert) per step.

    w_a(x) = clip(min_{b != a} L_b(x) - kappa, 0, wmax) with b ranging over the
    other experts at the same site, computed at the frozen target checkpoint,
    gradients stopped and normalised within the minibatch.
    """
    E = model.num_experts
    site = int(torch.randint(0, 2, (1,), generator=gen, device=gen.device).item())
    e = int(torch.randint(1, E + 1, (1,), generator=gen, device=gen.device).item())
    b = h.size(0)
    with torch.no_grad():
        hf = frozen.prefix(x)                                   # target checkpoint's own features
        others = [k for k in range(1, E + 1) if k != e]
        ls = []
        for k in others:
            act = torch.zeros(b, 2, dtype=torch.long, device=h.device)
            act[:, site] = k
            ls.append(F.cross_entropy(frozen.tail(hf, act).float(), y, reduction="none"))
        m = torch.stack(ls, 1).min(1).values if ls else torch.zeros(b, device=h.device)
        w = (m - cfg.bank.spec_kappa).clamp(0, cfg.bank.spec_wmax)
        w = w / w.sum().clamp_min(1e-12)
    act = torch.zeros(b, 2, dtype=torch.long, device=h.device)
    act[:, site] = e
    le = F.cross_entropy(model.tail(h, act).float(), y, reduction="none")
    return (w * le).sum()


def train_bank(cfg: Config, model: CorrectedWRN, bundle: DataBundle, device: torch.device, seed: int,
               logger: JsonlLogger) -> Dict[str, Any]:
    bc = cfg.bank
    n = model.n_choices
    model.to(device)
    gen = make_generator(seed + 1000, bundle.train.device)
    bank = bundle.splits["bank"]
    bs = bc.batch_size
    steps = len(bank) // bs
    val_data, val_idx = bundle.source("val")
    scaler = _scaler(device, cfg.amp)
    history: List[Dict[str, Any]] = []
    frozen = None

    def run_phase(name: str, epochs: int, opt: torch.optim.Optimizer, base_lrs: List[float],
                  freeze_shared: bool) -> None:
        nonlocal frozen
        total = max(1, epochs * steps)
        step = 0
        if bc.specialize:
            frozen = copy.deepcopy(model).eval()
            for p in frozen.parameters():
                p.requires_grad_(False)
        for ep in range(epochs):
            model.train()
            if freeze_shared:
                set_bn_eval(model)                       # shared path frozen incl. BN statistics
            tot, seen = {"sampled": 0.0, "anchor": 0.0, "spec": 0.0}, 0
            for x, y, _ in bundle.train.batches(bank, bs, shuffle=True, augment=True, gen=gen, drop_last=True):
                for g, blr in zip(opt.param_groups, base_lrs):
                    g["lr"] = cosine_warmup_lr(step, total, 0, blr)
                a = _sample_uniform_actions(x.size(0), n, gen, device)
                with _autocast(device, cfg.amp):
                    h = model.prefix(x)
                    ls = F.cross_entropy(model.tail(h, a).float(), y)
                    la = F.cross_entropy(model.tail(h, None).float(), y)
                    loss = ls + bc.anchor_eta * la
                    lsp = torch.zeros((), device=device)
                    if bc.specialize:
                        lsp = _spec_loss(model, frozen, x, h, y, cfg, gen)
                        loss = loss + bc.spec_weight * lsp
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot["sampled"] += float(ls) * x.size(0)
                tot["anchor"] += float(la) * x.size(0)
                tot["spec"] += float(lsp) * x.size(0)
                seen += x.size(0)
                step += 1
            v00 = eval_fixed_action(model, val_data, val_idx, 500, (0, 0))
            vrand = [eval_fixed_action(model, val_data, val_idx, 500, (i, j))["accuracy"]
                     for (i, j) in ((1, 1), (2, 3), (4, 4))]
            rec = {"stage": f"bank_{name}", "epoch": ep + 1,
                   "sampled_ce": tot["sampled"] / max(seen, 1), "anchor_ce": tot["anchor"] / max(seen, 1),
                   "spec_loss": tot["spec"] / max(seen, 1),
                   "val_acc_static_path": v00["accuracy"], "val_loss_static_path": v00["mean_loss"],
                   "val_acc_probe_actions_mean": float(np.mean(vrand)),
                   "gain_A": float(model.corr_a.gain), "gain_B": float(model.corr_b.gain)}
            logger.log(rec)
            history.append(rec)

    # Phase 1: freeze shared path, train corrections only.
    for p in model.shared_parameters():
        p.requires_grad_(False)
    for p in model.correction_parameters():
        p.requires_grad_(True)
    opt1 = torch.optim.SGD([
        {"params": model.correction_parameters(include_gain=False), "weight_decay": bc.corr_weight_decay},
        {"params": model.gain_parameters(), "weight_decay": 0.0},
    ], lr=bc.warm_lr, momentum=bc.momentum)
    run_phase("warm", bc.warm_epochs, opt1, [bc.warm_lr, bc.warm_lr], freeze_shared=True)

    # Phase 2: joint fine-tuning; shared lr <= corr lr / 10.
    for p in model.parameters():
        p.requires_grad_(True)
    shared_groups = model.shared_param_groups(bc.shared_weight_decay)
    opt2 = torch.optim.SGD([
        {"params": model.correction_parameters(include_gain=False), "weight_decay": bc.corr_weight_decay},
        {"params": model.gain_parameters(), "weight_decay": 0.0},
        *shared_groups,
    ], lr=bc.refine_corr_lr, momentum=bc.momentum)
    lrs = [bc.refine_corr_lr, bc.refine_corr_lr] + [bc.refine_shared_lr] * len(shared_groups)
    run_phase("refine", bc.refine_epochs, opt2, lrs, freeze_shared=False)
    model.eval()
    return {"history": history,
            "regularisation": "omega implemented as optimizer weight decay on correction maps only "
                              f"({bc.corr_weight_decay}); gains undecayed; no explicit L2 term",
            "final_val_static_path": eval_fixed_action(model, val_data, val_idx, 500, (0, 0))}


# ===================================================================== controllers
def _seq_composite(qa: torch.Tensor, qb_all: torch.Tensor) -> torch.Tensor:
    """Composite table whose argmin equals the sequential decision.

    q[i, j] = qa[i] + (qb_i[j] - min_j qb_i[j]).
    """
    return qa[:, :, None] + (qb_all - qb_all.min(2, keepdim=True).values)


@torch.no_grad()
def predict_table(ctrl: nn.Module, T: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicted table q (N, n, n) and the controller's deployment actions (N, 2).

    For frozen banks the stored B-input descriptors equal those the sequential
    controller would compute after executing its chosen i, so this matches
    real execution exactly (verified by tests).
    """
    ctrl.eval()
    dh = T["desc_h"].to(device)
    if isinstance(ctrl, SequentialController):
        n = ctrl.n
        qa = ctrl.predict_a(dh)
        db = T["desc_b"].to(device)
        qb = torch.stack([ctrl.predict_b(db[:, i], torch.full((len(dh),), i, device=device, dtype=torch.long))
                          for i in range(n)], 1)                      # (N, n, n)
        i = qa.argmin(1)
        j = qb[torch.arange(len(dh), device=device), i].argmin(1)
        return _seq_composite(qa, qb).cpu(), torch.stack([i, j], 1).cpu()
    q = ctrl(dh)
    return q.cpu(), decide(q).cpu()


def build_sized_controller(kind: str, cfg: Config, d_h: int, d_b: int, n: int,
                           wide_target: Optional[int] = None) -> nn.Module:
    """Construct a controller; ``wide_target`` matches its parameter count to a target."""
    cc = cfg.controller
    hidden = cc.hidden
    if wide_target is not None:
        if kind == "additive":
            hidden = hidden_for_params(d_h, 1 + 2 * (n - 1), wide_target)
        elif kind == "bilinear":
            hidden = hidden_for_params(d_h, 1 + 2 * (n - 1) + 2 * cc.bilinear_rank * (n - 1), wide_target)
        elif kind == "sequential":
            # split the budget equally between the two heads
            ha = hidden_for_params(d_h, n, wide_target // 2)
            hb = hidden_for_params(d_b + n, n, wide_target - wide_target // 2)
            return build_controller("sequential", d_h, ha, n, d_b=d_b, hidden_b=hb)
    return build_controller(kind, d_h, hidden, n, cc.bilinear_rank, d_b=d_b, hidden_b=hidden)


def fit_controller(kind: str, cfg: Config, T_train: Dict[str, Any], T_val: Dict[str, Any],
                   device: torch.device, seed: int, logger: JsonlLogger, tag: str,
                   wide_target: Optional[int] = None, init: Optional[nn.Module] = None,
                   epochs: Optional[int] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    cc = cfg.controller
    torch.manual_seed(seed)
    n = T_train["losses"].shape[-1]
    d_h = T_train["desc_h"].shape[1]
    d_b = T_train["desc_b"].shape[2] if "desc_b" in T_train else d_h
    ctrl = init if init is not None else build_sized_controller(kind, cfg, d_h, d_b, n, wide_target)
    ctrl.to(device)

    Dh = T_train["desc_h"].to(device)
    L = T_train["losses"].to(device)
    if init is None:
        if isinstance(ctrl, SequentialController):
            ctrl.norm_a.fit(Dh)
            ctrl.norm_b.fit(T_train["desc_b"].to(device).flatten(0, 1))
        else:
            ctrl.norm.fit(Dh)
    Db = T_train["desc_b"].to(device) if isinstance(ctrl, SequentialController) else None

    opt = torch.optim.AdamW(ctrl.parameters(), lr=cc.lr, weight_decay=cc.weight_decay)
    N = len(Dh)
    g = torch.Generator().manual_seed(seed)
    best = {"score": math.inf, "state": copy.deepcopy(ctrl.state_dict()), "epoch": 0}
    history = []
    for ep in range(epochs if epochs is not None else cc.epochs):
        ctrl.train()
        perm = torch.randperm(N, generator=g).to(device)
        tot, seen = 0.0, 0
        for s in range(0, N, cc.batch_size):
            bi = perm[s:s + cc.batch_size]
            if isinstance(ctrl, SequentialController):
                ta, tb = sequential_targets(L[bi])
                qa = ctrl.predict_a(Dh[bi])
                loss = router_objective(qa, ta, cc.score_loss, cc.huber_delta, cc.beta_kl, cc.tau)
                lb = 0.0
                for i in range(n):
                    ii = torch.full((len(bi),), i, device=device, dtype=torch.long)
                    qb = ctrl.predict_b(Db[bi, i], ii)
                    lb = lb + router_objective(qb, tb[:, i], cc.score_loss, cc.huber_delta, cc.beta_kl, cc.tau)
                loss = loss + lb / n
            else:
                q = ctrl(Dh[bi])
                loss = router_objective(q, L[bi], cc.score_loss, cc.huber_delta, cc.beta_kl, cc.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
            seen += len(bi)
        qv, av = predict_table(ctrl, T_val, device)
        m = controller_metrics(qv, T_val, av)
        score = m["regret_mean"] if cc.select_metric == "regret" else -m["accuracy"]
        rec = {"stage": f"controller_{tag}", "epoch": ep + 1, "train_objective": tot / max(seen, 1),
               "val_regret": m["regret_mean"], "val_acc": m["accuracy"], "val_spearman": m["spearman_mean"]}
        logger.log(rec)
        history.append(rec)
        if score < best["score"]:
            best = {"score": score, "state": copy.deepcopy(ctrl.state_dict()), "epoch": ep + 1}
    ctrl.load_state_dict(best["state"])
    ctrl.eval()
    params = sum(p.numel() for p in ctrl.parameters())
    return ctrl, {"history": history, "selected_epoch": best["epoch"], "params": params,
                  "selection": f"validation {cc.select_metric}"}
