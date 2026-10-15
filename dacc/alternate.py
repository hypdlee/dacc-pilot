"""Alternating bank / controller refinement (Section 10.2).

Round r:
  1. fine-tune the bank on the bank split with actions sampled from
     rho(a|h) = (1 - eps) * pi_phi(a|h) + eps / 25, pi_phi = softmax(-q / tau),
     plus the static-path anchor (Eq. 19, 21);
  2. BN statistics: ``frozen`` (fixed running stats) or ``recalibrate`` (reset,
     then re-estimate under the controller's hard decisions on the bank split,
     a declared reference action distribution; never on test labels);
  3. recollect route / validation targets; measure drift on a monitored subset;
  4. retrain the controller (warm start) on the refreshed targets.

Matched-budget controls (Section 10.2, 16): ordinary hard routing (B4)
fine-tuned end to end for the same number of bank epochs, and the static
backbone trained for the same extra epochs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .controller import SequentialController, descriptor, policy_probs
from .data import DataBundle
from .evaluate import eval_fixed_action
from .model import CorrectedWRN, set_bn_eval
from .targets import collect_targets, target_drift
from .train import _autocast, _scaler, fit_controller, predict_table
from .analysis import controller_metrics
from .utils import JsonlLogger, cosine_warmup_lr, make_generator


def epsilon_for_round(cfg: Config, r: int) -> float:
    ac = cfg.alternate
    if ac.eps_schedule == "fixed" or ac.rounds <= 1:
        return ac.eps
    if ac.eps_schedule == "linear":
        return ac.eps + (ac.eps_final - ac.eps) * r / (ac.rounds - 1)
    raise ValueError(f"unknown eps schedule {ac.eps_schedule}")


def sample_rho(q: torch.Tensor, eps: float, tau: float, gen: torch.Generator) -> torch.Tensor:
    n = q.shape[-1]
    pi = policy_probs(q, tau)
    rho = (1 - eps) * pi + eps / (n * n)
    a = torch.multinomial(rho.to(gen.device), 1, generator=gen).squeeze(1).to(q.device)
    return torch.stack([a // n, a % n], 1)


@torch.no_grad()
def recalibrate_bn(model: CorrectedWRN, ctrl: nn.Module, bundle: DataBundle, desc_kind: str,
                   batches: int, bs: int, gen: torch.Generator) -> None:
    """Reset BN running stats and re-estimate (cumulative average) under the
    controller's hard decisions on clean bank-split images.

    The prefix BN layers precede any decision, so their statistics are
    re-estimated from the same passes.
    """
    from .controller import decide

    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for k, (x, _, _) in enumerate(bundle.train.batches(bundle.splits["bank"], bs, shuffle=True, gen=gen)):
        if k >= batches:
            break
        h = model.prefix(x)
        if isinstance(ctrl, SequentialController):
            i = ctrl.predict_a(descriptor(h, desc_kind)).argmin(1)
            hb = model.middle(model.site_a(h, choice=i))
            j = ctrl.predict_b(descriptor(hb, desc_kind), i).argmin(1)
            model.suffix(model.site_b(hb, choice=j))
        else:
            model.tail(h, decide(ctrl(descriptor(h, desc_kind))))
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.1
    model.eval()


def finetune_bank_under_policy(cfg: Config, model: CorrectedWRN, ctrl: nn.Module, bundle: DataBundle,
                               device: torch.device, seed: int, eps: float, logger: JsonlLogger,
                               tag: str) -> List[Dict[str, Any]]:
    ac, bc = cfg.alternate, cfg.bank
    dk = cfg.controller.descriptor
    ctrl.eval()
    shared_groups = model.shared_param_groups(bc.shared_weight_decay)
    opt = torch.optim.SGD([
        {"params": model.correction_parameters(include_gain=False), "weight_decay": bc.corr_weight_decay},
        {"params": model.gain_parameters(), "weight_decay": 0.0},
        *shared_groups], lr=ac.corr_lr, momentum=bc.momentum)
    lrs = [ac.corr_lr, ac.corr_lr] + [ac.shared_lr] * len(shared_groups)
    gen = make_generator(seed + 3000, bundle.train.device)
    bank = bundle.splits["bank"]
    steps = len(bank) // bc.batch_size
    total = max(1, ac.bank_epochs * steps)
    scaler = _scaler(device, cfg.amp)
    val_data, val_idx = bundle.source("val")
    hist = []
    step = 0
    for ep in range(ac.bank_epochs):
        model.train()
        if ac.bn_mode == "frozen":
            set_bn_eval(model)
        for p in model.parameters():
            p.requires_grad_(True)
        tot, seen = 0.0, 0
        for x, y, _ in bundle.train.batches(bank, bc.batch_size, shuffle=True, augment=True, gen=gen, drop_last=True):
            for g, blr in zip(opt.param_groups, lrs):
                g["lr"] = cosine_warmup_lr(step, total, 0, blr)
            with _autocast(device, cfg.amp):
                h = model.prefix(x)
                with torch.no_grad():
                    if isinstance(ctrl, SequentialController):
                        # sequential policy: sample i, then j given the executed A
                        qa = ctrl.predict_a(descriptor(h.detach(), dk))
                        pa = (1 - eps) * F.softmax(-qa / ac.policy_tau, 1) + eps / ctrl.n
                        i = torch.multinomial(pa.to(gen.device), 1, generator=gen).squeeze(1).to(device)
                        hb0 = model.middle(model.site_a(h.detach(), choice=i))
                        qb = ctrl.predict_b(descriptor(hb0, dk), i)
                        pb = (1 - eps) * F.softmax(-qb / ac.policy_tau, 1) + eps / ctrl.n
                        j = torch.multinomial(pb.to(gen.device), 1, generator=gen).squeeze(1).to(device)
                        a = torch.stack([i, j], 1)
                    else:
                        a = sample_rho(ctrl(descriptor(h.detach(), dk)), eps, ac.policy_tau, gen)
                ls = F.cross_entropy(model.tail(h, a).float(), y)
                la = F.cross_entropy(model.tail(h, None).float(), y)
                loss = ls + bc.anchor_eta * la
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * x.size(0)
            seen += x.size(0)
            step += 1
        v = eval_fixed_action(model, val_data, val_idx, 500, (0, 0))
        rec = {"stage": f"alternate_{tag}_bank", "epoch": ep + 1, "objective": tot / max(seen, 1),
               "eps": eps, "val_acc_static_path": v["accuracy"]}
        logger.log(rec)
        hist.append(rec)
    model.eval()
    return hist


def run_alternation(cfg: Config, model: CorrectedWRN, ctrl: nn.Module, kind: str, bundle: DataBundle,
                    T_route: Dict[str, Any], device: torch.device, seed: int, logger: JsonlLogger,
                    save_cb=None) -> Tuple[CorrectedWRN, nn.Module, Dict[str, Any]]:
    """Run up to ``rounds`` alternating rounds on copies; returns the refined pair."""
    ac = cfg.alternate
    dk = cfg.controller.descriptor
    model = copy.deepcopy(model)
    ctrl = copy.deepcopy(ctrl)
    prev_route = T_route
    rounds: List[Dict[str, Any]] = []
    for r in range(ac.rounds):
        eps = epsilon_for_round(cfg, r)
        hist = finetune_bank_under_policy(cfg, model, ctrl, bundle, device, seed + r, eps, logger, f"r{r + 1}")
        if ac.bn_mode == "recalibrate":
            recalibrate_bn(model, ctrl, bundle, dk, ac.recalib_batches, cfg.bank.batch_size,
                           make_generator(seed + 4000 + r, bundle.train.device))
        data_r, idx_r = bundle.source("route")
        data_v, idx_v = bundle.source("val")
        T_r = collect_targets(model, data_r, idx_r, 250, device, dk, cfg.amp,
                              extra_meta={"split": "route", "round": r + 1})
        T_v = collect_targets(model, data_v, idx_v, 250, device, dk, cfg.amp,
                              extra_meta={"split": "val", "round": r + 1})
        drift = target_drift(prev_route, T_r, subset=ac.drift_subset)
        ctrl, info = fit_controller(kind, cfg, T_r, T_v, device, seed + 100 + r, logger,
                                    tag=f"{kind}_alt_r{r + 1}", init=ctrl, epochs=ac.controller_epochs)
        q, a = predict_table(ctrl, T_v, device)
        m = controller_metrics(q, T_v, a)
        rounds.append({"round": r + 1, "eps": eps, "bank_history": hist, "drift_route": drift,
                       "controller_selected_epoch": info["selected_epoch"], "val": m,
                       "target_collection_device_hours": T_r["meta"]["device_hours"] + T_v["meta"]["device_hours"]})
        logger.log({"stage": "alternate_round", "round": r + 1, "val_acc": m["accuracy"],
                    "val_regret": m["regret_mean"], "drift_mean_abs": drift["mean_abs_centered_change"],
                    "drift_best_agree": drift["best_action_agreement"]})
        if save_cb is not None:
            save_cb(r + 1, model, ctrl, T_r, T_v)
        prev_route = T_r
    return model, ctrl, {"rounds": rounds, "bn_mode": ac.bn_mode, "eps_schedule": ac.eps_schedule,
                         "policy": "rho = (1-eps) softmax(-q/tau) + eps/25"}
