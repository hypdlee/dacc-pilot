"""Ordinary end-to-end routing baselines over the same correction bank.

* B3 ``soft``: per-site softmax over the n choices (choice 0 = zero kernel),
  mixing the bank linearly (CondConv / DynamicConv style). Its continuous
  action space can represent functions outside the 25 discrete actions.
* B4 ``hard``: per-site top-1 routing trained with straight-through
  Gumbel-softmax; executes exactly one discrete choice per site at inference.

Both routers are local (site A sees h, site B sees B's actual input) and are
trained end to end with the classification loss only plus a Switch-style
load-balance term. No alternative-route loss supervision is used.
Frozen-bank comparison: bank weights and BN statistics are fixed.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .controller import MLP, Normalizer, descriptor, descriptor_dim
from .data import DataBundle
from .model import CorrectedWRN, set_bn_eval
from .utils import JsonlLogger, cosine_warmup_lr, make_generator


class OrdinaryRouter(nn.Module):
    def __init__(self, d_a: int, d_b: int, hidden: int, n: int, mode: str, desc_kind: str):
        super().__init__()
        if mode not in {"soft", "hard"}:
            raise ValueError("mode must be soft or hard")
        self.mode, self.n, self.desc_kind = mode, n, desc_kind
        self.norm_a, self.norm_b = Normalizer(d_a), Normalizer(d_b)
        self.head_a = MLP(d_a, hidden, n)
        self.head_b = MLP(d_b, hidden, n)

    def _weights(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        if self.mode == "soft":
            return F.softmax(logits, 1)
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=True)
        return F.one_hot(logits.argmax(1), self.n).to(logits.dtype)

    def forward(self, model: CorrectedWRN, x: torch.Tensor, tau: float = 1.0
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h = model.prefix(x)
        la = self.head_a(self.norm_a(descriptor(h, self.desc_kind)))
        wa = self._weights(la, tau)
        if self.mode == "hard" and not self.training:
            ia = la.argmax(1)
            ha = model.site_a(h, choice=ia)                       # true conditional execution
        else:
            ha = model.site_a(h, weights=wa)
        hb = model.middle(ha)
        lb = self.head_b(self.norm_b(descriptor(hb, self.desc_kind)))
        wb = self._weights(lb, tau)
        if self.mode == "hard" and not self.training:
            ib = lb.argmax(1)
            z = model.suffix(model.site_b(hb, choice=ib))
            actions = torch.stack([ia, ib], 1)
        else:
            z = model.suffix(model.site_b(hb, weights=wb))
            actions = torch.stack([wa.argmax(1), wb.argmax(1)], 1)
        return z, actions, {"logits_a": la, "logits_b": lb, "w_a": wa, "w_b": wb}


def balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Switch Transformer load balance: n * sum_e f_e * P_e."""
    n = logits.size(1)
    p = F.softmax(logits, 1)
    f = F.one_hot(logits.argmax(1), n).float().mean(0)
    return n * (f * p.mean(0)).sum()


@torch.no_grad()
def fit_router_normalizers(router: OrdinaryRouter, model: CorrectedWRN, bundle: DataBundle,
                           indices: np.ndarray) -> None:
    """Fit input standardisation on the route split under the (0,0) path."""
    model.eval()
    da, db = [], []
    for x, _, _ in bundle.train.batches(indices, 500):
        h = model.prefix(x)
        da.append(descriptor(h, router.desc_kind))
        db.append(descriptor(model.middle(model.site_a(h)), router.desc_kind))
    router.norm_a.fit(torch.cat(da))
    router.norm_b.fit(torch.cat(db))


def train_router(mode: str, cfg: Config, model: CorrectedWRN, bundle: DataBundle, device: torch.device,
                 seed: int, logger: JsonlLogger, finetune_bank: bool = False, epochs: Optional[int] = None,
                 split: str = "route", router: Optional[OrdinaryRouter] = None,
                 bank_lrs: Optional[Tuple[float, float]] = None) -> Tuple[OrdinaryRouter, Dict[str, Any]]:
    rc = cfg.router
    torch.manual_seed(seed)
    n = model.n_choices
    C_a, C_b = model.widths[1], model.widths[2]
    dk = cfg.controller.descriptor
    if router is None:
        router = OrdinaryRouter(descriptor_dim(C_a, dk), descriptor_dim(C_b, dk), rc.hidden, n, mode, dk).to(device)
        fit_router_normalizers(router, model, bundle, bundle.splits["route"])
    opts: List[Tuple[torch.optim.Optimizer, List[float]]] = []
    opts.append((torch.optim.AdamW(list(router.head_a.parameters()) + list(router.head_b.parameters()),
                                   lr=rc.lr, weight_decay=rc.weight_decay), [rc.lr]))
    if finetune_bank:
        # bank updated with the same SGD settings the alternating P1 refinement uses
        corr_lr, shared_lr = bank_lrs if bank_lrs is not None else (cfg.alternate.corr_lr, cfg.alternate.shared_lr)
        shared_groups = model.shared_param_groups(cfg.bank.shared_weight_decay)
        bank_opt = torch.optim.SGD([
            {"params": model.correction_parameters(include_gain=False), "weight_decay": cfg.bank.corr_weight_decay},
            {"params": model.gain_parameters(), "weight_decay": 0.0},
            *shared_groups], lr=corr_lr, momentum=cfg.bank.momentum)
        opts.append((bank_opt, [corr_lr, corr_lr] + [shared_lr] * len(shared_groups)))
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)

    gen = make_generator(seed + 2000, bundle.train.device)
    idx = bundle.splits[split]
    val_data, val_idx = bundle.source("val")
    epochs = rc.epochs if epochs is None else epochs
    bs = rc.batch_size
    steps = max(1, len(idx) // bs)
    total = epochs * steps
    best = {"acc": -1.0, "router": copy.deepcopy(router.state_dict()),
            "model": copy.deepcopy(model.state_dict()) if finetune_bank else None, "epoch": 0}
    history = []
    step = 0
    augment = rc.augment or finetune_bank
    for ep in range(epochs):
        router.train()
        model.train()
        if not finetune_bank or cfg.alternate.bn_mode == "frozen":
            set_bn_eval(model)
        tot, seen = 0.0, 0
        for x, y, _ in bundle.train.batches(idx, bs, shuffle=True, augment=augment, gen=gen, drop_last=len(idx) >= bs):
            for opt, lrs in opts:
                for g, blr in zip(opt.param_groups, lrs):
                    g["lr"] = cosine_warmup_lr(step, total, 0, blr)
            z, _, aux = router(model, x, rc.gumbel_tau)
            loss = F.cross_entropy(z.float(), y)
            if finetune_bank:
                # static-path anchor, as in the bank objective (Eq. 19)
                loss = loss + cfg.bank.anchor_eta * F.cross_entropy(model(x).float(), y)
            if rc.balance_coef > 0:
                loss = loss + rc.balance_coef * (balance_loss(aux["logits_a"]) + balance_loss(aux["logits_b"]))
            for opt, _ in opts:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt, _ in opts:
                opt.step()
            tot += float(loss) * x.size(0)
            seen += x.size(0)
            step += 1
        val = eval_router(router, model, bundle, "val")
        rec = {"stage": f"router_{mode}{'_ft' if finetune_bank else ''}", "epoch": ep + 1,
               "train_objective": tot / max(seen, 1), "val_acc": val["accuracy"], "val_loss": val["mean_loss"]}
        logger.log(rec)
        history.append(rec)
        if val["accuracy"] > best["acc"]:
            best = {"acc": val["accuracy"], "router": copy.deepcopy(router.state_dict()),
                    "model": copy.deepcopy(model.state_dict()) if finetune_bank else None, "epoch": ep + 1}
    router.load_state_dict(best["router"])
    if finetune_bank and best["model"] is not None:
        model.load_state_dict(best["model"])
    for p in model.parameters():
        p.requires_grad_(True)
    router.eval()
    model.eval()
    return router, {"history": history, "selected_epoch": best["epoch"],
                    "params": sum(p.numel() for p in router.parameters()), "selection": "validation accuracy"}


@torch.no_grad()
def eval_router(router: OrdinaryRouter, model: CorrectedWRN, bundle: DataBundle, split: str) -> Dict[str, Any]:
    was_r, was_m = router.training, model.training
    router.eval()
    model.eval()
    data, idx = bundle.source(split)
    tot, corr, loss = 0, 0, 0.0
    for x, y, _ in data.batches(idx, 500):
        z, _, _ = router(model, x)
        loss += float(F.cross_entropy(z.float(), y, reduction="sum"))
        corr += int((z.argmax(1) == y).sum())
        tot += x.size(0)
    router.train(was_r)
    model.train(was_m)
    return {"accuracy": 100.0 * corr / max(tot, 1), "mean_loss": loss / max(tot, 1), "n": tot}
