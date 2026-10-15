"""Deterministic alternative-route target collection (Sections 7 and 13.1).

For each example: run the shared prefix once (cache h), evaluate each of the
n choices at A, and for each resulting state recompute the intervening
backbone (B's input depends on the A choice; reusing the zero-correction input
would erase the interaction) before evaluating the n choices at B and the
common classifier. Per-example unsmoothed cross-entropy is stored in float32.

Collection runs in evaluation mode with gradients disabled, fixed BN
statistics, no dropout. All alternatives see the same clean transformed input.
The collector asserts that no parameter or BN running statistic changed.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .controller import descriptor
from .data import DeviceData
from .model import CorrectedWRN, action_table, bn_state
from .utils import hash_state_dict, synchronize


@torch.no_grad()
def collect_targets(model: CorrectedWRN, data: DeviceData, indices: np.ndarray, batch_size: int,
                    device: torch.device, desc_kind: str = "full", amp: bool = False,
                    store_b_desc: bool = True, extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    params_before = hash_state_dict({k: v for k, v in model.state_dict().items()})
    bn_before = bn_state(model)
    n = model.n_choices
    N = len(indices)

    losses = torch.empty(N, n, n, dtype=torch.float32)
    preds = torch.empty(N, n, n, dtype=torch.int32)
    labels = torch.empty(N, dtype=torch.long)
    idx_out = torch.empty(N, dtype=torch.long)
    d_h_list, d_b_list = [], []

    autocast = torch.autocast(device_type=device.type, dtype=torch.float16) if (amp and device.type == "cuda") else nullcontext()
    synchronize(device)
    t0 = time.perf_counter()
    pos = 0
    for x, y, idx in data.batches(indices, batch_size, shuffle=False, augment=False):
        b = x.size(0)
        with autocast:
            h = model.prefix(x)
        d_h_list.append(descriptor(h, desc_kind).cpu())
        db = []
        for i in range(n):
            ci = torch.full((b,), i, dtype=torch.long, device=device)
            with autocast:
                ha = model.site_a(h, choice=ci)
                hb = model.middle(ha)
            if store_b_desc:
                db.append(descriptor(hb, desc_kind).cpu())
            # evaluate all n choices at B in one grouped call
            hb_rep = hb.repeat(n, 1, 1, 1)
            cj = torch.arange(n, device=device).repeat_interleave(b)
            with autocast:
                z = model.suffix(model.site_b(hb_rep, choice=cj))
            logp = F.log_softmax(z.float(), 1)                       # float32 losses
            yy = y.repeat(n)
            l = -logp.gather(1, yy[:, None]).squeeze(1)
            losses[pos:pos + b, i, :] = l.view(n, b).t().cpu()
            preds[pos:pos + b, i, :] = z.argmax(1).view(n, b).t().int().cpu()
        if store_b_desc:
            d_b_list.append(torch.stack(db, 1))
        labels[pos:pos + b] = y.cpu()
        idx_out[pos:pos + b] = idx.cpu()
        pos += b
    synchronize(device)
    elapsed = time.perf_counter() - t0

    params_after = hash_state_dict({k: v for k, v in model.state_dict().items()})
    bn_after = bn_state(model)
    if params_after != params_before:
        raise RuntimeError("target collection modified model parameters")
    for k in bn_before:
        if not torch.equal(bn_before[k], bn_after[k]):
            raise RuntimeError(f"target collection modified BN statistic {k}")
    model.train(was_training)

    correct = preds == labels[:, None, None]
    meta = {
        "checkpoint_hash": params_before,
        "action_mapping": action_table(n),
        "index_rule": "a = n * i + j (row-major)",
        "transform": "clean (no augmentation), bank-split channel normalisation",
        "normalization_mode": "eval (fixed running statistics)",
        "loss": "unsmoothed cross-entropy, float32 log-softmax",
        "amp": bool(amp),
        "descriptor": desc_kind,
        "n_examples": N,
        "seconds": elapsed,
        "images_per_second": N / max(elapsed, 1e-9),
        "gpu_hours": elapsed / 3600.0 if device.type == "cuda" else 0.0,
        "device_hours": elapsed / 3600.0,
        "device": str(device),
    }
    if extra_meta:
        meta.update(extra_meta)
    out = {"losses": losses, "preds": preds, "correct": correct, "labels": labels, "indices": idx_out,
           "desc_h": torch.cat(d_h_list, 0), "meta": meta}
    if store_b_desc:
        out["desc_b"] = torch.cat(d_b_list, 0)                       # (N, n, D_b)
    return out


def save_targets(t: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)


def load_targets(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def single_action_losses(model: CorrectedWRN, x: torch.Tensor, y: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Independent complete forward pass, used to verify the cached collector."""
    model.eval()
    z = model(x, actions)
    return F.cross_entropy(z.float(), y, reduction="none")


def target_drift(old: Dict[str, Any], new: Dict[str, Any], subset: Optional[int] = None) -> Dict[str, float]:
    """Change in centred losses and in the best action between two tables."""
    if not torch.equal(old["indices"], new["indices"]):
        raise ValueError("drift requires tables over the same examples in the same order")
    a, b = old["losses"], new["losses"]
    if subset is not None:
        a, b = a[:subset], b[:subset]
    ca = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    cb = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return {
        "mean_abs_centered_change": float((ca - cb).abs().mean()),
        "best_action_agreement": float((a.flatten(1).argmin(1) == b.flatten(1).argmin(1)).float().mean()),
        "mean_loss_old": float(a.mean()),
        "mean_loss_new": float(b.mean()),
    }
