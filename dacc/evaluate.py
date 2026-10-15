"""Real-execution evaluation (no label input to any decision).

These functions run the actual grouped conditional forward pass, unlike the
table-based diagnostics in ``analysis.py``. A correctness test checks that both
agree exactly for a frozen bank.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import SequentialController, decide, descriptor
from .data import DeviceData
from .model import CorrectedWRN


@torch.no_grad()
def eval_fixed_action(model: CorrectedWRN, data: DeviceData, indices: np.ndarray, batch_size: int,
                      action: Tuple[int, int] = (0, 0)) -> Dict[str, float]:
    was = model.training
    model.eval()
    tot, correct, loss = 0, 0, 0.0
    for x, y, _ in data.batches(indices, batch_size):
        a = torch.tensor(action, device=x.device).repeat(x.size(0), 1)
        z = model(x, a).float()
        loss += float(F.cross_entropy(z, y, reduction="sum"))
        correct += int((z.argmax(1) == y).sum())
        tot += x.size(0)
    model.train(was)
    return {"accuracy": 100.0 * correct / max(tot, 1), "mean_loss": loss / max(tot, 1), "n": tot}


@torch.no_grad()
def deploy_forward(model: CorrectedWRN, ctrl: nn.Module, x: torch.Tensor, desc_kind: str,
                   cost: Optional[torch.Tensor] = None, lam: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Label-free deployment execution (Section 13.3). Returns logits and (B, 2) actions."""
    h = model.prefix(x)
    if isinstance(ctrl, SequentialController):
        qa = ctrl.predict_a(descriptor(h, desc_kind))
        i = torch.argmin(qa, 1)
        hb = model.middle(model.site_a(h, choice=i))
        qb = ctrl.predict_b(descriptor(hb, desc_kind), i)
        j = torch.argmin(qb, 1)
        z = model.suffix(model.site_b(hb, choice=j))
        return z, torch.stack([i, j], 1)
    actions = decide(ctrl(descriptor(h, desc_kind)), cost, lam)
    ha = model.site_a(h, choice=actions[:, 0])
    hb = model.middle(ha)
    z = model.suffix(model.site_b(hb, choice=actions[:, 1]))
    return z, actions


@torch.no_grad()
def eval_with_policy(model: CorrectedWRN, data: DeviceData, indices: np.ndarray, batch_size: int,
                     policy, num_classes: int) -> Dict[str, Any]:
    """Generic evaluation. ``policy(x) -> (logits, actions_or_None)``.

    Also runs the (0,0) path to report corrected / damaged counts.
    """
    was = model.training
    model.eval()
    preds, labels, losses, top5, acts, static_ok = [], [], [], [], [], []
    for x, y, _ in data.batches(indices, batch_size):
        z, a = policy(x)
        z = z.float()
        zs = model(x).float()
        losses.append(F.cross_entropy(z, y, reduction="none").cpu())
        preds.append(z.argmax(1).cpu())
        k = min(5, z.size(1))
        top5.append((z.topk(k, 1).indices == y[:, None]).any(1).cpu())
        labels.append(y.cpu())
        static_ok.append((zs.argmax(1) == y).cpu())
        if a is not None:
            acts.append(a.cpu())
    model.train(was)
    preds_t, labels_t = torch.cat(preds), torch.cat(labels)
    correct = preds_t == labels_t
    sc = torch.cat(static_ok)
    per_class = []
    for c in range(num_classes):
        m = labels_t == c
        per_class.append(float(correct[m].float().mean() * 100) if m.any() else None)
    out: Dict[str, Any] = {
        "accuracy": float(correct.float().mean() * 100),
        "top5_accuracy": float(torch.cat(top5).float().mean() * 100),
        "mean_loss": float(torch.cat(losses).mean()),
        "n": int(len(labels_t)),
        "corrected_vs_static": int((correct & ~sc).sum()),
        "damaged_vs_static": int((~correct & sc).sum()),
        "per_class_accuracy": per_class,
        "_correct": correct,
        "_preds": preds_t,
        "_labels": labels_t,
    }
    if acts:
        A = torch.cat(acts)
        n = model.n_choices
        out["usage"] = torch.bincount(A[:, 0] * n + A[:, 1], minlength=n * n).view(n, n).tolist()
        out["_actions"] = A
    return out


def strip_private(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}
