"""Parameters, MACs and latency (Section 17).

MAC convention: one multiply-add = one MAC (= two FLOPs). Counts cover
convolutions and linear layers; BN, activations, pooling, gains, descriptor
statistics and execution overhead are excluded and reported as such.
Latency: warm-up iterations, explicit device synchronisation, at least
``latency_images`` measured images, median and tail percentiles, reported
separately per batch size and precision.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from .controller import SequentialController
from .model import CorrectedWRN
from .utils import synchronize


def conv_linear_macs(module: nn.Module, run: Callable[[], Any], skip_prefixes=("corr_",)) -> int:
    """Count MACs of nn.Conv2d / nn.Linear executed during ``run`` (hooks)."""
    total = [0]
    handles = []
    for name, m in module.named_modules():
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        if isinstance(m, nn.Conv2d):
            def hook(mod, inp, out):
                k = mod.kernel_size[0] * mod.kernel_size[1]
                total[0] += out.numel() * (mod.in_channels // mod.groups) * k
            handles.append(m.register_forward_hook(hook))
        elif isinstance(m, nn.Linear):
            def hook_l(mod, inp, out):
                total[0] += out.numel() * mod.in_features
            handles.append(m.register_forward_hook(hook_l))
    try:
        run()
    finally:
        for h in handles:
            h.remove()
    return total[0]


@torch.no_grad()
def macs_report(model: CorrectedWRN, ctrl: nn.Module, device: torch.device, hw: int = 32) -> Dict[str, Any]:
    model.eval()
    x = torch.zeros(1, 3, hw, hw, device=device)
    backbone = conv_linear_macs(model, lambda: model(x))
    shapes = model.site_shapes(hw)
    _, ha, wa = shapes["A"]
    _, hb, wb = shapes["B"]
    ca, cb = model.corr_a.macs_selected(ha, wa), model.corr_b.macs_selected(hb, wb)
    if isinstance(ctrl, SequentialController):
        ctrl_macs = (ctrl.head_a.fc1.in_features * ctrl.head_a.fc1.out_features
                     + ctrl.head_a.fc2.in_features * ctrl.head_a.fc2.out_features
                     + ctrl.head_b.fc1.in_features * ctrl.head_b.fc1.out_features
                     + ctrl.head_b.fc2.in_features * ctrl.head_b.fc2.out_features)
    else:
        ctrl_macs = ctrl.mlp.fc1.in_features * ctrl.mlp.fc1.out_features + ctrl.mlp.fc2.in_features * ctrl.mlp.fc2.out_features
    n = model.n_choices
    per_action = [[backbone + (ca if i > 0 else 0) + (cb if j > 0 else 0) + ctrl_macs for j in range(n)] for i in range(n)]
    return {
        "convention": "1 MAC = 1 multiply-add (2 FLOPs); conv + linear only",
        "static_backbone_macs": backbone,
        "correction_A_selected_macs": ca,
        "correction_B_selected_macs": cb,
        "corrections_both_selected_macs": ca + cb,
        "controller_linear_macs": ctrl_macs,
        "per_action_total_macs": per_action,
        "excluded": "BN, activations, pooling, gains, descriptor statistics, grouping/scatter overhead",
        "note": "experts 1..4 at a site have equal nominal cost; only zero vs non-zero changes MACs",
    }


def param_report(model: CorrectedWRN, controllers: Dict[str, nn.Module]) -> Dict[str, Any]:
    corr_a = model.corr_a.params_excluding_gain()
    corr_b = model.corr_b.params_excluding_gain()
    total = sum(p.numel() for p in model.parameters())
    out = {
        "model_total_including_corrections": total,
        "static_backbone": total - corr_a - corr_b - 2,
        "correction_A_weights": corr_a,
        "correction_B_weights": corr_b,
        "correction_gains": 2,
        "controllers": {k: sum(p.numel() for p in c.parameters()) for k, c in controllers.items()},
    }
    return out


def target_collection_cost(n_choices: int) -> Dict[str, Any]:
    return {"prefix_runs_per_image": 1, "A_to_B_segment_runs_per_image": n_choices,
            "suffix_runs_per_image": n_choices * n_choices,
            "note": "never summarise target collection as a lightweight router update"}


@torch.no_grad()
def measure_latency(fn: Callable[[torch.Tensor], Any], device: torch.device, batch: int, n_images: int,
                    warmup: int, hw: int = 32, precision: str = "fp32", seed: int = 0) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 3, hw, hw, generator=g).to(device)
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (precision == "fp16" and device.type == "cuda") \
        else torch.autocast(device_type="cpu", enabled=False)
    with ctx:
        for _ in range(warmup):
            fn(x)
        synchronize(device)
        iters = max(1, int(np.ceil(n_images / batch)))
        times = []
        for _ in range(iters):
            synchronize(device)
            t0 = time.perf_counter()
            fn(x)
            synchronize(device)
            times.append(time.perf_counter() - t0)
    t = np.asarray(times) * 1000.0
    return {"batch_size": batch, "iterations": iters, "images_measured": iters * batch,
            "median_ms_per_batch": float(np.median(t)), "p90_ms_per_batch": float(np.percentile(t, 90)),
            "p99_ms_per_batch": float(np.percentile(t, 99)),
            "throughput_images_per_s": float(batch / (np.median(t) / 1000.0)),
            "precision": precision, "device": str(device)}
