"""Deployable inference module, bundle save/load and ONNX / TorchScript export.

Two execution modes with identical outputs:
* ``grouped`` (default, Section 13.3): samples grouped by choice, only the
  selected correction runs, zero-choice samples bypass the path.
* ``dense``: one-hot weights over all experts, free of data-dependent control
  flow so it exports to ONNX / TorchScript. It computes every expert's
  depthwise branch, so it is NOT the efficient path and must not be used for
  latency claims.

Inputs are raw images in [0, 1], shape (B, 3, 32, 32); normalisation with the
bank-split statistics is built in. No label is ever an input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import Config, config_from_dict
from .controller import SequentialController, descriptor
from .model import CorrectedWRN, action_table, build_model


def _onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return (torch.arange(n, device=idx.device)[None, :] == idx[:, None]).to(torch.float32)


class DeployModel(nn.Module):
    def __init__(self, model: CorrectedWRN, ctrl: nn.Module, desc_kind: str, mean: List[float],
                 std: List[float], mode: str = "grouped", cost_lambda: float = 0.0,
                 action_cost: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model.eval()
        self.ctrl = ctrl.eval()
        self.desc_kind = desc_kind
        self.mode = mode
        self.sequential = isinstance(ctrl, SequentialController)
        self.n = model.n_choices
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.cost_lambda = float(cost_lambda)
        cost = action_cost if action_cost is not None else torch.zeros(self.n * self.n)
        self.register_buffer("action_cost", cost.float().flatten())

    def _decide_joint(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.ctrl(descriptor(h, self.desc_kind)).flatten(1)
        if self.cost_lambda != 0.0:
            q = q + self.cost_lambda * self.action_cost[None, :]
        a = torch.argmin(q, 1)                                         # first minimum -> lexicographic ties
        return a // self.n, a % self.n

    def forward(self, x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (x01 - self.mean) / self.std
        m = self.model
        h = m.prefix(x)
        dense = self.mode == "dense"
        if self.sequential:
            i = torch.argmin(self.ctrl.predict_a(descriptor(h, self.desc_kind)), 1)
            ha = m.site_a(h, weights=_onehot(i, self.n)) if dense else m.site_a(h, choice=i)
            hb = m.middle(ha)
            j = torch.argmin(self.ctrl.predict_b(descriptor(hb, self.desc_kind), i), 1)
        else:
            i, j = self._decide_joint(h)
            ha = m.site_a(h, weights=_onehot(i, self.n)) if dense else m.site_a(h, choice=i)
            hb = m.middle(ha)
        zb = m.site_b(hb, weights=_onehot(j, self.n)) if dense else m.site_b(hb, choice=j)
        return m.suffix(zb), torch.stack([i, j], 1)


def save_bundle(path: Path, cfg: Config, model: CorrectedWRN, ctrl: nn.Module, ctrl_kind: str,
                ctrl_build: Dict[str, Any], mean: List[float], std: List[float], class_names: List[str],
                extra: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": cfg.to_dict(),
        "num_classes": model.num_classes,
        "model_state": model.state_dict(),
        "controller_state": ctrl.state_dict(),
        "controller_kind": ctrl_kind,
        "controller_build": ctrl_build,
        "descriptor": cfg.controller.descriptor,
        "mean": mean, "std": std,
        "class_names": class_names,
        "action_mapping": action_table(model.n_choices),
        "input": "RGB float in [0,1], (B,3,32,32); normalisation built in",
        "extra": extra or {},
    }, path)


def controller_from_build(build: Dict[str, Any]) -> nn.Module:
    from .controller import build_controller

    return build_controller(build["kind"], build["d_in"], build["hidden"], build["n"], build.get("rank", 2),
                            d_b=build.get("d_b"), hidden_b=build.get("hidden_b"))


def controller_build_spec(ctrl: nn.Module) -> Dict[str, Any]:
    if isinstance(ctrl, SequentialController):
        return {"kind": "sequential", "d_in": ctrl.head_a.fc1.in_features, "hidden": ctrl.head_a.fc1.out_features,
                "n": ctrl.n, "d_b": ctrl.head_b.fc1.in_features - ctrl.n, "hidden_b": ctrl.head_b.fc1.out_features}
    spec = {"kind": ctrl.kind, "d_in": ctrl.mlp.fc1.in_features, "hidden": ctrl.mlp.fc1.out_features, "n": ctrl.n}
    if ctrl.kind == "bilinear":
        spec["rank"] = ctrl.rank
    return spec


def load_bundle(path: Path, mode: str = "grouped", device: str = "cpu") -> Tuple[DeployModel, Dict[str, Any]]:
    b = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(b["config"])
    model = build_model(cfg, b["num_classes"])
    model.load_state_dict(b["model_state"])
    ctrl = controller_from_build(b["controller_build"])
    ctrl.load_state_dict(b["controller_state"])
    dm = DeployModel(model, ctrl, b["descriptor"], b["mean"], b["std"], mode=mode).to(device).eval()
    return dm, b


@torch.no_grad()
def export_onnx(dm: DeployModel, path: Path, opset: int = 17) -> None:
    dm_dense = DeployModel(dm.model, dm.ctrl, dm.desc_kind, dm.mean.flatten().tolist(),
                           dm.std.flatten().tolist(), mode="dense").eval()
    x = torch.rand(2, 3, 32, 32)
    torch.onnx.export(dm_dense, (x,), str(path), input_names=["image"], output_names=["logits", "actions"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "actions": {0: "batch"}},
                      opset_version=opset, dynamo=False)


@torch.no_grad()
def export_torchscript(dm: DeployModel, path: Path) -> None:
    dm_dense = DeployModel(dm.model, dm.ctrl, dm.desc_kind, dm.mean.flatten().tolist(),
                           dm.std.flatten().tolist(), mode="dense").eval()
    traced = torch.jit.trace(dm_dense, torch.rand(2, 3, 32, 32), check_trace=False)
    traced.save(str(path))


@torch.no_grad()
def verify_exports(dm: DeployModel, onnx_path: Optional[Path], ts_path: Optional[Path], n: int = 64,
                   seed: int = 0) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 32, 32, generator=g)
    dm = dm.cpu().eval()
    ref_logits, ref_actions = dm(x)
    out: Dict[str, Any] = {"n": n}
    dense = DeployModel(dm.model, dm.ctrl, dm.desc_kind, dm.mean.flatten().tolist(), dm.std.flatten().tolist(),
                        mode="dense").eval()
    dl, da = dense(x)
    out["dense_vs_grouped_max_abs"] = float((dl - ref_logits).abs().max())
    out["dense_actions_equal"] = bool(torch.equal(da, ref_actions))
    if ts_path is not None and Path(ts_path).exists():
        ts = torch.jit.load(str(ts_path))
        tl, ta = ts(x)
        out["torchscript_max_abs"] = float((tl - ref_logits).abs().max())
        out["torchscript_actions_equal"] = bool(torch.equal(ta, ref_actions))
    if onnx_path is not None and Path(onnx_path).exists():
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            ol, oa = sess.run(None, {"image": x.numpy()})
            out["onnx_max_abs"] = float(abs(torch.from_numpy(ol) - ref_logits).max())
            out["onnx_actions_equal"] = bool(torch.equal(torch.from_numpy(oa).long(), ref_actions.long()))
        except ImportError:
            out["onnx_check"] = "onnxruntime not installed; ONNX file written but not executed"
    return out
