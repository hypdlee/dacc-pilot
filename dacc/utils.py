"""Seeding, devices, hashing, logging and run-artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------- seeding
def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(seed)
    return g


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- hashing
def hash_indices(idx: Iterable[int]) -> str:
    arr = np.asarray(sorted(int(i) for i in idx), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def hash_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- versions
def software_versions() -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# ---------------------------------------------------------------- logging
class JsonlLogger:
    def __init__(self, path: Path, echo: bool = True):
        self.path = path
        self.echo = echo
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(_jsonable(record)) + "\n")
        if self.echo:
            short = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in record.items() if k != "time"}
            print(json.dumps(_jsonable(short)), flush=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if torch.is_tensor(obj):
        return _jsonable(obj.detach().cpu().tolist())
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


@contextmanager
def timed(store: Dict[str, float], key: str, device: Optional[torch.device] = None):
    if device is not None:
        synchronize(device)
    t0 = time.perf_counter()
    yield
    if device is not None:
        synchronize(device)
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def cosine_warmup_lr(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warm-up then cosine decay to zero (per optimizer step)."""
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    if total <= warmup:
        return base
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1.0 + math.cos(math.pi * min(1.0, progress)))


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
