"""Datasets, the fixed stratified split, and a device-resident batch loader.

* Section 10.1: a fixed stratified 40,000 / 5,000 / 5,000 split of the official
  50,000 CIFAR training images into bank-training, route-training and
  validation subsets. The same split (``data.split_seed``) is used by every
  method and every training seed. The official test set is only touched when
  ``eval.final_test`` is enabled.
* Section 9.1: random crop with reflection padding (4 px) and horizontal flip,
  followed by normalisation with training-channel statistics. Statistics are
  computed on the bank-training subset only.
* Target collection uses the clean (un-augmented) view, so every one of the 25
  alternatives sees exactly the same transformed input (Section 7.1).

Images are kept as uint8 on the compute device and augmented there, which is
deterministic given a ``torch.Generator`` and avoids DataLoader worker drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .utils import hash_indices


@dataclass
class RawData:
    train_x: torch.Tensor          # (N, 3, 32, 32) uint8
    train_y: torch.Tensor          # (N,) int64
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int
    class_names: List[str]


# ------------------------------------------------------------------ loading
def _synthetic(n_train: int, n_test: int, k: int, seed: int = 1234) -> RawData:
    """Class-conditional textured images. Only for smoke tests."""
    g = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    protos = np.zeros((k, 3, 40, 40), dtype=np.float32)
    for c in range(k):
        for ch in range(3):
            for _ in range(3):
                fx, fy = g.uniform(0.05, 0.45, size=2)
                ph = g.uniform(0, 2 * math.pi)
                protos[c, ch] += np.sin(2 * math.pi * (fx * xx + fy * yy) / 4 + ph)
    protos = (protos - protos.min()) / (protos.max() - protos.min())

    def draw(n: int, rs: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        y = np.arange(n) % k
        rs.shuffle(y)
        x = np.empty((n, 3, 32, 32), dtype=np.uint8)
        for i in range(n):
            ox, oy = rs.integers(0, 9, size=2)
            img = protos[y[i], :, oy:oy + 32, ox:ox + 32]
            img = img + rs.normal(0, 0.35, size=img.shape)
            x[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
        return x, y

    tx, ty = draw(n_train, np.random.default_rng(seed + 1))
    vx, vy = draw(n_test, np.random.default_rng(seed + 2))
    return RawData(
        torch.from_numpy(tx), torch.from_numpy(ty).long(),
        torch.from_numpy(vx), torch.from_numpy(vy).long(),
        k, [f"class_{c}" for c in range(k)],
    )


def load_raw(cfg: Config) -> RawData:
    d = cfg.data
    if d.dataset == "synthetic":
        return _synthetic(d.synthetic_train, d.synthetic_test, d.synthetic_classes)
    import torchvision

    cls = torchvision.datasets.CIFAR100 if d.dataset == "cifar100" else torchvision.datasets.CIFAR10
    tr = cls(root=d.root, train=True, download=d.download)
    te = cls(root=d.root, train=False, download=d.download)

    def conv(ds) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
        y = torch.as_tensor(ds.targets, dtype=torch.long)
        return x, y

    tx, ty = conv(tr)
    vx, vy = conv(te)
    return RawData(tx, ty, vx, vy, len(tr.classes), list(tr.classes))


# ------------------------------------------------------------------ splits
def stratified_split(labels: torch.Tensor, sizes: Dict[str, int], seed: int) -> Dict[str, np.ndarray]:
    """Disjoint stratified subsets with exact requested sizes.

    Per-class quotas are allocated with the largest-remainder rule so that
    each subset has the requested total and class proportions match the
    source as closely as integer counts allow.
    """
    y = labels.cpu().numpy()
    n = len(y)
    total = sum(sizes.values())
    if total > n:
        raise ValueError(f"requested {total} examples but only {n} available")
    classes, counts = np.unique(y, return_counts=True)
    rng = np.random.default_rng(seed)
    per_class_idx = {c: rng.permutation(np.flatnonzero(y == c)) for c in classes}

    names = list(sizes.keys())
    quotas = {}
    for name in names:
        exact = counts * sizes[name] / n
        base = np.floor(exact).astype(int)
        short = sizes[name] - base.sum()
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:short]] += 1
        quotas[name] = base

    # ensure no class is over-allocated (possible only for tiny datasets)
    alloc = sum(quotas.values())
    if np.any(alloc > counts):
        raise ValueError("split sizes too large for per-class counts")

    out: Dict[str, List[int]] = {name: [] for name in names}
    for ci, c in enumerate(classes):
        pool = per_class_idx[c]
        start = 0
        for name in names:
            q = quotas[name][ci]
            out[name].extend(pool[start:start + q].tolist())
            start += q
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in out.items()}


def make_splits(cfg: Config, raw: RawData) -> Dict[str, np.ndarray]:
    d = cfg.data
    if d.dataset == "synthetic":
        n = len(raw.train_y)
        sizes = {"bank": int(0.8 * n), "route": int(0.1 * n), "val": n - int(0.8 * n) - int(0.1 * n)}
    else:
        sizes = {"bank": d.bank_size, "route": d.route_size, "val": d.val_size}
    splits = stratified_split(raw.train_y, sizes, d.split_seed)
    splits["test"] = np.arange(len(raw.test_y), dtype=np.int64)
    return splits


def split_hashes(splits: Dict[str, np.ndarray]) -> Dict[str, str]:
    return {k: hash_indices(v) for k, v in splits.items()}


def channel_stats(x_uint8: torch.Tensor) -> Tuple[List[float], List[float]]:
    x = x_uint8.float() / 255.0
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    return mean.tolist(), std.tolist()


# ------------------------------------------------------------------ loader
class DeviceData:
    """uint8 images resident on ``device`` with deterministic GPU augmentation."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, mean: List[float], std: List[float],
                 device: torch.device, on_device: bool = True):
        self.device = device
        store = device if on_device else torch.device("cpu")
        self.x = x.to(store)
        self.y = y.to(store)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def normalize(self, xb_uint8: torch.Tensor) -> torch.Tensor:
        return (xb_uint8.to(self.device).float() / 255.0 - self.mean) / self.std

    @staticmethod
    def augment(xb: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
        """Random crop with reflection padding + horizontal flip, batched."""
        b, c, h, w = xb.shape
        padded = F.pad(xb.float(), (pad, pad, pad, pad), mode="reflect")
        dev = xb.device
        # random numbers come from the generator's device (CPU generator on MPS), then move
        dy = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device=gen.device).to(dev)
        dx = torch.randint(0, 2 * pad + 1, (b,), generator=gen, device=gen.device).to(dev)
        rows = dy[:, None] + torch.arange(h, device=dev)[None, :]          # (b, h)
        cols = dx[:, None] + torch.arange(w, device=dev)[None, :]          # (b, w)
        bidx = torch.arange(b, device=dev)[:, None, None]
        out = padded.permute(0, 2, 3, 1)[bidx, rows[:, :, None], cols[:, None, :]]  # (b,h,w,c)
        out = out.permute(0, 3, 1, 2)
        flip = (torch.rand(b, generator=gen, device=gen.device) < 0.5).to(dev)
        out = torch.where(flip[:, None, None, None], out.flip(3), out)
        return out

    def batches(self, indices: np.ndarray, batch_size: int, shuffle: bool = False,
                augment: bool = False, gen: Optional[torch.Generator] = None,
                drop_last: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        idx = torch.as_tensor(indices, dtype=torch.long)
        if shuffle:
            if gen is None:
                raise ValueError("shuffle requires a generator for determinism")
            perm = torch.randperm(len(idx), generator=gen, device=gen.device).cpu()
            idx = idx[perm]
        n = len(idx)
        stop = n - (n % batch_size) if drop_last else n
        for s in range(0, stop, batch_size):
            bi = idx[s:s + batch_size]
            if len(bi) == 0:
                break
            bi_store = bi.to(self.x.device)
            xb = self.x[bi_store].to(self.device)
            yb = self.y[bi_store].to(self.device)
            if augment:
                if gen is None:
                    raise ValueError("augmentation requires a generator for determinism")
                xb = self.augment(xb, gen)
            xb = (xb.float() / 255.0 - self.mean) / self.std
            yield xb, yb, bi.to(self.device)


def mix_batch(x: torch.Tensor, y: torch.Tensor, num_classes: int, mode: str, alpha: float,
              gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """MixUp / CutMix producing soft targets (recipe experiment, Section 9.1)."""
    y1 = F.one_hot(y, num_classes).float()
    if mode == "none":
        return x, y1
    lam = float(np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen,
                                                          device=gen.device).item())).beta(alpha, alpha))
    perm = torch.randperm(x.size(0), generator=gen, device=gen.device).to(x.device)
    if mode == "mixup":
        xm = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[2:]
        cut = math.sqrt(1 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy = int(torch.randint(0, h, (1,), generator=gen, device=gen.device).item())
        cx = int(torch.randint(0, w, (1,), generator=gen, device=gen.device).item())
        y0, y1b = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x0, x1b = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        xm = x.clone()
        xm[:, :, y0:y1b, x0:x1b] = x[perm][:, :, y0:y1b, x0:x1b]
        lam = 1 - ((y1b - y0) * (x1b - x0) / (h * w))
    else:
        raise ValueError(f"unknown mix mode {mode}")
    return xm, lam * y1 + (1 - lam) * y1[perm]


class DataBundle:
    """Everything a stage needs: splits, hashes, normalisation and loaders."""

    def __init__(self, cfg: Config, device: torch.device, raw: Optional[RawData] = None):
        self.raw = raw if raw is not None else load_raw(cfg)
        self.splits = make_splits(cfg, self.raw)
        self.hashes = split_hashes(self.splits)
        self.mean, self.std = channel_stats(self.raw.train_x[torch.as_tensor(self.splits["bank"])])
        self.num_classes = self.raw.num_classes
        self.class_names = self.raw.class_names
        self.train = DeviceData(self.raw.train_x, self.raw.train_y, self.mean, self.std, device,
                                cfg.data.data_on_device)
        self.test = DeviceData(self.raw.test_x, self.raw.test_y, self.mean, self.std, device,
                               cfg.data.data_on_device)

    def source(self, split: str) -> Tuple[DeviceData, np.ndarray]:
        if split == "test":
            return self.test, self.splits["test"]
        return self.train, self.splits[split]

    def assert_disjoint(self) -> None:
        b, r, v = (set(self.splits[k].tolist()) for k in ("bank", "route", "val"))
        if b & r or b & v or r & v:
            raise AssertionError("bank / route / validation splits overlap")
