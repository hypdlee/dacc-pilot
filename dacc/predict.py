"""Label-free inference CLI on image files.

    python -m dacc.predict --bundle runs/<name>/seed11/deploy/bundle.pt \
        --images cat.png dog.jpg --topk 5

Images are converted to RGB and resized to 32x32 (bilinear) to match the
CIFAR training resolution. Output is JSON: top-k classes, probabilities and
the executed correction action (i, j) per image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from .deploy import load_bundle


def load_images(paths: List[str], size: int = 32) -> torch.Tensor:
    from PIL import Image

    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="grouped", choices=["grouped", "dense"])
    args = ap.parse_args(argv)

    dm, b = load_bundle(Path(args.bundle), mode=args.mode, device=args.device)
    x = load_images(args.images).to(args.device)
    with torch.no_grad():
        logits, actions = dm(x)
    probs = torch.softmax(logits.float(), 1)
    k = min(args.topk, probs.size(1))
    top = probs.topk(k, 1)
    names = b["class_names"]
    out = []
    for n, p in enumerate(args.images):
        out.append({
            "image": p,
            "action": {"site_A": int(actions[n, 0]), "site_B": int(actions[n, 1])},
            "topk": [{"class": names[int(c)], "index": int(c), "prob": float(v)}
                     for v, c in zip(top.values[n], top.indices[n])],
        })
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
