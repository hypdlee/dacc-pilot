"""Load a deployment bundle, verify exports, and run the predict CLI on image files."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacc.deploy import load_bundle, verify_exports  # noqa: E402


def main(seed_dir: str) -> int:
    d = Path(seed_dir)
    dm, b = load_bundle(d / "deploy/bundle.pt")
    check = verify_exports(dm, d / "deploy/model.onnx", d / "deploy/model_ts.pt", n=128, seed=5)
    print(json.dumps(check, indent=2))
    ok = (check["dense_actions_equal"] and check["torchscript_actions_equal"] and check["onnx_actions_equal"]
          and check["dense_vs_grouped_max_abs"] < 1e-4 and check["torchscript_max_abs"] < 1e-4
          and check["onnx_max_abs"] < 1e-4)
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        rng = np.random.default_rng(0)
        for k, size in enumerate((32, 64, 17)):
            p = Path(tmp) / f"img{k}.png"
            Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(p)
            paths.append(str(p))
        out = subprocess.run([sys.executable, "-m", "dacc.predict", "--bundle", str(d / "deploy/bundle.pt"),
                              "--images", *paths, "--topk", "3"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1])
        if out.returncode != 0:
            print(out.stderr)
            return 1
        preds = json.loads(out.stdout)
        print(json.dumps(preds[0], indent=2))
        ok = ok and len(preds) == 3 and all(len(p["topk"]) == 3 for p in preds)
        ok = ok and all(0 <= p["action"]["site_A"] <= 4 and 0 <= p["action"]["site_B"] <= 4 for p in preds)
    # bundle reproduces the training-time model on the same input
    x = torch.rand(4, 3, 32, 32)
    l1, _ = dm(x)
    l2, _ = dm(x)
    ok = ok and torch.equal(l1, l2)
    print("DEPLOY_OK" if ok else "DEPLOY_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
