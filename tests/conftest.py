import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dacc.model import CorrectedWRN  # noqa: E402


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    m = CorrectedWRN(num_classes=10, depth=16, widen=1, init_seed=3)
    # give BN non-trivial running statistics so eval-mode tests are meaningful
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(16, 3, 32, 32))
    # make corrections non-negligible
    with torch.no_grad():
        m.corr_a.gain.fill_(0.5)
        m.corr_b.gain.fill_(0.5)
    return m.eval()


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(12, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (12,), generator=g)
    return x, y
