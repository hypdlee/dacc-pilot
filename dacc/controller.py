"""Controllers that predict per-action losses (Section 8).

Descriptor (Eq. 6): concat of channel means, channel standard deviations and
a 2x2 adaptive average pool of the features h before site A. For 128 channels
this is 768 values. Ablations: ``mean`` (C values) and ``meanstd`` (2C).

Controllers return predicted loss tables q of shape (B, n, n):
* ``joint``      (P1)  full 5x5 table from one MLP (25 outputs)
* ``additive``   (B5)  q_ij = c + u_i + v_j with u_0 = v_0 = 0 (Eq. 14)
* ``bilinear``   (P2)  additive - sum_r a_ir b_jr, index-0 factors zero (Eq. 16)
* ``sequential`` (B6)  head A predicts min_j L_ij from h; after executing A,
                       head B predicts L_i. from B's actual input (plus i).

Training objective (Eq. 10-13): centre target and prediction over actions per
example, mean Huber (transition 0.1) or squared error, optional beta * KL(t||pi)
with t = softmax(-L / tau), pi = softmax(-q / tau).

Decision (Eq. 17-18): argmin_a q_a (+ lambda C_a), ties broken toward the
lowest row-major index, i.e. lexicographically from (0, 0).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STD_EPS = 1e-5


# ------------------------------------------------------------------ descriptor
def descriptor(h: torch.Tensor, kind: str = "full") -> torch.Tensor:
    """d(h) = [mu(h), sigma(h), vec(Pool_2x2(h))] (Eq. 6), computed in float32."""
    h = h.float()
    mu = h.mean(dim=(2, 3))
    if kind == "mean":
        return mu
    var = h.var(dim=(2, 3), unbiased=False)
    sigma = torch.sqrt(var + STD_EPS)
    if kind == "meanstd":
        return torch.cat([mu, sigma], 1)
    if kind == "full":
        pool = F.adaptive_avg_pool2d(h, 2).flatten(1)
        return torch.cat([mu, sigma, pool], 1)
    raise ValueError(f"unknown descriptor {kind}")


def descriptor_dim(channels: int, kind: str = "full") -> int:
    return {"mean": channels, "meanstd": 2 * channels, "full": 6 * channels}[kind]


# ------------------------------------------------------------------ modules
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def mlp_params(d_in: int, hidden: int, d_out: int) -> int:
    return d_in * hidden + hidden + hidden * d_out + d_out


def hidden_for_params(d_in: int, d_out: int, target: int) -> int:
    """Hidden width whose MLP parameter count is closest to ``target``."""
    return max(1, int(round((target - d_out) / (d_in + 1 + d_out))))


class Normalizer(nn.Module):
    """Fixed standardisation of descriptor inputs (fitted on route-train only)."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: torch.Tensor) -> None:
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class JointController(nn.Module):
    kind = "joint"

    def __init__(self, d_in: int, hidden: int, n: int = 5):
        super().__init__()
        self.n = n
        self.norm = Normalizer(d_in)
        self.mlp = MLP(d_in, hidden, n * n)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(d)).view(-1, self.n, self.n)


class AdditiveController(nn.Module):
    kind = "additive"

    def __init__(self, d_in: int, hidden: int, n: int = 5):
        super().__init__()
        self.n = n
        self.norm = Normalizer(d_in)
        self.mlp = MLP(d_in, hidden, 1 + 2 * (n - 1))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        o = self.mlp(self.norm(d))
        n = self.n
        c = o[:, :1]
        zero = o.new_zeros(o.size(0), 1)
        u = torch.cat([zero, o[:, 1:n]], 1)
        v = torch.cat([zero, o[:, n:2 * n - 1]], 1)
        return c[:, :, None] + u[:, :, None] + v[:, None, :]


class BilinearController(nn.Module):
    kind = "bilinear"

    def __init__(self, d_in: int, hidden: int, n: int = 5, rank: int = 2):
        super().__init__()
        self.n, self.rank = n, rank
        self.norm = Normalizer(d_in)
        self.mlp = MLP(d_in, hidden, 1 + 2 * (n - 1) + 2 * rank * (n - 1))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        o = self.mlp(self.norm(d))
        n, r, b = self.n, self.rank, o.size(0)
        zero = o.new_zeros(b, 1)
        c = o[:, :1]
        u = torch.cat([zero, o[:, 1:n]], 1)
        v = torch.cat([zero, o[:, n:2 * n - 1]], 1)
        off = 2 * n - 1
        fa = o[:, off:off + r * (n - 1)].view(b, n - 1, r)
        fb = o[:, off + r * (n - 1):].view(b, n - 1, r)
        za = o.new_zeros(b, 1, r)
        fa = torch.cat([za, fa], 1)                                   # (b, n, r), row 0 zero
        fb = torch.cat([za, fb], 1)
        inter = torch.einsum("bir,bjr->bij", fa, fb)
        return c[:, :, None] + u[:, :, None] + v[:, None, :] - inter


class SequentialController(nn.Module):
    """B6: decide i from h, execute A, decide j from B's actual input."""

    kind = "sequential"

    def __init__(self, d_a: int, d_b: int, hidden_a: int, hidden_b: int, n: int = 5):
        super().__init__()
        self.n = n
        self.norm_a = Normalizer(d_a)
        self.norm_b = Normalizer(d_b)
        self.head_a = MLP(d_a, hidden_a, n)
        self.head_b = MLP(d_b + n, hidden_b, n)

    def predict_a(self, d_a: torch.Tensor) -> torch.Tensor:
        return self.head_a(self.norm_a(d_a))

    def predict_b(self, d_b: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(i, self.n).float()
        return self.head_b(torch.cat([self.norm_b(d_b), onehot], 1))


def build_controller(kind: str, d_in: int, hidden: int, n: int = 5, rank: int = 2,
                     d_b: Optional[int] = None, hidden_b: Optional[int] = None) -> nn.Module:
    if kind == "joint":
        return JointController(d_in, hidden, n)
    if kind == "additive":
        return AdditiveController(d_in, hidden, n)
    if kind == "bilinear":
        return BilinearController(d_in, hidden, n, rank)
    if kind == "sequential":
        if d_b is None:
            raise ValueError("sequential controller needs d_b")
        return SequentialController(d_in, d_b, hidden, hidden_b if hidden_b is not None else hidden, n)
    raise ValueError(f"unknown controller kind {kind}")


def trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ objectives
def center(t: torch.Tensor) -> torch.Tensor:
    """Subtract per-example mean over actions (Eq. 10). Works on (B, n, n) or (B, n)."""
    flat = t.flatten(1)
    return (flat - flat.mean(1, keepdim=True)).view_as(t)


def score_loss(q: torch.Tensor, target: torch.Tensor, kind: str = "huber", delta: float = 0.1) -> torch.Tensor:
    qc, tc = center(q).flatten(1), center(target).flatten(1)
    if kind == "huber":
        per = F.huber_loss(qc, tc, reduction="none", delta=delta)
    elif kind == "mse":
        per = (qc - tc) ** 2
    else:
        raise ValueError(f"unknown score loss {kind}")
    return per.mean(1).mean()


def kl_route_loss(q: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    t = F.softmax(-target.flatten(1) / tau, 1)
    log_pi = F.log_softmax(-q.flatten(1) / tau, 1)
    log_t = F.log_softmax(-target.flatten(1) / tau, 1)
    return (t * (log_t - log_pi)).sum(1).mean()


def router_objective(q: torch.Tensor, target: torch.Tensor, kind: str, delta: float,
                     beta: float, tau: float) -> torch.Tensor:
    loss = score_loss(q, target, kind, delta)
    if beta > 0:
        loss = loss + beta * kl_route_loss(q, target, tau)
    return loss


def sequential_targets(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Head-A target min_j L_ij (B, n); head-B targets L (B, n, n) per executed i."""
    return L.min(2).values, L


# ------------------------------------------------------------------ decisions
def decide(q: torch.Tensor, cost: Optional[torch.Tensor] = None, lam: float = 0.0) -> torch.Tensor:
    """Hard argmin over the flattened table; returns (B, 2) actions (i, j).

    ``torch.argmin`` returns the first minimal index, so exact ties resolve to
    the lowest row-major index (lexicographic from (0, 0)).
    """
    n = q.shape[-1]
    s = q.flatten(1)
    if cost is not None and lam != 0.0:
        s = s + lam * cost.flatten()[None, :]
    a = torch.argmin(s, dim=1)
    return torch.stack([a // n, a % n], 1)


def policy_probs(q: torch.Tensor, tau: float) -> torch.Tensor:
    """pi_phi(a|h) = softmax(-q / tau) over the flattened table."""
    return F.softmax(-q.flatten(1) / tau, 1)
