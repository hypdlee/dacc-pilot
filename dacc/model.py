"""WRN backbone with two correction sites (Section 6).

Backbone: CIFAR Wide ResNet, depth 28, widening factor 4 (Zagoruyko and
Komodakis, 2016), pre-activation basic blocks. Stem 3x3 with 16 channels,
three stages of (depth - 4) / 6 blocks with widths 16k, 32k, 64k, final
BN-ReLU, global average pooling and a linear classifier.

Sites: A is the last stride-one block of stage 2, B is the last stride-one
block of stage 3. Both preserve shape.

Correction (Eq. 3-4): R_i(X) = gamma * P(D_i(A(X))), R_0(X) = 0 exactly,
T_i(X) = F_shared(X) + R_i(X). A and P are shared bias-free 1x1 maps,
D_i is an expert-specific depthwise 3x3 operator at rank r = C / 4.
The correction is added to the block output (which already contains the
shortcut); there is no second shortcut and no normalisation inside R.

Joint action index convention (Section 14): a = n_choices * i + j, row-major,
with i the choice at A and j the choice at B, both in 0..n_choices-1.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ actions
def action_index(i: int, j: int, n: int = 5) -> int:
    return n * i + j


def action_pair(a: int, n: int = 5) -> Tuple[int, int]:
    return divmod(int(a), n)


def action_table(n: int = 5) -> List[Tuple[int, int]]:
    """Immutable mapping action index -> (i, j)."""
    return [action_pair(a, n) for a in range(n * n)]


# ------------------------------------------------------------------ blocks
class BasicBlock(nn.Module):
    """Pre-activation WRN basic block. Output = residual branch + shortcut."""

    def __init__(self, in_c: int, out_c: int, stride: int, dropout: float):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.dropout = dropout
        self.equal = in_c == out_c and stride == 1
        self.shortcut = None if self.equal else nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.relu(self.bn1(x))
        sc = x if self.equal else self.shortcut(o)
        out = self.conv1(o)
        out = F.relu(self.bn2(out))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return out + sc


class Correction(nn.Module):
    """Bank of ``num_experts`` low-rank corrections plus the exact zero choice."""

    def __init__(self, channels: int, rank: int, num_experts: int, gain_init: float, untied: bool = False,
                 init_seed: Optional[int] = None):
        super().__init__()
        self.c, self.r, self.e = channels, rank, num_experts
        self.untied = untied
        n_maps = num_experts if untied else 1
        self.reduce = nn.ModuleList([nn.Conv2d(channels, rank, 1, bias=False) for _ in range(n_maps)])
        self.expand = nn.ModuleList([nn.Conv2d(rank, channels, 1, bias=False) for _ in range(n_maps)])
        self.depthwise = nn.Parameter(torch.empty(num_experts, rank, 1, 3, 3))
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.reset_parameters(init_seed)

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        for m in list(self.reduce) + list(self.expand):
            fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * math.sqrt(2.0 / fan_in))
        with torch.no_grad():
            # each expert gets an independent draw (different initialisation per D_i)
            self.depthwise.copy_(torch.randn(self.depthwise.shape, generator=g) * math.sqrt(2.0 / 9.0))

    # -- helpers --------------------------------------------------------
    def _dw(self, a: torch.Tensor, e: int) -> torch.Tensor:
        return F.conv2d(a, self.depthwise[e], padding=1, groups=self.r)

    def _maps(self, e: int) -> Tuple[nn.Conv2d, nn.Conv2d]:
        k = e if self.untied else 0
        return self.reduce[k], self.expand[k]

    # -- discrete execution (deployment, Section 13.3) -------------------
    def forward_choice(self, x: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
        """Correction for per-sample discrete choices (0 = zero correction).

        Samples are grouped by choice; zero-choice samples bypass the path.
        Returns a tensor shaped like ``x`` whose rows for choice 0 are exactly 0.
        """
        out = torch.zeros_like(x)
        nz = torch.nonzero(choice > 0, as_tuple=False).squeeze(1)
        if nz.numel() == 0:
            return out
        xs = x.index_select(0, nz)
        ch = choice.index_select(0, nz)
        if not self.untied:
            a = self.reduce[0](xs)
            z = torch.zeros_like(a)
            for e in range(self.e):
                m = torch.nonzero(ch == e + 1, as_tuple=False).squeeze(1)
                if m.numel() == 0:
                    continue
                z = z.index_copy(0, m, self._dw(a.index_select(0, m), e))
            y = self.expand[0](z)
        else:
            y = torch.zeros_like(xs)
            for e in range(self.e):
                m = torch.nonzero(ch == e + 1, as_tuple=False).squeeze(1)
                if m.numel() == 0:
                    continue
                red, exp = self._maps(e)
                y = y.index_copy(0, m, exp(self._dw(red(xs.index_select(0, m)), e)))
        return out.index_add(0, nz, self.gain * y)

    def forward_choice_looped(self, x: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
        """Per-sample reference loop (numerical verification only)."""
        rows = []
        for b in range(x.size(0)):
            c = int(choice[b])
            if c == 0:
                rows.append(torch.zeros_like(x[b:b + 1]))
            else:
                red, exp = self._maps(c - 1)
                rows.append(self.gain * exp(self._dw(red(x[b:b + 1]), c - 1)))
        return torch.cat(rows, 0)

    # -- soft / dense execution (B3 baseline and export path) -----------
    def forward_weights(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Output mixing with per-sample weights ``w`` of shape (B, E+1).

        Column 0 weights the zero correction and contributes nothing.
        Computes all experts' depthwise outputs (cheap) then mixes linearly.
        Free of data-dependent control flow, hence ONNX/TorchScript friendly.
        """
        b = x.size(0)
        we = w[:, 1:]
        if not self.untied:
            a = self.reduce[0](x)
            weight = self.depthwise.permute(1, 0, 2, 3, 4).reshape(self.r * self.e, 1, 3, 3)
            allz = F.conv2d(a, weight, padding=1, groups=self.r)              # (B, r*E, H, W)
            allz = allz.view(b, self.r, self.e, allz.shape[-2], allz.shape[-1])
            z = (allz * we.view(b, 1, self.e, 1, 1)).sum(2)
            return self.gain * self.expand[0](z)
        y = torch.zeros_like(x)
        for e in range(self.e):
            red, exp = self._maps(e)
            y = y + we[:, e].view(b, 1, 1, 1) * exp(self._dw(red(x), e))
        return self.gain * y

    def forward_kernel_fusion(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Kernel mixing (Eq. 1): fuse per-sample depthwise kernels, then convolve.

        Only valid for tied maps. Used to verify equivalence with output mixing.
        """
        if self.untied:
            raise ValueError("kernel fusion requires tied reduction/expansion maps")
        b, _, h, wd = x.shape
        a = self.reduce[0](x)
        k = torch.einsum("be,erkij->brkij", w[:, 1:], self.depthwise)          # (B, r, 1, 3, 3)
        z = F.conv2d(a.reshape(1, b * self.r, h, wd), k.reshape(b * self.r, 1, 3, 3),
                     padding=1, groups=b * self.r)
        z = z.view(b, self.r, h, wd)
        return self.gain * self.expand[0](z)

    def params_excluding_gain(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if n != "gain")

    def macs_selected(self, h: int, w: int) -> int:
        """Eq. 27: H W (2 C r + 9 r) for one selected expert."""
        return h * w * (2 * self.c * self.r + 9 * self.r)


class CorrectedWRN(nn.Module):
    def __init__(self, num_classes: int, depth: int = 28, widen: int = 4, stem: int = 16,
                 dropout: float = 0.0, num_experts: int = 4, rank_divisor: int = 4,
                 gain_init: float = 0.05, untied: bool = False,
                 site_ranks: Optional[Tuple[int, int]] = None, init_seed: Optional[int] = None):
        super().__init__()
        n = (depth - 4) // 6
        if n < 2:
            raise ValueError("need >= 2 blocks per stage")
        widths = [16 * widen, 32 * widen, 64 * widen]
        self.widths = widths
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.n_choices = num_experts + 1
        self.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False)

        def stage(in_c: int, out_c: int, stride: int) -> nn.Sequential:
            blocks = [BasicBlock(in_c, out_c, stride, dropout)]
            blocks += [BasicBlock(out_c, out_c, 1, dropout) for _ in range(n - 1)]
            return nn.Sequential(*blocks)

        self.stage1 = stage(stem, widths[0], 1)
        self.stage2 = stage(widths[0], widths[1], 2)
        self.stage3 = stage(widths[1], widths[2], 2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.fc = nn.Linear(widths[2], num_classes)

        ra, rb = site_ranks if site_ranks is not None else (widths[1] // rank_divisor, widths[2] // rank_divisor)
        self.corr_a = Correction(widths[1], ra, num_experts, gain_init, untied,
                                 init_seed=None if init_seed is None else init_seed + 1)
        self.corr_b = Correction(widths[2], rb, num_experts, gain_init, untied,
                                 init_seed=None if init_seed is None else init_seed + 2)
        self._init_backbone()

    def _init_backbone(self) -> None:
        for name, m in self.named_modules():
            if name.startswith("corr_"):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------- decomposition
    def prefix(self, x: torch.Tensor) -> torch.Tensor:
        """Features h immediately before site A (input of its block)."""
        x = self.stem(x)
        x = self.stage1(x)
        for blk in list(self.stage2)[:-1]:
            x = blk(x)
        return x

    def site_a(self, h: torch.Tensor, choice: Optional[torch.Tensor] = None,
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._site(self.stage2[-1], self.corr_a, h, choice, weights)

    def middle(self, ha: torch.Tensor) -> torch.Tensor:
        """Everything between the two site boundaries; output is B's input."""
        x = ha
        for blk in list(self.stage3)[:-1]:
            x = blk(x)
        return x

    def site_b(self, hb: torch.Tensor, choice: Optional[torch.Tensor] = None,
               weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._site(self.stage3[-1], self.corr_b, hb, choice, weights)

    def suffix(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.bn(z))
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z)

    @staticmethod
    def _site(block: nn.Module, corr: Correction, x: torch.Tensor, choice: Optional[torch.Tensor],
              weights: Optional[torch.Tensor]) -> torch.Tensor:
        out = block(x)
        if weights is not None:
            return out + corr.forward_weights(x, weights)
        if choice is not None:
            return out + corr.forward_choice(x, choice)
        return out

    # ---------------------------------------------------- full forwards
    def forward(self, x: torch.Tensor, actions: Optional[torch.Tensor] = None,
                weights_a: Optional[torch.Tensor] = None,
                weights_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``actions``: (B, 2) long tensor of (i, j); None means (0, 0)."""
        h = self.prefix(x)
        if weights_a is not None or weights_b is not None:
            ha = self.site_a(h, weights=weights_a)
            hb = self.middle(ha)
            return self.suffix(self.site_b(hb, weights=weights_b))
        ca = actions[:, 0] if actions is not None else None
        cb = actions[:, 1] if actions is not None else None
        ha = self.site_a(h, choice=ca)
        hb = self.middle(ha)
        return self.suffix(self.site_b(hb, choice=cb))

    def forward_static(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, actions=None)

    def tail(self, h: torch.Tensor, actions: Optional[torch.Tensor]) -> torch.Tensor:
        """Everything after the prefix, for discrete actions (B, 2) or None = (0, 0)."""
        ca = actions[:, 0] if actions is not None else None
        cb = actions[:, 1] if actions is not None else None
        return self.suffix(self.site_b(self.middle(self.site_a(h, choice=ca)), choice=cb))

    # ---------------------------------------------------- parameter groups
    def correction_parameters(self, include_gain: bool = True) -> List[nn.Parameter]:
        ps = []
        for corr in (self.corr_a, self.corr_b):
            for n, p in corr.named_parameters():
                if n == "gain" and not include_gain:
                    continue
                ps.append(p)
        return ps

    def gain_parameters(self) -> List[nn.Parameter]:
        return [self.corr_a.gain, self.corr_b.gain]

    def shared_parameters(self) -> List[nn.Parameter]:
        ids = {id(p) for p in self.correction_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def shared_param_groups(self, weight_decay: float) -> List[Dict]:
        """Weight decay on conv/linear weights only, not on BN or biases."""
        ids = {id(p) for p in self.correction_parameters()}
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if id(p) in ids:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def site_shapes(self, input_hw: int = 32) -> Dict[str, Tuple[int, int, int]]:
        return {"A": (self.widths[1], input_hw // 2, input_hw // 2),
                "B": (self.widths[2], input_hw // 4, input_hw // 4)}


def build_model(cfg, num_classes: int, site_ranks: Optional[Tuple[int, int]] = None,
                num_experts: Optional[int] = None, init_seed: Optional[int] = None) -> CorrectedWRN:
    m = cfg.model
    return CorrectedWRN(
        num_classes=num_classes, depth=m.depth, widen=m.widen, stem=m.stem_channels,
        dropout=m.dropout, num_experts=num_experts if num_experts is not None else m.num_experts,
        rank_divisor=m.rank_divisor, gain_init=m.gain_init, untied=m.untied,
        site_ranks=site_ranks, init_seed=init_seed,
    )


def bn_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.endswith("running_mean") or k.endswith("running_var") or k.endswith("num_batches_tracked")}


def set_bn_eval(model: nn.Module) -> None:
    """Freeze BatchNorm running statistics (keeps other modules' mode)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
