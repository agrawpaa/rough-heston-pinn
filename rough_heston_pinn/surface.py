"""Neural option surface: a network for total variance w(T, k) and Black-Scholes pricing in torch.

The surface is parameterised as ``w(T, k) = T * softplus(f(sqrt(T / T_max), k / k_scale))`` so that
total variance is positive and vanishes at T = 0 by construction. Prices are obtained by pushing
``w`` through the Black-Scholes formula, which makes the PDE residual a function of the network only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import ndtr

from .sde import OptionQuotes  # noqa: F401  (re-exported for convenience)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y / d x for elementwise y(x), keeping the graph so higher derivatives and training work."""
    (g,) = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)
    return g


def bs_call_tv(w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Normalised Black-Scholes call C/S0 for total variance ``w`` and log-moneyness ``k`` (zero rates)."""
    sw = torch.sqrt(w.clamp_min(1e-12))
    d1 = (-k + 0.5 * w) / sw
    return ndtr(d1) - torch.exp(k) * ndtr(d1 - sw)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class SurfaceNet(nn.Module):
    """MLP mapping (T, k) to positive total variance w(T, k) with w(0, k) = 0."""

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 4,
        t_max: float = 2.0,
        k_scale: float = 0.5,
        init_variance: float = 0.04,
    ) -> None:
        super().__init__()
        self.t_max = float(t_max)
        self.k_scale = float(k_scale)
        dims = [2] + [hidden] * layers
        blocks: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(d_in, d_out), nn.Tanh()]
        out = nn.Linear(hidden, 1)
        nn.init.zeros_(out.weight)  # start from a flat surface with variance ``init_variance``
        nn.init.constant_(out.bias, _inv_softplus(init_variance))
        self.mlp = nn.Sequential(*blocks, out)

    def forward(self, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = torch.stack([torch.sqrt(T / self.t_max), k / self.k_scale], dim=-1)
        return T * F.softplus(self.mlp(x).squeeze(-1))

    def implied_vol(self, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.forward(T, k).clamp_min(1e-12) / T)
