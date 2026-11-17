"""Physics-informed model: Markovian-projection Dupire residual for Rough Heston.

Physics
-------
With zero rates and S0 = 1, call prices c(T, k) in log-moneyness k = ln(K) obey Dupire's forward PDE

    dc/dT = 1/2 sigma_loc^2(T, k) (d^2c/dk^2 - dc/dk).

The rough variance process is not Markovian, so the local variance is obtained by a Markovian
(Gyongy) projection, sigma_loc^2(T, k) = E[V_T | X_T = k] with X_T = ln S_T. We evaluate it with a
Gaussian (linear) closure of the pair (X_T, V_T):

    E[V_T | X_T = k] ~ Vbar(T) + beta(T) (k + I(T)/2),
    Vbar(T) = theta + (V0 - theta) E_alpha(-lam T^alpha)          (exact mean of V_T, Mittag-Leffler),
    I(T)    = int_0^T Vbar(s) ds                                   (variance of X_T to leading order),
    beta(T) = rho nu (int_0^T K(T - s) Vbar(s) ds) / I(T),         (first order in nu)

with alpha = H + 1/2 and K(u) = u^(alpha - 1) / Gamma(alpha). This captures the level and the
H-dependent skew of the surface; it does not capture smile curvature from vol-of-vol (see README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .surface import SurfaceNet, bs_call_tv, grad

PARAM_NAMES = ("H", "nu", "theta", "rho", "V0")
BOUNDS: dict[str, tuple[float, float]] = {
    "H": (0.02, 0.49),
    "nu": (0.02, 2.0),
    "theta": (0.005, 0.5),
    "rho": (-0.99, 0.99),
    "V0": (0.005, 0.5),
}
DEFAULT_INIT: dict[str, float] = {"H": 0.25, "nu": 0.4, "theta": 0.05, "rho": -0.5, "V0": 0.05}


class RoughHestonParams(nn.Module):
    """Learnable Rough Heston parameters, each squashed into a box through a sigmoid.

    The mean-reversion speed ``lam`` is held fixed (it is weakly identified from a single snapshot
    of the surface); H, nu, theta, rho and V0 are calibrated.
    """

    def __init__(self, lam: float = 1.0, init: dict[str, float] | None = None) -> None:
        super().__init__()
        self.lam = float(lam)
        start = {**DEFAULT_INIT, **(init or {})}
        self.raw = nn.ParameterDict()
        for name in PARAM_NAMES:
            lo, hi = BOUNDS[name]
            u = (start[name] - lo) / (hi - lo)
            u = min(max(u, 1e-4), 1.0 - 1e-4)
            self.raw[name] = nn.Parameter(torch.tensor(float(np.log(u / (1.0 - u)))))

    def get(self, name: str) -> torch.Tensor:
        lo, hi = BOUNDS[name]
        return lo + (hi - lo) * torch.sigmoid(self.raw[name])

    def as_floats(self) -> dict[str, float]:
        return {n: float(self.get(n).detach().cpu()) for n in PARAM_NAMES}


def mittag_leffler(alpha: torch.Tensor, z: torch.Tensor, n_terms: int = 40) -> torch.Tensor:
    """Mittag-Leffler function E_alpha(z) = sum_n z^n / Gamma(alpha n + 1), evaluated in float64.

    Accurate for |z| up to roughly 20, which covers lam * T^alpha for the maturities used here.
    ``z`` must be non-zero (the closure only evaluates it at strictly positive times).
    """
    zd, ad = z.double(), alpha.double()
    n = torch.arange(n_terms, device=z.device, dtype=torch.float64)
    terms = zd.unsqueeze(-1) ** n * torch.exp(-torch.lgamma(ad * n + 1.0))
    return terms.sum(-1).to(z.dtype)


class MarkovianProjection(nn.Module):
    """Closed-form-plus-quadrature local variance sigma_loc^2(T, k; H, nu, theta, rho, V0)."""

    def __init__(self, params: RoughHestonParams, n_quad: int = 24, variance_floor: float = 1e-4):
        super().__init__()
        self.params = params
        self.floor = variance_floor
        x, w = np.polynomial.legendre.leggauss(n_quad)
        self.register_buffer("u", torch.tensor(0.5 * (x + 1.0), dtype=torch.float64))
        self.register_buffer("wq", torch.tensor(0.5 * w, dtype=torch.float64))

    def mean_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Vbar(t) = E[V_t] for t > 0."""
        H, theta, V0 = self.params.get("H"), self.params.get("theta"), self.params.get("V0")
        alpha = H + 0.5
        z = -self.params.lam * t.clamp_min(1e-9) ** alpha
        return theta + (V0 - theta) * mittag_leffler(alpha, z)

    def local_variance(self, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        H, nu, rho = self.params.get("H"), self.params.get("nu"), self.params.get("rho")
        alpha = H + 0.5
        u, wq = self.u.to(T.dtype), self.wq.to(T.dtype)
        Tc = T.unsqueeze(-1)

        # I(T) = int_0^T Vbar(s) ds
        integral_v = T * (wq * self.mean_variance(Tc * u)).sum(-1)
        # int_0^T K(T-s) Vbar(s) ds, after substituting T - s = T z^(1/alpha) to remove the singularity
        s2 = Tc * (1.0 - u ** (1.0 / alpha))
        conv = T**alpha / torch.exp(torch.lgamma(alpha + 1.0)) * (wq * self.mean_variance(s2)).sum(-1)

        beta = rho * nu * conv / integral_v.clamp_min(1e-12)
        raw = self.mean_variance(T) + beta * (k + 0.5 * integral_v)
        # smooth floor keeps the diffusion coefficient positive with non-vanishing gradients
        return self.floor + F.softplus((raw - self.floor) * 50.0) / 50.0


# --------------------------------------------------------------------------- residual and penalties


def dupire_residual(
    net: SurfaceNet, sigma2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    T: torch.Tensor, k: torch.Tensor,
) -> torch.Tensor:
    """Residual dc/dT - 1/2 sigma^2 (c_kk - c_k) at points where ``T`` and ``k`` require grad."""
    c = bs_call_tv(net(T, k), k)
    c_T = grad(c, T)
    c_k = grad(c, k)
    c_kk = grad(c_k, k)
    return c_T - 0.5 * sigma2(T, k) * (c_kk - c_k)


def arbitrage_penalties(net: SurfaceNet, T: torch.Tensor, k: torch.Tensor) -> dict[str, torch.Tensor]:
    """Squared violations of calendar, butterfly (Gatheral's g >= 0) and static price bounds."""
    w = net(T, k)
    w_T, w_k = grad(w, T), grad(w, k)
    w_kk = grad(w_k, k)
    wc = w.clamp_min(1e-8)
    g = (1.0 - k * w_k / (2.0 * wc)) ** 2 - 0.25 * w_k**2 * (1.0 / wc + 0.25) + 0.5 * w_kk
    c = bs_call_tv(w, k)
    lower = F.relu(F.relu(1.0 - torch.exp(k)) - c)
    upper = F.relu(c - 1.0)
    return {
        "calendar": F.relu(-w_T),
        "butterfly": F.relu(-g),
        "bounds": lower + upper,
    }


@dataclass
class LossWeights:
    """Heuristic loss weights and scales; tune these for your data."""

    data: float = 1.0
    pde: float = 1.0
    arbitrage: float = 10.0
    iv_scale: float = 0.01  # one vol point
    pde_scale: float = 1e-3
    calendar_scale: float = 1e-2
    butterfly_scale: float = 1e-1
    bounds_scale: float = 1e-3


class RoughHestonPINN(nn.Module):
    """Surface network + learnable Rough Heston parameters + Markovian-projection PDE loss."""

    def __init__(
        self,
        t_range: tuple[float, float],
        k_range: tuple[float, float],
        lam: float = 1.0,
        hidden: int = 64,
        layers: int = 4,
        init_params: dict[str, float] | None = None,
        weights: LossWeights | None = None,
        n_quad: int = 24,
    ) -> None:
        super().__init__()
        self.t_min, self.t_max = t_range
        self.k_min, self.k_max = k_range
        self.weights = weights or LossWeights()
        self.net = SurfaceNet(hidden, layers, t_max=self.t_max, k_scale=max(abs(self.k_min), abs(self.k_max)))
        self.params = RoughHestonParams(lam, init_params)
        self.closure = MarkovianProjection(self.params, n_quad)

    def sigma2(self, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return self.closure.local_variance(T, k)

    def implied_vol(self, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return self.net.implied_vol(T, k)

    def sample_collocation(self, n: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.get_default_dtype()
        T = self.t_min + (self.t_max - self.t_min) * torch.rand(n, device=device, dtype=dtype)
        k = self.k_min + (self.k_max - self.k_min) * torch.rand(n, device=device, dtype=dtype)
        return T.requires_grad_(True), k.requires_grad_(True)

    def loss_terms(
        self, T_d: torch.Tensor, k_d: torch.Tensor, iv_d: torch.Tensor,
        T_c: torch.Tensor, k_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Weighted total loss plus its unweighted components (all scalars)."""
        wt = self.weights
        data = (((self.implied_vol(T_d, k_d) - iv_d) / wt.iv_scale) ** 2).mean()
        pde = ((dupire_residual(self.net, self.sigma2, T_c, k_c) / wt.pde_scale) ** 2).mean()
        pen = arbitrage_penalties(self.net, T_c, k_c)
        cal = ((pen["calendar"] / wt.calendar_scale) ** 2).mean()
        but = ((pen["butterfly"] / wt.butterfly_scale) ** 2).mean()
        bnd = ((pen["bounds"] / wt.bounds_scale) ** 2).mean()
        total = wt.data * data + wt.pde * pde + wt.arbitrage * (cal + but + bnd)
        return {"total": total, "data": data, "pde": pde, "calendar": cal, "butterfly": but, "bounds": bnd}
