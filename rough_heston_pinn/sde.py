"""Rough Heston Monte Carlo and synthetic option surfaces (NumPy only).

Model (zero rates, S0 = 1), with alpha = H + 1/2 and kernel K(u) = u^(alpha-1) / Gamma(alpha)::

    V_t = V0 + int_0^t K(t-s) lam (theta - V_s) ds + nu int_0^t K(t-s) sqrt(V_s) dW_s
    dS_t / S_t = sqrt(V_t) (rho dW_t + sqrt(1 - rho^2) dW_perp_t)

The simulator is a hybrid-scheme-style Volterra Euler scheme (Bennedsen, Lunde and
Pakkanen, 2017): the singular kernel is integrated exactly against the nearest
Brownian cell, using the exact joint Gaussian law of (dW, int K dW), and is
evaluated at the optimal point b*_k for the remaining cells. The drift kernel is
integrated exactly cell by cell. Variance is truncated at zero where it enters
square roots and drifts (full truncation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, ndtr


@dataclass(frozen=True)
class RHParams:
    """Ground-truth Rough Heston parameters (floats). ``lam`` is held fixed in calibration."""

    H: float = 0.10
    nu: float = 0.20
    theta: float = 0.04
    rho: float = -0.70
    V0: float = 0.04
    lam: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.H <= 0.5:
            raise ValueError("H must lie in (0, 0.5]")
        if self.nu < 0 or self.theta <= 0 or self.V0 <= 0 or self.lam < 0:
            raise ValueError("nu and lam must be >= 0, theta and V0 must be > 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class OptionQuotes:
    """Scattered implied-vol quotes: maturity T, log-moneyness k = ln(K/S0), implied vol."""

    T: np.ndarray
    k: np.ndarray
    iv: np.ndarray

    def __len__(self) -> int:
        return int(self.T.shape[0])

    def to_csv(self, path: str | Path) -> None:
        arr = np.column_stack([self.T, self.k, self.iv])
        np.savetxt(path, arr, delimiter=",", header="T,k,iv", comments="")

    @classmethod
    def from_csv(cls, path: str | Path) -> "OptionQuotes":
        arr = np.genfromtxt(path, delimiter=",", names=True)
        return cls(np.atleast_1d(arr["T"]), np.atleast_1d(arr["k"]), np.atleast_1d(arr["iv"]))

    def split(self, test_fraction: float, seed: int = 0) -> tuple["OptionQuotes", "OptionQuotes"]:
        """Random train/test split; the test part is held out of calibration."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_test = int(round(test_fraction * len(self)))
        te, tr = idx[:n_test], idx[n_test:]

        def pick(i: np.ndarray) -> "OptionQuotes":
            return OptionQuotes(self.T[i], self.k[i], self.iv[i])

        return pick(tr), pick(te)


# --------------------------------------------------------------------------- Black-Scholes


def bs_call_total_variance(w: np.ndarray | float, k: np.ndarray | float) -> np.ndarray:
    """Normalised Black-Scholes call price C/S0 as a function of total variance w = sigma^2 T."""
    w = np.maximum(np.asarray(w, dtype=float), 1e-14)
    k = np.asarray(k, dtype=float)
    sw = np.sqrt(w)
    d1 = (-k + 0.5 * w) / sw
    return ndtr(d1) - np.exp(k) * ndtr(d1 - sw)


def implied_total_variance(price: float, k: float, w_hi: float = 4.0) -> float:
    """Invert the Black-Scholes call price for total variance; NaN if there is no valid solution."""
    intrinsic = max(1.0 - float(np.exp(k)), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-9 or price >= 1.0 - 1e-9:
        return float("nan")

    def f(w: float) -> float:
        return float(bs_call_total_variance(w, k)) - price

    if f(1e-10) > 0.0 or f(w_hi) < 0.0:
        return float("nan")
    return float(brentq(f, 1e-10, w_hi, xtol=1e-13))


# --------------------------------------------------------------------------- simulation


def simulate_paths(
    p: RHParams, t_max: float, n_steps: int, n_paths: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate (S, V) on a uniform grid. Returns ``times``, ``S`` and ``V`` (paths x (n_steps + 1)).

    Antithetic variates are used, so ``n_paths`` must be even. Cost is O(n_paths * n_steps^2).
    """
    if n_paths % 2:
        raise ValueError("n_paths must be even (antithetic variates)")
    rng = np.random.default_rng(seed)
    a = p.H + 0.5
    dt = t_max / n_steps
    g = gamma(a)

    k = np.arange(1, n_steps + 1, dtype=float)
    cell = k**a - (k - 1.0) ** a
    w_drift = dt**a / gamma(a + 1.0) * cell  # exact integral of K over cell k
    if abs(a - 1.0) < 1e-9:
        w_stoch = np.ones(n_steps) / g
    else:
        b_star = (cell / a) ** (1.0 / (a - 1.0))  # optimal evaluation points (cells k >= 2)
        w_stoch = (b_star * dt) ** (a - 1.0) / g

    # exact joint law of (dW, int_cell K dW) for the nearest cell
    var_w, cov = dt, dt**a / (a * g)
    var_wt = dt ** (2 * a - 1.0) / ((2 * a - 1.0) * g**2)
    l11 = np.sqrt(var_w)
    l21 = cov / l11
    l22 = np.sqrt(max(var_wt - l21**2, 0.0))

    half = n_paths // 2
    z = rng.standard_normal((3, half, n_steps))
    z = np.concatenate([z, -z], axis=1)
    dW = l11 * z[0]
    dWt = l21 * z[0] + l22 * z[1]
    dWp = np.sqrt(dt) * z[2]

    P = n_paths
    V = np.empty((P, n_steps + 1))
    V[:, 0] = p.V0
    drift = np.empty((P, n_steps))
    sdw = np.empty((P, n_steps))
    logS = np.zeros((P, n_steps + 1))
    rho_c = np.sqrt(1.0 - p.rho**2)

    for i in range(n_steps):
        vp = np.maximum(V[:, i], 0.0)
        sv = np.sqrt(vp)
        drift[:, i] = p.lam * (p.theta - vp)
        sdw[:, i] = sv * dW[:, i]
        conv = drift[:, : i + 1] @ w_drift[: i + 1][::-1]
        stoch = sv * dWt[:, i]
        if i > 0:
            stoch = stoch + sdw[:, :i] @ w_stoch[1 : i + 1][::-1]
        V[:, i + 1] = p.V0 + conv + p.nu * stoch
        logS[:, i + 1] = logS[:, i] - 0.5 * vp * dt + sv * (p.rho * dW[:, i] + rho_c * dWp[:, i])

    times = np.linspace(0.0, t_max, n_steps + 1)
    return times, np.exp(logS), V


def mc_option_prices(
    S: np.ndarray, dt: float, maturities: np.ndarray, ks: np.ndarray
) -> np.ndarray:
    """Normalised option prices C/S0 on a (maturity x log-moneyness) grid.

    The terminal sample is rescaled to have mean one (martingale correction). Calls are used for
    k >= 0 and puts converted by parity for k < 0, which reduces Monte Carlo noise in the wings.
    """
    out = np.empty((len(maturities), len(ks)))
    for a, T in enumerate(maturities):
        ST = S[:, int(round(T / dt))]
        ST = ST / ST.mean()
        for b, kk in enumerate(ks):
            K = np.exp(kk)
            if kk >= 0:
                out[a, b] = np.maximum(ST - K, 0.0).mean()
            else:
                out[a, b] = np.maximum(K - ST, 0.0).mean() + 1.0 - K
    return out


def generate_synthetic_surface(
    p: RHParams,
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    n_paths: int = 20_000,
    steps_per_year: int = 128,
    seed: int = 0,
    iv_noise: float = 0.0,
) -> OptionQuotes:
    """Monte Carlo implied-vol quotes from known ground-truth parameters (for validation)."""
    maturities = np.asarray(maturities, dtype=float)
    log_moneyness = np.asarray(log_moneyness, dtype=float)
    t_max = float(maturities.max())
    n_steps = max(64, int(np.ceil(t_max * steps_per_year)))
    _, S, _ = simulate_paths(p, t_max, n_steps, n_paths, seed)
    prices = mc_option_prices(S, t_max / n_steps, maturities, log_moneyness)

    rng = np.random.default_rng(seed + 1)
    Ts, ks, ivs = [], [], []
    for a, T in enumerate(maturities):
        for b, kk in enumerate(log_moneyness):
            w = implied_total_variance(prices[a, b], kk)
            if np.isfinite(w):
                Ts.append(T)
                ks.append(kk)
                ivs.append(np.sqrt(w / T) + iv_noise * rng.standard_normal())
    return OptionQuotes(np.array(Ts), np.array(ks), np.array(ivs))
