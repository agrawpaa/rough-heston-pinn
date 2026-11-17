"""NumPy-only tests for the simulator and Black-Scholes utilities (no PyTorch needed)."""
from math import gamma

import numpy as np

from rough_heston_pinn.sde import (
    RHParams, bs_call_total_variance, implied_total_variance, mc_option_prices, simulate_paths,
)


def mittag_leffler_ref(alpha: float, z: float, n_terms: int = 80) -> float:
    return sum(z**j / gamma(alpha * j + 1.0) for j in range(n_terms))


def test_implied_variance_roundtrip():
    for w, k in [(0.04, 0.0), (0.09, -0.3), (0.02, 0.2)]:
        c = float(bs_call_total_variance(w, k))
        assert abs(implied_total_variance(c, k) - w) < 1e-9


def test_flat_vol_limit_matches_black_scholes():
    p = RHParams(H=0.2, nu=0.0, theta=0.04, V0=0.04, rho=-0.7)
    _, S, V = simulate_paths(p, 1.0, 64, 20_000, seed=1)
    assert np.allclose(V, 0.04)
    prices = mc_option_prices(S, 1.0 / 64, np.array([1.0]), np.array([-0.2, 0.0, 0.2]))[0]
    for c, k in zip(prices, [-0.2, 0.0, 0.2]):
        assert abs(np.sqrt(implied_total_variance(c, k)) - 0.2) < 0.006


def test_spot_is_a_martingale():
    p = RHParams(H=0.1, nu=0.2, theta=0.04, V0=0.04, rho=-0.7)
    _, S, _ = simulate_paths(p, 1.0, 64, 20_000, seed=2)
    assert abs(S[:, -1].mean() - 1.0) < 0.01


def test_mean_variance_matches_mittag_leffler_without_vol_of_vol():
    # With nu = 0 the scheme is deterministic and must reproduce E[V_t] = theta + (V0-theta) E_alpha(-lam t^alpha).
    p = RHParams(H=0.1, nu=0.0, theta=0.03, V0=0.06, rho=-0.7, lam=1.0)
    a = p.H + 0.5
    times, _, V = simulate_paths(p, 2.0, 128, 200, seed=3)
    for T in (0.5, 1.0, 2.0):
        i = int(round(T / (2.0 / 128)))
        ref = p.theta + (p.V0 - p.theta) * mittag_leffler_ref(a, -p.lam * T**a)
        assert abs(V[:, i].mean() - ref) < 2e-4


def test_mean_variance_with_small_vol_of_vol():
    p = RHParams(H=0.1, nu=0.1, theta=0.03, V0=0.06, rho=-0.7, lam=1.0)
    a = p.H + 0.5
    _, _, V = simulate_paths(p, 2.0, 128, 20_000, seed=4)
    for T in (0.5, 2.0):
        i = int(round(T / (2.0 / 128)))
        ref = p.theta + (p.V0 - p.theta) * mittag_leffler_ref(a, -p.lam * T**a)
        assert abs(V[:, i].mean() / ref - 1.0) < 0.02


def test_short_dated_skew_is_steeper_for_rougher_paths():
    skews = {}
    for H in (0.1, 0.4):
        p = RHParams(H=H, nu=0.5, theta=0.04, V0=0.04, rho=-0.7, lam=1.0)
        _, S, _ = simulate_paths(p, 0.25, 64, 20_000, seed=3)
        c = mc_option_prices(S, 0.25 / 64, np.array([0.25]), np.array([-0.05, 0.05]))[0]
        iv = [np.sqrt(implied_total_variance(x, k) / 0.25) for x, k in zip(c, (-0.05, 0.05))]
        skews[H] = (iv[1] - iv[0]) / 0.1
    assert skews[0.1] < skews[0.4] < 0.0
