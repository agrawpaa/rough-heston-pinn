"""PyTorch tests (skipped automatically when torch is not installed)."""
from math import gamma

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rough_heston_pinn.optimizer import TrainConfig, TwoStageTrainer  # noqa: E402
from rough_heston_pinn.pinn import (  # noqa: E402
    MarkovianProjection, RoughHestonPINN, RoughHestonParams, arbitrage_penalties, dupire_residual,
    mittag_leffler,
)
from rough_heston_pinn.sde import OptionQuotes, bs_call_total_variance  # noqa: E402
from rough_heston_pinn.surface import SurfaceNet, bs_call_tv  # noqa: E402

torch.set_default_dtype(torch.float64)


def ml_ref(alpha, z, n=80):
    return sum(z**j / gamma(alpha * j + 1.0) for j in range(n))


def test_mittag_leffler_matches_series_and_exponential():
    for a, z in [(0.6, -1.3), (0.8, -0.4), (0.55, -2.0)]:
        got = float(mittag_leffler(torch.tensor(a), torch.tensor([z]))[0])
        assert abs(got - ml_ref(a, z)) < 1e-10
    assert abs(float(mittag_leffler(torch.tensor(1.0), torch.tensor([-0.7]))[0]) - np.exp(-0.7)) < 1e-10


def test_black_scholes_matches_numpy():
    w, k = np.array([0.04, 0.09]), np.array([0.0, -0.3])
    got = bs_call_tv(torch.tensor(w), torch.tensor(k)).numpy()
    assert np.allclose(got, bs_call_total_variance(w, k), atol=1e-12)


def test_mean_variance_is_flat_when_theta_equals_v0():
    p = RoughHestonParams(lam=1.0, init={"theta": 0.05, "V0": 0.05})
    vb = MarkovianProjection(p).mean_variance(torch.tensor([0.1, 0.5, 1.5]))
    assert torch.allclose(vb, torch.full_like(vb, 0.05), atol=1e-4)


def test_local_variance_is_positive_and_skewed_for_negative_rho():
    p = RoughHestonParams(lam=1.0, init={"rho": -0.7, "nu": 0.3})
    cl = MarkovianProjection(p)
    T = torch.full((3,), 0.5)
    s2 = cl.local_variance(T, torch.tensor([-0.2, 0.0, 0.2]))
    assert torch.all(s2 > 0) and s2[0] > s2[1] > s2[2]


def test_dupire_residual_is_zero_for_black_scholes_with_matching_local_variance():
    class FlatNet(torch.nn.Module):
        def forward(self, T, k):
            return 0.04 * T

    T = torch.rand(64) * 1.5 + 0.1
    k = torch.rand(64) * 0.6 - 0.3
    T.requires_grad_(True)
    k.requires_grad_(True)
    res = dupire_residual(FlatNet(), lambda T_, k_: torch.full_like(T_, 0.04), T, k)
    assert float(res.abs().max()) < 1e-8


def test_arbitrage_penalties_are_finite_and_nonnegative():
    net = SurfaceNet(hidden=16, layers=2)
    T = (torch.rand(32) + 0.1).requires_grad_(True)
    k = (torch.rand(32) - 0.5).requires_grad_(True)
    for v in arbitrage_penalties(net, T, k).values():
        assert torch.isfinite(v).all() and (v >= 0).all()


def test_gradients_reach_physical_parameters_and_network():
    model = RoughHestonPINN((0.05, 1.0), (-0.3, 0.3), hidden=16, layers=2)
    T_d, k_d, iv_d = torch.tensor([0.25, 0.5]), torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.21])
    T_c, k_c = model.sample_collocation(64)
    model.loss_terms(T_d, k_d, iv_d, T_c, k_c)["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.params.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net.parameters())


def test_training_smoke_run_reduces_data_loss():
    T = np.repeat([0.25, 0.5], 5)
    k = np.tile(np.linspace(-0.2, 0.2, 5), 2)
    quotes = OptionQuotes(T, k, 0.2 - 0.1 * k)
    model = RoughHestonPINN((0.1, 0.5), (-0.2, 0.2), hidden=16, layers=2)
    cfg = TrainConfig(adam_epochs=60, lbfgs_outer=1, lbfgs_inner=5, n_collocation=128,
                      n_collocation_lbfgs=128, log_every=20)
    hist = TwoStageTrainer(model, quotes, cfg).fit(verbose=False)
    assert hist.rows[-1]["data"] < hist.rows[0]["data"]
