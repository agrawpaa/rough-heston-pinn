"""Validation suite: implied-vol error, parameter recovery and static-arbitrage diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pinn import PARAM_NAMES, RoughHestonPINN, arbitrage_penalties
from .sde import OptionQuotes, RHParams


@torch.no_grad()
def iv_errors(model: RoughHestonPINN, quotes: OptionQuotes, device: torch.device | str = "cpu") -> dict[str, float]:
    """Implied-vol RMSE / MAE / max error in volatility points (1 point = 0.01)."""
    dt = torch.get_default_dtype()
    T = torch.as_tensor(quotes.T, dtype=dt, device=device)
    k = torch.as_tensor(quotes.k, dtype=dt, device=device)
    err = (model.implied_vol(T, k).cpu().numpy() - quotes.iv) * 100.0
    return {
        "rmse_volpts": float(np.sqrt(np.mean(err**2))),
        "mae_volpts": float(np.mean(np.abs(err))),
        "max_abs_volpts": float(np.max(np.abs(err))),
        "n": int(len(err)),
    }


def parameter_recovery(model: RoughHestonPINN, truth: RHParams) -> dict[str, dict[str, float]]:
    """Estimated vs ground-truth parameters with absolute and relative errors."""
    est, true = model.params.as_floats(), truth.as_dict()
    return {
        n: {"true": true[n], "estimated": est[n], "abs_err": abs(est[n] - true[n]),
            "rel_err": abs(est[n] - true[n]) / abs(true[n])}
        for n in PARAM_NAMES
    }


def arbitrage_report(
    model: RoughHestonPINN, n_T: int = 40, n_k: int = 60, tol: float = 1e-6,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Fraction of a dense (T, k) grid violating each static no-arbitrage condition, and the worst violation."""
    dt = torch.get_default_dtype()
    Tg, kg = torch.meshgrid(
        torch.linspace(model.t_min, model.t_max, n_T, dtype=dt, device=device),
        torch.linspace(model.k_min, model.k_max, n_k, dtype=dt, device=device),
        indexing="ij",
    )
    T = Tg.reshape(-1).clone().requires_grad_(True)
    k = kg.reshape(-1).clone().requires_grad_(True)
    with torch.enable_grad():
        pen = arbitrage_penalties(model.net, T, k)
    out: dict[str, float] = {}
    for name, v in pen.items():
        v = v.detach()
        out[f"{name}_violation_fraction"] = float((v > tol).double().mean())
        out[f"{name}_worst"] = float(v.max())
    return out


def full_report(
    model: RoughHestonPINN, train: OptionQuotes, test: OptionQuotes | None = None,
    truth: RHParams | None = None, device: torch.device | str = "cpu",
) -> dict:
    report: dict = {"fit_train": iv_errors(model, train, device)}
    if test is not None and len(test):
        report["fit_heldout"] = iv_errors(model, test, device)
    if truth is not None:
        report["parameter_recovery"] = parameter_recovery(model, truth)
    else:
        report["parameters"] = model.params.as_floats()
    report["arbitrage"] = arbitrage_report(model, device=device)
    return report


def format_report(report: dict) -> str:
    lines = []
    for key in ("fit_train", "fit_heldout"):
        if key in report:
            r = report[key]
            lines.append(f"{key:12s} RMSE {r['rmse_volpts']:.3f} vol pts | MAE {r['mae_volpts']:.3f} | "
                         f"max {r['max_abs_volpts']:.3f} | n={r['n']}")
    if "parameter_recovery" in report:
        lines.append(f"{'param':7s}{'true':>9s}{'estimated':>11s}{'rel.err':>9s}")
        for n, r in report["parameter_recovery"].items():
            lines.append(f"{n:7s}{r['true']:9.4f}{r['estimated']:11.4f}{100 * r['rel_err']:8.1f}%")
    elif "parameters" in report:
        lines.append("calibrated: " + ", ".join(f"{n}={v:.4f}" for n, v in report["parameters"].items()))
    a = report["arbitrage"]
    lines.append("arbitrage violations on dense grid: " + ", ".join(
        f"{n}={100 * a[f'{n}_violation_fraction']:.1f}%" for n in ("calendar", "butterfly", "bounds")))
    return "\n".join(lines)


def plot_report(
    model: RoughHestonPINN, quotes: OptionQuotes, history_rows: list[dict], out_dir: str | Path,
    device: torch.device | str = "cpu",
) -> list[Path]:
    """Save smile fits (one panel per maturity) and the training curves as PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dt = torch.get_default_dtype()
    paths: list[Path] = []

    maturities = np.unique(np.round(quotes.T, 6))
    fig, axes = plt.subplots(1, len(maturities), figsize=(3.2 * len(maturities), 3.2), sharey=True, squeeze=False)
    kk = np.linspace(model.k_min, model.k_max, 100)
    for ax, T in zip(axes[0], maturities):
        m = np.isclose(quotes.T, T)
        with torch.no_grad():
            Tt = torch.full((len(kk),), float(T), dtype=dt, device=device)
            fit = model.implied_vol(Tt, torch.as_tensor(kk, dtype=dt, device=device)).cpu().numpy()
        ax.scatter(quotes.k[m], quotes.iv[m], s=14, label="quotes")
        ax.plot(kk, fit, label="PINN surface")
        ax.set_title(f"T = {T:.2f}")
        ax.set_xlabel("log-moneyness k")
    axes[0][0].set_ylabel("implied vol")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    p = out / "smile_fit.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for key in ("total", "data", "pde"):
        ax.semilogy([r[key] for r in history_rows], label=key)
    ax.set_xlabel("logged step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out / "loss_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths
