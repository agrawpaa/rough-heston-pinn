"""Command-line interface.

Synthetic validation (simulate with known parameters, calibrate, report recovery)::

    python main.py synthetic --H 0.10 --nu 0.20 --theta 0.04 --rho -0.7 --v0 0.04

Calibrate to your own quotes (CSV with columns T,k,iv; k = ln(K/S) with zero rates assumed)::

    python main.py calibrate --csv quotes.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rough_heston_pinn.optimizer import TrainConfig, TwoStageTrainer
from rough_heston_pinn.pinn import RoughHestonPINN
from rough_heston_pinn.sde import OptionQuotes, RHParams, generate_synthetic_surface
from rough_heston_pinn.validation import format_report, full_report, plot_report


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rough Heston calibration with a physics-informed neural network")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--device", default="auto", help="cpu, cuda, mps or auto")
        sp.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"],
                        help="auto = float64 on CPU, float32 on GPU")
        sp.add_argument("--lam", type=float, default=1.0, help="fixed mean-reversion speed")
        sp.add_argument("--adam-epochs", type=int, default=3000)
        sp.add_argument("--lbfgs-outer", type=int, default=10)
        sp.add_argument("--collocation", type=int, default=2048)
        sp.add_argument("--hidden", type=int, default=64)
        sp.add_argument("--layers", type=int, default=4)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out-dir", default="results")
        sp.add_argument("--no-plots", action="store_true")

    s = sub.add_parser("synthetic", help="simulate ground-truth data, calibrate, report recovery")
    for name, default in (("H", 0.10), ("nu", 0.20), ("theta", 0.04), ("rho", -0.70), ("v0", 0.04)):
        s.add_argument(f"--{name}", type=float, default=default)
    s.add_argument("--paths", type=int, default=20_000)
    s.add_argument("--steps-per-year", type=int, default=128)
    s.add_argument("--maturities", type=_floats, default=_floats("0.1,0.25,0.5,1.0,1.5,2.0"))
    s.add_argument("--log-moneyness", type=_floats, default=list(np.linspace(-0.4, 0.3, 15)))
    s.add_argument("--iv-noise", type=float, default=0.0, help="absolute std of Gaussian noise added to IVs")
    s.add_argument("--test-fraction", type=float, default=0.2)
    common(s)

    c = sub.add_parser("calibrate", help="calibrate to a CSV of quotes (columns T,k,iv)")
    c.add_argument("--csv", required=True)
    c.add_argument("--test-fraction", type=float, default=0.0)
    common(c)
    return p


def setup_device(device: str, dtype: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dtype == "auto":
        dtype = "float64" if dev.type == "cpu" else "float32"
    torch.set_default_dtype(getattr(torch, dtype))
    return dev


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    dev = setup_device(args.device, args.dtype)
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    truth: RHParams | None = None
    if args.mode == "synthetic":
        truth = RHParams(H=args.H, nu=args.nu, theta=args.theta, rho=args.rho, V0=args.v0, lam=args.lam)
        print(f"Simulating {args.paths} paths with {truth} ...")
        quotes = generate_synthetic_surface(
            truth, np.array(args.maturities), np.array(args.log_moneyness),
            n_paths=args.paths, steps_per_year=args.steps_per_year, seed=args.seed, iv_noise=args.iv_noise,
        )
        quotes.to_csv(out / "synthetic_quotes.csv")
        print(f"Generated {len(quotes)} implied-vol quotes -> {out / 'synthetic_quotes.csv'}")
    else:
        quotes = OptionQuotes.from_csv(args.csv)

    train, test = (quotes.split(args.test_fraction, args.seed) if args.test_fraction > 0 else (quotes, None))
    model = RoughHestonPINN(
        t_range=(max(0.5 * float(quotes.T.min()), 0.02), float(quotes.T.max())),
        k_range=(float(quotes.k.min()), float(quotes.k.max())),
        lam=args.lam, hidden=args.hidden, layers=args.layers,
    )
    cfg = TrainConfig(adam_epochs=args.adam_epochs, lbfgs_outer=args.lbfgs_outer,
                      n_collocation=args.collocation, seed=args.seed)
    trainer = TwoStageTrainer(model, train, cfg, dev)
    history = trainer.fit()

    report = full_report(model, train, test, truth, dev)
    print("\n" + format_report(report))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "history.json").write_text(json.dumps(history.rows, indent=2))
    if not args.no_plots:
        for p in plot_report(model, quotes, history.rows, out, dev):
            print(f"saved {p}")
    torch.save(model.state_dict(), out / "model.pt")
    return report


if __name__ == "__main__":
    main()
