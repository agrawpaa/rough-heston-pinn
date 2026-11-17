"""Two-stage training: Adam with resampled collocation points, then L-BFGS on a fixed set."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .pinn import RoughHestonPINN
from .sde import OptionQuotes


@dataclass
class TrainConfig:
    adam_epochs: int = 3000
    adam_lr: float = 2e-3
    param_lr: float = 1e-2  # larger step for the five physical parameters
    lr_decay: float = 0.1  # final / initial Adam learning rate
    n_collocation: int = 2048
    grad_clip: float = 10.0
    lbfgs_outer: int = 10
    lbfgs_inner: int = 20  # L-BFGS iterations per outer step
    n_collocation_lbfgs: int = 4096
    log_every: int = 100
    seed: int = 0


@dataclass
class History:
    rows: list[dict] = field(default_factory=list)

    def log(self, stage: str, step: int, terms: dict[str, torch.Tensor], params: dict[str, float]) -> None:
        row = {"stage": stage, "step": step, **{k: float(v.detach()) for k, v in terms.items()}}
        row.update({f"param_{k}": v for k, v in params.items()})
        self.rows.append(row)


class TwoStageTrainer:
    """Calibrate network weights and Rough Heston parameters jointly.

    Stage 1 (Adam) resamples collocation points every step, which regularises the PDE residual.
    Stage 2 (L-BFGS, strong-Wolfe line search) polishes on one fixed, larger collocation set so
    that the objective is deterministic.
    """

    def __init__(
        self, model: RoughHestonPINN, quotes: OptionQuotes, config: TrainConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = config or TrainConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        dt = torch.get_default_dtype()
        self.T_d = torch.as_tensor(quotes.T, dtype=dt, device=self.device)
        self.k_d = torch.as_tensor(quotes.k, dtype=dt, device=self.device)
        self.iv_d = torch.as_tensor(quotes.iv, dtype=dt, device=self.device)
        self.history = History()

    def _terms(self, T_c: torch.Tensor, k_c: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model.loss_terms(self.T_d, self.k_d, self.iv_d, T_c, k_c)

    def fit(self, verbose: bool = True) -> History:
        torch.manual_seed(self.cfg.seed)
        self._adam(verbose)
        self._lbfgs(verbose)
        return self.history

    def _adam(self, verbose: bool) -> None:
        c = self.cfg
        opt = torch.optim.Adam(
            [
                {"params": self.model.net.parameters(), "lr": c.adam_lr},
                {"params": self.model.params.parameters(), "lr": c.param_lr},
            ]
        )
        gamma = c.lr_decay ** (1.0 / max(c.adam_epochs, 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        start = time.time()
        for step in range(c.adam_epochs):
            T_c, k_c = self.model.sample_collocation(c.n_collocation, self.device)
            opt.zero_grad(set_to_none=True)
            terms = self._terms(T_c, k_c)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            opt.step()
            sched.step()
            if step % c.log_every == 0 or step == c.adam_epochs - 1:
                self.history.log("adam", step, terms, self.model.params.as_floats())
                if verbose:
                    print(f"[adam {step:5d}] loss={float(terms['total']):.4g} data={float(terms['data']):.4g} "
                          f"pde={float(terms['pde']):.4g} ({time.time() - start:.0f}s)")

    def _lbfgs(self, verbose: bool) -> None:
        c = self.cfg
        if c.lbfgs_outer <= 0:
            return
        T_c, k_c = self.model.sample_collocation(c.n_collocation_lbfgs, self.device)
        opt = torch.optim.LBFGS(
            self.model.parameters(), lr=1.0, max_iter=c.lbfgs_inner, history_size=50,
            tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            total = self._terms(T_c, k_c)["total"]
            total.backward()
            return total

        for outer in range(c.lbfgs_outer):
            opt.step(closure)
            terms = self._terms(T_c, k_c)
            self.history.log("lbfgs", outer, terms, self.model.params.as_floats())
            if verbose:
                print(f"[lbfgs {outer:3d}] loss={float(terms['total']):.4g} data={float(terms['data']):.4g} "
                      f"pde={float(terms['pde']):.4g}")
