# rough-heston-pinn

![CI](https://github.com/agrawpaa/rough-heston-pinn/actions/workflows/ci.yml/badge.svg)

Calibrate the **Rough Heston** stochastic-volatility model to option surfaces with a **physics-informed neural network (PINN)**. The network represents the implied-volatility surface, a Markovian-projection Dupire PDE ties it to the model parameters (H, ν, θ, ρ, V0), and no-arbitrage penalties keep the fitted surface admissible. A hybrid-scheme Monte Carlo simulator generates synthetic option chains with known ground truth, so you can test whether calibration recovers the parameters.

## How it works

**Model** (zero rates, S0 = 1, α = H + ½, kernel K(u) = u^(α−1)/Γ(α)):

```
V_t = V0 + ∫ K(t−s) λ(θ − V_s) ds + ν ∫ K(t−s) √V_s dW_s
dS_t / S_t = √V_t ( ρ dW_t + √(1−ρ²) dW⊥_t )
```

`λ` (mean reversion) is fixed, not calibrated.

**Surface network** (`surface.py`). A tanh MLP gives total variance `w(T,k) = T · softplus(f(√T, k))`, which is positive and vanishes at T = 0 by construction. Call prices come from Black-Scholes applied to `w`, and all derivatives use `torch.autograd`.

**Physics residual** (`pinn.py`). Call prices obey Dupire's forward equation in log-moneyness:

```
∂c/∂T = ½ σ²_loc(T,k) ( ∂²c/∂k² − ∂c/∂k )
```

The rough variance is not Markovian, so `σ²_loc(T,k) = E[V_T | ln S_T = k]` comes from a **Markovian projection** with a Gaussian (linear) closure:

```
σ²_loc ≈ V̄(T) + β(T) ( k + I(T)/2 )
V̄(T)  = θ + (V0 − θ) E_α(−λ T^α)           exact mean of V_T (Mittag-Leffler function)
I(T)   = ∫₀ᵀ V̄(s) ds
β(T)   = ρ ν ( ∫₀ᵀ K(T−s) V̄(s) ds ) / I(T)   first order in ν
```

This is why H matters: at short maturities the skew scales like T^(H−½), so lower H means a steeper short-dated skew. The singular convolution is integrated with the substitution T−s = T·z^(1/α) and Gauss-Legendre quadrature.

**No-arbitrage penalties.** Calendar (∂w/∂T ≥ 0), butterfly (Gatheral's g(k) ≥ 0) and static bounds on call prices, evaluated on collocation points.

**Training** (`optimizer.py`). Stage 1 is Adam with collocation points resampled each step and a higher learning rate on the five physical parameters. Stage 2 is L-BFGS (strong-Wolfe) on one fixed, larger collocation set.

**Simulator** (`sde.py`). A hybrid-scheme-style Volterra Euler scheme: the singular kernel is integrated exactly against the nearest Brownian cell (exact joint Gaussian law) and evaluated at Bennedsen–Lunde–Pakkanen's optimal points for older cells. It uses antithetic variates, full truncation of V at zero, and a martingale correction on terminal spot.

**Validation** (`validation.py`). Implied-vol RMSE/MAE (in vol points) on training and held-out quotes, parameter recovery with relative errors, and the fraction of a dense (T, k) grid violating each no-arbitrage condition.

## Install

```bash
git clone https://github.com/agrawpaa/rough-heston-pinn.git
cd rough-heston-pinn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Usage

Synthetic validation (simulate with known parameters, calibrate, report recovery):

```bash
python main.py synthetic --H 0.10 --nu 0.20 --theta 0.04 --rho -0.7 --v0 0.04
```

Calibrate to your own quotes (CSV with columns `T,k,iv`, where `k = ln(K/S)` and rates are assumed zero):

```bash
python main.py calibrate --csv quotes.csv
```

Useful flags: `--device cuda`, `--adam-epochs 5000`, `--paths 50000`, `--iv-noise 0.002`, `--lam 1.0`, `--no-plots`. Outputs go to `results/`: `report.json`, `history.json`, `synthetic_quotes.csv`, `smile_fit.png`, `loss_curves.png`, `model.pt`.

## Layout

```
main.py                    CLI (synthetic | calibrate)
rough_heston_pinn/
  sde.py                   hybrid-scheme Monte Carlo, Black-Scholes inversion, synthetic surfaces (NumPy)
  surface.py               total-variance network, torch Black-Scholes, autograd helper
  pinn.py                  parameters, Mittag-Leffler, Markovian projection, Dupire residual, penalties, loss
  optimizer.py             two-stage Adam then L-BFGS trainer
  validation.py            IV error, parameter recovery, arbitrage diagnostics, plots
tests/                     simulator tests (NumPy) and PINN tests (PyTorch)
```

## Results

Add your own run here after you execute the synthetic experiment, for example the parameter-recovery table printed by `main.py` and the `smile_fit.png` figure from `results/`. Report the seed, number of paths, and noise level next to the numbers.

## Limitations

- **Closure accuracy.** The Gaussian projection is first order in ν. It captures the level and the H-dependent skew but not smile curvature from vol-of-vol, and σ²_loc can turn negative far out of the money at short maturities, where a smooth positive floor is applied. Calibrated parameters on strongly curved surfaces will be biased toward whatever matches the skew.
- **Identifiability.** A single snapshot of the surface constrains H, ν, ρ and V0/θ only weakly in some regimes. Expect correlated errors, and check several seeds.
- **Simulator bias.** Full truncation of V at zero pulls the simulated mean of V below the exact Mittag-Leffler mean as ν grows relative to √V0 (many negative variance draws). In my checks the scheme matched the exact mean when ν = 0 and to within about 0.4% at ν = 0.1, but was roughly 3% to 5% low at ν = 0.2 (H = 0.1, V0 = 0.06, θ = 0.03). Keep ν moderate for synthetic tests, or increase the number of steps.
- **Loss weights.** The default scales and weights in `LossWeights` are heuristics, not tuned values.
- **Scope.** Zero rates and dividends, a single underlying, and European options only. Real data needs forward and rate adjustment before conversion to `(T, k, iv)`.

## Testing

`tests/test_sde.py` checks the simulator against Black-Scholes in the flat-volatility limit, the martingale property, the exact Mittag-Leffler mean variance, and the short-dated skew ordering in H. `tests/test_pinn.py` checks the torch Mittag-Leffler function, the Black-Scholes implementation, the Dupire residual (zero for a Black-Scholes surface with matching local variance), penalty signs, gradient flow to the physical parameters, and a short training smoke run.

## References

- El Euch and Rosenbaum (2019), *The characteristic function of rough Heston models*.
- Bennedsen, Lunde and Pakkanen (2017), *Hybrid scheme for Brownian semistationary processes*.
- Dupire (1994), *Pricing with a smile*. Gyöngy (1986), *Mimicking the one-dimensional marginal distributions of processes having an Itô differential*.
- Gatheral and Jacquier (2014), *Arbitrage-free SVI volatility surfaces*.
- Raissi, Perdikaris and Karniadakis (2019), *Physics-informed neural networks*.

## License

MIT
