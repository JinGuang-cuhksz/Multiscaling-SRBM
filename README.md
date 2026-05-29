# Numerical Experiments — Asymptotic Product-form Steady-state Distribution for SRBM in the Multi-scaling Regime

Code and data to reproduce the numerical experiments in

> **Asymptotic Product-form Steady-state Distribution for Semimartingale
> Reflecting Brownian Motion in Multi-scaling Regime**
> Jin Guang, Xinyun Chen, J. G. Dai, Peter W. Glynn.

The paper shows that, in a multi-scaling regime where the traffic-slackness
vector satisfies `delta = (r, r^2, ..., r^d)`, the scaled stationary
distribution of a semimartingale reflecting Brownian motion (SRBM) converges
weakly to a product of independent exponentials as `r -> 0`. This repository
contains the three numerical experiments of the paper, each producing one
figure.

## Experiments

| folder | paper subsection | figure(s) | what it shows |
|---|---|---|---|
| [`exp1_mean_accuracy/`](exp1_mean_accuracy/) | Mean Approximation Accuracy | `rbm_relative_error_fixed_beta_*_linear.pdf` | Relative error of the multi-scaling vs skew-symmetric mean against the exact Foddy (1984) value, 2D SRBM. |
| [`exp2_distribution/`](exp2_distribution/) | Distribution and Tail Approximation Accuracy | `fig_sec3_3_dist.pdf` | PDF and tail (CCDF) of `S = Z_1 + Z_2 + Z_3` vs an MLMC reference, 3D SRBM (diagonal and non-diagonal covariance). |
| [`exp3_warmstart/`](exp3_warmstart/) | Warm Start for MLMC Simulation | `sec3_3_warmstart_3panel.pdf` | MLMC mean estimates vs complexity under three initial distributions, showing the multi-scaling warm start accelerates convergence, 3D SRBM. |

Each experiment folder has its own `README.md` with the exact SRBM parameters,
file roles, and reproduction commands. The final figures used in the paper are
also collected in [`figures/`](figures/).

## Layout

```
publication/
├── README.md
├── requirements.txt          # Python deps for exp2 (numpy, matplotlib, h5py)
├── lib/                       # shared MATLAB code (see below)
├── exp1_mean_accuracy/        # Experiment 1 (analytic, MATLAB)
├── exp2_distribution/         # Experiment 2 (MLMC data-gen + Python figure)
│   └── results/...            # bundled MLMC .mat data
├── exp3_warmstart/            # Experiment 3 (MLMC sweep + MATLAB figure)
│   └── results/...            # bundled MLMC .mat data
└── figures/                   # final figures as used in the paper
```

### Shared library (`lib/`)

| file | used by | role |
|---|---|---|
| `rbm_moments.m`        | exp1 | Exact stationary moments of a 2D SRBM (Foddy 1984 formulas). |
| `Skorokhod_linear.m`   | exp2, exp3 | Skorokhod reflection map. **Third-party — Blanchet et al. (2021); see [Third-party code](#third-party-code).** |
| `MLMC_RBM_Trial_Mem.m` | exp2, exp3 | Multi-level Monte Carlo simulator. **Third-party — Blanchet et al. (2021); see [Third-party code](#third-party-code).** |
| `hypoexp_ccdf.m`       | exp2 | CCDF/PDF of a sum of independent exponentials (phase-type). |
| `legend_first_lines.m` | exp1 | Plotting helper. |

## Third-party code

The multi-level Monte Carlo simulator — `lib/MLMC_RBM_Trial_Mem.m` and
`lib/Skorokhod_linear.m` — is **not** part of this work. It is the code of
Blanchet et al. (2021), obtained from the authors and redistributed here with
their permission. Copyright remains with the original authors; these two files
are **not** covered by this repository's MIT license (see [`NOTICE`](NOTICE)).

> J. Blanchet, X. Chen, N. Si, P. W. Glynn, *Efficient steady-state simulation
> of high-dimensional stochastic networks*, Stochastic Systems 11 (2021)
> 174–192.

## Quick start (regenerate figures from bundled data)

The MLMC data needed for the figures is included, so the figures regenerate
without re-running the heavy simulations.

```matlab
% Experiment 1 — runs in seconds; edit (alpha, beta) per panel, see its README
cd exp1_mean_accuracy && make_figure

% Experiment 3 — combine bundled config data and plot
cd exp3_warmstart && aggregate && make_figure
```

```bash
# Experiment 2 — Python figure from bundled MLMC data
pip install -r requirements.txt
cd exp2_distribution && python make_figure.py
```

## Regenerating the MLMC data from scratch

The MLMC runs are computationally heavy (the `L = 8`, `T = 1e5` cases take tens
of CPU-hours) and use the Parallel Computing Toolbox (`parfor`), which spins up
a local pool sized to the available cores. Run them directly with MATLAB — no
scheduler required:

```bash
# Experiment 2 — distribution/tail data (two covariance settings)
cd exp2_distribution
WARMSTART=origin SEC33_L=8 SEC33_SIGMA=identity matlab -batch generate_data
WARMSTART=origin SEC33_L=8 SEC33_SIGMA=corr_v1  matlab -batch generate_data

# Experiment 3 — warm-start sweep (all 24 configs, run sequentially)
cd ../exp3_warmstart
matlab -batch generate_data        # -> results/data/all_results.mat
matlab -batch make_figure          # -> results/data/sec3_3_warmstart_3panel.pdf
```

`generate_data` reads optional environment variables to support cluster
parallelism: if `SLURM_ARRAY_TASK_ID` is set, Experiment 3 runs a single config
(write `results/data/config_<NN>.mat`, then combine the per-task files with
`aggregate` before `make_figure`); `SLURM_CPUS_PER_TASK` sizes the parallel
pool. A quick pipeline check uses the smoke flag:
`SEC33_SMOKE=1 matlab -batch "cd exp3_warmstart; generate_data"`.

## Notes

- **Software**: MATLAB (R2025b, with Parallel Computing Toolbox) for
  Experiments 1 and 3 and the data generation of Experiment 2; Python 3 with
  `numpy`, `matplotlib`, `h5py` for the Experiment 2 figure.
- **Reconstructed script**: `exp3_warmstart/make_figure.m` is a clean
  reimplementation of the original (unversioned) plotting script for the
  warm-start figure; it reads the same `all_results.mat` and reproduces the
  published layout. See that folder's README.
