# Experiment 3 — Warm Start for MLMC Simulation

Reproduces the figure `sec3_3_warmstart_3panel.pdf`: MLMC estimates of
`E[Z_k^{(0.2)}]` for `k = 1, 2, 3` versus computational complexity, under three
initial distributions, showing that the multi-scaling product-form
approximation is an effective warm start.

## SRBM setting

Three-dimensional SRBM with

```
Gamma = I_3,
R     = [ 1.0 -0.6 -0.4;
         -0.5  1.0 -0.4;
         -0.2 -0.3  1.0],
delta = (r, r^2, r^3),   r = 0.2,   mu = -R*delta.
```

Three initial distributions (independent exponentials with the given means):

| name            | initial mean vector      |
|-----------------|--------------------------|
| `origin`        | `(0, 0, 0)`              |
| `skew`          | `(2.5, 12.5, 62.5)`      |
| `multi_scaling` | `(2.5, 22.32, 179.69)`   |

## Files

| file | role |
|---|---|
| `generate_data.m`  | Data generation. MLMC sweep over `init x L x T` at `gamma = 0.2`. Writes `results/data/config_<NN>.mat` (array mode) or `all_results.mat` (single-job mode). |
| `aggregate.m`      | Combines the per-task `config_*.mat` files into `results/data/all_results.mat`. |
| `make_figure.m`    | Plotting. Builds the 1x3 paper figure from `all_results.mat`. |
| `results/data/config_01..24.mat` | Pre-computed MLMC data (50 estimators x 2000 paths per config). |

The shared MLMC engine `MLMC_RBM_Trial_Mem.m` and the Skorokhod map
`Skorokhod_linear.m` live in `../lib/`.

## MLMC hyperparameter grid

`generate_data.m` runs the flat grid `gamma = 0.2`, `T in {5000, 100000}`,
`L in {2, 4, 6, 8}`, `init in {origin, skew, multi_scaling}` = 24 configs. The
flat index is the `SLURM_ARRAY_TASK_ID`:

- `config_01..12` -> `T = 5000`  (the warm-start figure uses these)
- `config_13..24` -> `T = 100000` (the longer-horizon reference run; `L = 8`
  gives the reference means `(2.75, 24.86, 166.06)`)

At level `l`, each sample path is simulated over horizon `l*T` with step size
`gamma^l`; complexity is the total number of discretization steps. Each config
uses `Nest = 50` independent estimators, each from `Nepsilon = 2000` paths.

## Reproduce

The pre-computed data is bundled, so you can regenerate the figure directly:

```matlab
cd exp3_warmstart
aggregate     % config_*.mat  ->  all_results.mat
make_figure   % all_results.mat -> sec3_3_warmstart_3panel.pdf
```

To regenerate the data from scratch (heavy — tens of CPU-hours; uses the
Parallel Computing Toolbox). Run with MATLAB directly, no scheduler required:

```bash
cd exp3_warmstart
matlab -batch generate_data     # all 24 configs sequentially -> results/data/all_results.mat
matlab -batch make_figure       # -> results/data/sec3_3_warmstart_3panel.pdf
```

On a cluster you can parallelise across configs: set `SLURM_ARRAY_TASK_ID`
(1..24) so each task runs one config into `results/data/config_<NN>.mat`, then
combine them with `aggregate` before `make_figure`.

A quick pipeline check (tiny grid, finishes in minutes) is available via the
smoke flag: `SEC33_SMOKE=1 matlab -batch "cd exp3_warmstart; generate_data"`.

## Note on the plotting script

`make_figure.m` is a clean reimplementation of the figure script
used for the paper (the original was an ad-hoc script that was not kept under
version control). It reads the same `all_results.mat` produced by
`generate_data.m` / `aggregate.m` and reproduces the published 1x3
layout. It has not been re-executed in this packaging; run it as above to
regenerate the PDF.
