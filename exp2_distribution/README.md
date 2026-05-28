# Experiment 2 — Distribution and Tail Approximation Accuracy

Reproduces `fig_sec3_3_dist.pdf`: the PDF and CCDF (tail) of
`S = Z_1^{(r)} + Z_2^{(r)} + Z_3^{(r)}` for a 3D SRBM, comparing the
multi-scaling approximation and the skew-symmetric approximation against an MLMC
reference.

## SRBM setting

Three-dimensional SRBM with

```
R     = [ 1.0 -0.6 -0.4;
         -0.5  1.0 -0.4;
         -0.2 -0.3  1.0],
delta = (r, r^2, r^3),   r = 0.2,   mu = -R*delta.
```

Two covariance settings are shown:

- `identity`  : `Gamma = I_3`        — panels (a) PDF and (b) CCDF.
- `corr_v1`   : non-diagonal `Gamma` with `Gamma_12 = -0.6`, `Gamma_13 = -0.3`,
  `Gamma_23 = -0.4` — panel (c) CCDF.

Since no closed form is available, the reference is an MLMC estimate.

## Files

| file | role |
|---|---|
| `generate_data.m` | Data generation. MLMC run with a per-path raw-sum CCDF estimator on the grid. Writes `results/data/*.mat`. Configured via environment variables (`WARMSTART`, `SEC33_L`, `SEC33_SIGMA`, ...). |
| `make_figure.py` | Plotting. Reads the two `.mat` files below and writes the combined 1x3 figure `fig_sec3_3_dist.pdf` (plus standalone panels) to `outputs/figures_paper/`. |
| `results/data/origin_gamma_0p2_L_8_T_100000_Nest_50_Neps_2000.mat` | MLMC data, `Gamma = I_3`  (panels a, b). |
| `results/data/origin_sigma_corr_v1_gamma_0p2_L_8_T_100000_Nest_50_Neps_2000.mat` | MLMC data, non-diagonal `Gamma` (panel c). |

The shared MLMC engine `MLMC_RBM_Trial_Mem.m`, the Skorokhod map
`Skorokhod_linear.m`, and the hypoexponential CCDF/PDF routine `hypoexp_ccdf.m`
live in `../lib/`. The multi-scaling and skew-symmetric coordinate means are
hard-coded in the Python script (`MULTI_SCALING_BY_SIGMA`, `M_SKEW`).

## Reproduce

The MLMC reference data is bundled, so the figure regenerates directly:

```bash
cd exp2_distribution
python make_figure.py
# -> outputs/figures_paper/fig_sec3_3_dist.{pdf,png}  (+ standalone panels)
```

Python dependencies: `numpy`, `matplotlib`, `h5py` (see `../requirements.txt`).

To regenerate the MLMC data from scratch (heavy — tens of CPU-hours for the
`L = 8` runs; uses the Parallel Computing Toolbox). Run with MATLAB directly,
no scheduler required:

```bash
cd exp2_distribution
WARMSTART=origin SEC33_L=8 SEC33_SIGMA=identity matlab -batch generate_data  # panels a, b
WARMSTART=origin SEC33_L=8 SEC33_SIGMA=corr_v1  matlab -batch generate_data  # panel c
```
