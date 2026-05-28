#!/usr/bin/env python3
"""Generate the standalone subpanel PDFs and the combined distribution figure:

    fig_sec3_3_pdf_diag.pdf   - panel (a)  PDF,  Γ = I_3,            origin L=8
    fig_sec3_3_ccdf_diag.pdf  - panel (b)  CCDF, Γ = I_3,            origin L=8  (semi-log)
    fig_sec3_3_ccdf_corr.pdf  - panel (c)  CCDF, non-diagonal Γ,     origin L=8  (semi-log)

Each panel is a self-contained figure (own axes, own legend) sized so that
\includegraphics[width=\textwidth] inside a 0.32\textwidth subfigure looks
sensible. No matplotlib titles—LaTeX subfigure captions supply the (a)/(b)/(c)
labels in the paper.

Curves and styling match the paragraph in the paper:
  - MLMC mean: black solid
  - MLMC 95% CI: gray band
  - multi-scaling prediction: red dashed
  - skew-symmetric prediction: blue dash-dot

PDF panel uses binned histogram with bin Δs=10 (computed by interpolating
each MLMC estimator's CCDF at uniform bin edges, differencing, dividing by
the bin width—per the identity stated in the paragraph).
"""

from __future__ import annotations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "data"
OUTDIR  = ROOT / "outputs" / "figures_paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

GAMMA_TAG = "gamma_0p2"
NEST_TAG  = "Nest_50_Neps_2000"
T_TAG     = "T_100000"

MULTI_SCALING_BY_SIGMA = {
    "identity": np.array([2.5, 22.32142857, 179.6875]),
    "corr_v1":  np.array([2.5, 11.60714286, 52.45535714]),
}
M_SKEW = np.array([2.5, 12.5, 62.5])

BIN_WIDTH_PDF = 10.0   # bin Δs for the PDF panel (Γ = I_3)

# Per-panel single-figure size: ~3.0 x 2.8 inches.  LaTeX will scale to
# 0.32\textwidth.  Keeping the aspect uniform across the three panels.
PANEL_SIZE = (3.4, 2.9)


def hypoexp_ccdf(means: np.ndarray, t: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    rates = 1.0 / means
    n = len(rates)
    if n != len(set(rates.tolist())):
        rates = rates + np.linspace(0.0, 1e-9, n)
    out = np.zeros_like(t, dtype=float)
    for k in range(n):
        coeff = 1.0
        for j in range(n):
            if j != k:
                coeff *= rates[j] / (rates[j] - rates[k])
        out += coeff * np.exp(-rates[k] * t)
    return out


def hypoexp_pdf(means: np.ndarray, t: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    rates = 1.0 / means
    n = len(rates)
    if n != len(set(rates.tolist())):
        rates = rates + np.linspace(0.0, 1e-9, n)
    out = np.zeros_like(t, dtype=float)
    for k in range(n):
        coeff = 1.0
        for j in range(n):
            if j != k:
                coeff *= rates[j] / (rates[j] - rates[k])
        out += coeff * rates[k] * np.exp(-rates[k] * t)
    return out


def load_case(warmstart: str, L: int, sigma_tag: str) -> dict:
    sigma_suffix = "" if sigma_tag == "identity" else f"_sigma_{sigma_tag}"
    fname = (f"{warmstart}{sigma_suffix}_{GAMMA_TAG}"
             f"_L_{L}_{T_TAG}_{NEST_TAG}.mat")
    fpath = RESULTS / fname
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    with h5py.File(fpath, "r") as f:
        completed = int(np.array(f["completed"]).flatten()[0])
        t_grid    = np.array(f["config/raw_t_grid"]).flatten()
        est       = np.array(f["raw_est_ccdf_grid"])
        if est.shape[0] == completed and est.shape[1] != completed:
            est = est.T
        est = est[:, :completed]
    return dict(t_grid=t_grid, est_ccdf=est, completed=completed,
                sigma_tag=sigma_tag, warmstart=warmstart, L=L)


def save(fig: plt.Figure, basename: str) -> None:
    for ext in ("pdf", "png"):
        out = OUTDIR / f"{basename}.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


# -------------------------- panel (a): PDF, Γ = I_3 --------------------------

def render_pdf_diag() -> None:
    c = load_case("origin", L=8, sigma_tag="identity")
    t = c["t_grid"]
    est = c["est_ccdf"]                              # (n_grid, completed)

    xmax = 800.0
    bw   = BIN_WIDTH_PDF
    edges = np.arange(0.0, xmax + bw, bw)
    ccdf_edges = np.empty((len(edges), c["completed"]))
    for j in range(c["completed"]):
        ccdf_edges[:, j] = np.interp(edges, t, est[:, j])
    bin_mass = ccdf_edges[:-1, :] - ccdf_edges[1:, :]
    widths   = np.diff(edges)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    pdf_perE = bin_mass / widths[:, None]
    pdf_mean = pdf_perE.mean(axis=1)
    pdf_se   = pdf_perE.std(axis=1, ddof=1) / np.sqrt(c["completed"])

    skew_pdf = hypoexp_pdf(M_SKEW, t)
    ms_pdf   = hypoexp_pdf(MULTI_SCALING_BY_SIGMA["identity"], t)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.bar(centers, pdf_mean, width=widths, color="0.65",
           edgecolor="black", linewidth=0.3, alpha=0.85, label="MLMC")
    ax.errorbar(centers, pdf_mean, yerr=1.96 * pdf_se,
                fmt="none", ecolor="black", linewidth=0.5, capsize=0, alpha=0.5)
    ax.plot(t, ms_pdf,   color="#d62728", linestyle="--",  linewidth=1.6,
            label="multi-scaling")
    ax.plot(t, skew_pdf, color="#1f77b4", linestyle="-.", linewidth=1.6,
            label="skew symmetric")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 0.013)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$f_S(s)$")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_sec3_3_pdf_diag")


# -------------------------- panel (b): CCDF, Γ = I_3 -------------------------

def render_ccdf_diag() -> None:
    c = load_case("origin", L=8, sigma_tag="identity")
    t = c["t_grid"]
    est = c["est_ccdf"]
    est_mean = est.mean(axis=1)
    est_se   = est.std(axis=1, ddof=1) / np.sqrt(c["completed"])

    skew_ccdf = hypoexp_ccdf(M_SKEW, t)
    ms_ccdf   = hypoexp_ccdf(MULTI_SCALING_BY_SIGMA["identity"], t)

    y_lo = 1e-3
    # x_max = where the slowest of (mlmc, ms) drops below y_lo
    last = 0
    for m in (ms_ccdf, est_mean):
        ok = np.where(m >= y_lo)[0]
        if len(ok):
            last = max(last, ok[-1])
    xmax = t[min(last + 5, len(t) - 1)]

    lo = np.maximum(est_mean - 1.96 * est_se, y_lo / 10)
    hi = np.maximum(est_mean + 1.96 * est_se, y_lo / 10)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.fill_between(t, lo, hi, color="0.78", alpha=0.55, linewidth=0,
                    label="MLMC 95% CI")
    ax.plot(t, np.maximum(est_mean, y_lo / 10), color="black", linewidth=1.6,
            label="MLMC")
    ax.plot(t, np.maximum(ms_ccdf, y_lo / 10), color="#d62728", linestyle="--",
            linewidth=1.6, label="multi-scaling")
    ax.plot(t, np.maximum(skew_ccdf, y_lo / 10), color="#1f77b4",
            linestyle="-.", linewidth=1.6, label="skew symmetric")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(y_lo, 1.05)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$P(S > s)$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_sec3_3_ccdf_diag")


# ---------------------- panel (c): CCDF, non-diagonal Γ ----------------------

def render_ccdf_corr() -> None:
    c = load_case("origin", L=8, sigma_tag="corr_v1")
    t = c["t_grid"]
    est = c["est_ccdf"]
    est_mean = est.mean(axis=1)
    est_se   = est.std(axis=1, ddof=1) / np.sqrt(c["completed"])

    skew_ccdf = hypoexp_ccdf(M_SKEW, t)
    ms_ccdf   = hypoexp_ccdf(MULTI_SCALING_BY_SIGMA["corr_v1"], t)

    y_lo = 1e-3
    xmax = 500.0

    lo = np.maximum(est_mean - 1.96 * est_se, y_lo / 10)
    hi = np.maximum(est_mean + 1.96 * est_se, y_lo / 10)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.fill_between(t, lo, hi, color="0.78", alpha=0.55, linewidth=0,
                    label="MLMC 95% CI")
    ax.plot(t, np.maximum(est_mean, y_lo / 10), color="black", linewidth=1.6,
            label="MLMC")
    ax.plot(t, np.maximum(ms_ccdf, y_lo / 10), color="#d62728", linestyle="--",
            linewidth=1.6, label="multi-scaling")
    ax.plot(t, np.maximum(skew_ccdf, y_lo / 10), color="#1f77b4",
            linestyle="-.", linewidth=1.6, label="skew symmetric")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(y_lo, 1.05)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$P(S > s)$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_sec3_3_ccdf_corr")


def render_combined() -> None:
    """Single 1x3 figure stitching (a), (b), (c) together, ready for one
    \\includegraphics in the paper. Each subpanel keeps its own y-axis
    (linear vs log) and x-range. Subplot titles are just (a)/(b)/(c)
    centered; per-panel content is described in the LaTeX caption.
    A single figure-level legend sits above all three panels."""
    c_diag = load_case("origin", L=8, sigma_tag="identity")
    c_corr = load_case("origin", L=8, sigma_tag="corr_v1")
    t = c_diag["t_grid"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # --- (a) PDF diag ---
    est = c_diag["est_ccdf"]
    bw = BIN_WIDTH_PDF
    xmax_a = 800.0
    edges = np.arange(0.0, xmax_a + bw, bw)
    ccdf_e = np.empty((len(edges), c_diag["completed"]))
    for j in range(c_diag["completed"]):
        ccdf_e[:, j] = np.interp(edges, t, est[:, j])
    bin_mass = ccdf_e[:-1, :] - ccdf_e[1:, :]
    widths   = np.diff(edges)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    pdf_perE = bin_mass / widths[:, None]
    pdf_mean = pdf_perE.mean(axis=1)
    pdf_se   = pdf_perE.std(axis=1, ddof=1) / np.sqrt(c_diag["completed"])
    skew_pdf = hypoexp_pdf(M_SKEW, t)
    ms_pdf_d = hypoexp_pdf(MULTI_SCALING_BY_SIGMA["identity"], t)
    ax = axes[0]
    h_mlmc_bar = ax.bar(centers, pdf_mean, width=widths, color="0.65",
                        edgecolor="black", linewidth=0.3, alpha=0.85,
                        label="MLMC")
    ax.errorbar(centers, pdf_mean, yerr=1.96 * pdf_se,
                fmt="none", ecolor="black", linewidth=0.5, capsize=0, alpha=0.5)
    h_ms,   = ax.plot(t, ms_pdf_d,  color="#d62728", linestyle="--",
                      linewidth=1.6, label="multi-scaling")
    h_skew, = ax.plot(t, skew_pdf,  color="#1f77b4", linestyle="-.",
                      linewidth=1.6, label="skew symmetric")
    ax.set_xlim(0, xmax_a); ax.set_ylim(0, 0.013)
    ax.set_xlabel(r"$s$"); ax.set_ylabel(r"$f_S(s)$")
    ax.set_title("(a)", fontsize=11, loc="center")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # --- (b) CCDF diag ---
    est_mean = est.mean(axis=1)
    est_se   = est.std(axis=1, ddof=1) / np.sqrt(c_diag["completed"])
    skew_ccdf = hypoexp_ccdf(M_SKEW, t)
    ms_ccdf_d = hypoexp_ccdf(MULTI_SCALING_BY_SIGMA["identity"], t)
    y_lo = 1e-3
    last = 0
    for m in (ms_ccdf_d, est_mean):
        ok = np.where(m >= y_lo)[0]
        if len(ok):
            last = max(last, ok[-1])
    xmax_b = t[min(last + 5, len(t) - 1)]
    lo = np.maximum(est_mean - 1.96 * est_se, y_lo / 10)
    hi = np.maximum(est_mean + 1.96 * est_se, y_lo / 10)
    ax = axes[1]
    ax.fill_between(t, lo, hi, color="0.78", alpha=0.55, linewidth=0)
    h_mlmc_line, = ax.plot(t, np.maximum(est_mean, y_lo / 10), color="black",
                           linewidth=1.6, label="MLMC")
    h_ms_b,      = ax.plot(t, np.maximum(ms_ccdf_d, y_lo / 10),
                           color="#d62728", linestyle="--", linewidth=1.6,
                           label="multi-scaling")
    h_skew_b,    = ax.plot(t, np.maximum(skew_ccdf, y_lo / 10),
                           color="#1f77b4", linestyle="-.", linewidth=1.6,
                           label="skew symmetric")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax_b); ax.set_ylim(y_lo, 1.05)
    ax.set_xlabel(r"$s$"); ax.set_ylabel(r"$P(S > s)$")
    ax.set_title("(b)", fontsize=11, loc="center")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(handles=[h_mlmc_line, h_ms_b, h_skew_b],
              loc="upper right", fontsize=8, framealpha=0.9)

    # --- (c) CCDF corr ---
    est_c      = c_corr["est_ccdf"]
    est_mean_c = est_c.mean(axis=1)
    est_se_c   = est_c.std(axis=1, ddof=1) / np.sqrt(c_corr["completed"])
    ms_ccdf_c  = hypoexp_ccdf(MULTI_SCALING_BY_SIGMA["corr_v1"], t)
    xmax_c     = 500.0
    lo_c = np.maximum(est_mean_c - 1.96 * est_se_c, y_lo / 10)
    hi_c = np.maximum(est_mean_c + 1.96 * est_se_c, y_lo / 10)
    ax = axes[2]
    ax.fill_between(t, lo_c, hi_c, color="0.78", alpha=0.55, linewidth=0)
    h_mlmc_c, = ax.plot(t, np.maximum(est_mean_c, y_lo / 10), color="black",
                        linewidth=1.6, label="MLMC")
    h_ms_c,   = ax.plot(t, np.maximum(ms_ccdf_c, y_lo / 10), color="#d62728",
                        linestyle="--", linewidth=1.6, label="multi-scaling")
    h_skew_c, = ax.plot(t, np.maximum(skew_ccdf, y_lo / 10), color="#1f77b4",
                        linestyle="-.", linewidth=1.6, label="skew symmetric")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax_c); ax.set_ylim(y_lo, 1.05)
    ax.set_xlabel(r"$s$"); ax.set_ylabel(r"$P(S > s)$")
    ax.set_title("(c)", fontsize=11, loc="center")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(handles=[h_mlmc_c, h_ms_c, h_skew_c],
              loc="upper right", fontsize=8, framealpha=0.9)

    # Single legend in panel (a) only — panels (b)(c) reuse the same colors
    axes[0].legend(handles=[h_mlmc_bar, h_ms, h_skew],
                   labels=["MLMC", "multi-scaling", "skew symmetric"],
                   loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_sec3_3_dist")


if __name__ == "__main__":
    render_pdf_diag()
    render_ccdf_diag()
    render_ccdf_corr()
    render_combined()
