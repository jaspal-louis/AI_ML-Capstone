"""Surrogate quality after week 13 — can each model predict a point it has not seen?

Computes leave-one-out Q2 and training R2 for all eight surrogates, writes
runs/week_13/surrogate_quality.csv, and draws figures/surrogate_quality.png
for the model card.

Q2 is the honest counterpart to R2. Where the fitted noise is negligible the GP
reproduces its own observations exactly, so R2 is ~1.000 whatever the model is
worth. Q2 removes each observation in turn, refits, and predicts it.

The leave-one-out loop refits once per observation, so the cost is
(n_obs x RESTARTS) fits per function — a few minutes in total. RESTARTS is set
high enough that the estimate is stable; at 2 restarts (as in the notebook's
inline version, which is tuned for a fast end-to-end run) f6's Q2 moves by more
than 0.2 between runs.

Run from the repository root:  python scripts/make_quality_figure.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)

warnings.filterwarnings("ignore")

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir())
OUT_CSV = ROOT / "runs" / "week_13" / "surrogate_quality.csv"
OUT_PNG = ROOT / "figures" / "surrogate_quality.png"

RESTARTS = 10      # restarts per leave-one-out refit
ALPHA = 1e-6       # jitter, as used by the weekly runs
SEED = 0           # fixed so the table is reproducible
NU = 2.5

INK, ACC, NEG = "#1F1F1F", "#8A6D3B", "#B03A2E"


def length_scale_bounds(fn):
    """Per-function bounds, matching build_kernel() in the notebook."""
    if fn == 8:
        return (1e-3, 10.0)
    if fn == 1:
        return (0.05, 5.0)
    return (1e-3, 5.0)


def transform(fn, y):
    """f1 and f5 are modelled on log10; f1 floors its negatives at -300."""
    if fn == 1:
        return np.where(y > 0, np.log10(np.abs(y)), -300.0)
    if fn == 5:
        return np.log10(y)
    return y.astype(float)


def kernel(fn, d):
    return C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d),
        length_scale_bounds=length_scale_bounds(fn),
        nu=NU,
    ) + WhiteKernel(1e-6, (1e-10, 1e-1))


def fit(fn, X, y, restarts):
    return GaussianProcessRegressor(
        kernel=kernel(fn, X.shape[1]),
        alpha=ALPHA,
        n_restarts_optimizer=restarts,
        normalize_y=True,
        random_state=SEED,
    ).fit(X, y)


def quality(fn):
    df = pd.read_csv(ROOT / "data" / f"function_{fn}.csv")
    cols = [c for c in df.columns if c.startswith("x")]
    X = df[cols].values
    y = transform(fn, df["output"].values)

    loo = np.empty(len(y))
    for i in range(len(y)):
        gp = fit(fn, np.delete(X, i, axis=0), np.delete(y, i), RESTARTS)
        loo[i] = gp.predict(X[i : i + 1])[0]

    ss_tot = np.sum((y - y.mean()) ** 2)
    q2 = 1 - np.sum((y - loo) ** 2) / (ss_tot + 1e-12)

    gp_full = fit(fn, X, y, RESTARTS)
    resid = y - gp_full.predict(X)
    r2 = 1 - np.sum(resid**2) / (ss_tot + 1e-12)

    return dict(fn=fn, d=len(cols), n=len(y), R2=r2, Q2=q2,
                max_resid=np.abs(resid).max())


def draw(d):
    d = d.sort_values("Q2")
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12,
                         "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 10})
    lab = [f"f{int(r.fn)}  ({int(r.d)}D, n={int(r.n)})" for r in d.itertuples()]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=120)
    ax.barh(y, d.Q2, color=[NEG if q < 0.5 else ACC for q in d.Q2],
            height=0.62, zorder=3)
    ax.scatter(d.R2, y, marker="|", s=170, color=INK, lw=2, zorder=5,
               label="training R² (≈1 wherever the fit interpolates)")
    ax.axvline(0, color="0.35", lw=1.1, zorder=4)
    for i, q in enumerate(d.Q2):
        inside = q >= 0.9
        ax.text(q - 0.022 if inside else (q + 0.022 if q >= 0 else q - 0.022),
                i, f"{q:.2f}", va="center",
                ha="right" if (inside or q < 0) else "left",
                fontsize=9.5, color="white" if inside else INK)
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.set_xlim(-0.45, 1.15)
    ax.set_xlabel("Q²  (leave-one-out predictive R²)")
    ax.set_title("Surrogate quality after week 13 — can the model predict a "
                 "point it has not seen?", pad=10)
    ax.grid(axis="x", color="0.93", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.legend(loc="lower center", bbox_to_anchor=(0.55, -0.06),
               frameon=False, fontsize=9.5)
    ax.text(-0.43, 3.4, "below zero = worse than\npredicting the mean;\n"
            "the model was abandoned\nfor f1 from week 5",
            fontsize=8.5, color=NEG, va="center", ha="left")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    print(f"Leave-one-out Q² with {RESTARTS} restarts per refit — a few minutes.\n")
    print(f"{'fn':>4} {'n':>4} {'R²':>10} {'Q²':>8} {'max resid':>11}")
    rows = []
    for fn in range(1, 9):
        r = quality(fn)
        rows.append(r)
        print(f"  f{r['fn']} {r['n']:>4} {r['R2']:>10.5f} {r['Q2']:>8.3f} "
              f"{r['max_resid']:>11.3g}")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}")
    draw(df)
