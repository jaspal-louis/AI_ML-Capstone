"""Where does the information about a length scale actually live?

Two measurements, both reported in the README under "Should the distant
observations have been dropped?".

1. Analytic. For a stationary kernel, the sensitivity of the correlation to a
   change in the length scale, as a function of pair separation. It peaks at an
   intermediate separation and vanishes at both extremes: pairs much closer than
   l are almost perfectly correlated, pairs much further apart are uncorrelated,
   and neither responds to a change in l.

2. Empirical. Refit the model on nested neighbourhoods of the final best point,
   with radius measured in length-scale units, and compare each fit's length
   scales against the full-data fit.

Run from the repository root:  python scripts/length_scale_support.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)

warnings.filterwarnings("ignore")

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir())
NU = 2.5
RADII = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def corr_m52(u):
    """Matern 2.5 correlation at separation u, in units of the length scale."""
    a = np.sqrt(5.0) * u
    return (1.0 + a + a * a / 3.0) * np.exp(-a)


def sensitivity(u, h=1e-6):
    """|d corr / d l| at l = 1, i.e. how much this pair responds to a change in l."""
    return abs((corr_m52(u / (1 + h)) - corr_m52(u / (1 - h))) / (2 * h))


def analytic():
    peak = minimize_scalar(lambda u: -sensitivity(u), bounds=(0.01, 8), method="bounded").x
    grid = np.array([0.1, 0.25, 0.5, 1.0, peak, 2.0, 3.0, 5.0])
    rel = np.array([sensitivity(u) for u in grid])
    rel /= rel.max()
    print(f"Matern {NU}: sensitivity to the length scale peaks at a separation of "
          f"{peak:.2f} x l\n")
    print("  separation r/l :", "  ".join(f"{u:5.2f}" for u in grid))
    print("  relative info  :", "  ".join(f"{v:5.2f}" for v in rel))
    print("\n  Pairs at 5 l contribute essentially nothing, and so do pairs at 0.1 l.")


def fit(X, y, restarts=30):
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * X.shape[1],
        length_scale_bounds=(1e-2, 1e1),
        nu=NU,
    ) + WhiteKernel(1e-8, (1e-10, 1e-6))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=restarts, normalize_y=True, random_state=0
    ).fit(X, y)
    return np.atleast_1d(gp.kernel_.k1.k2.length_scale)


def empirical(functions=(4, 6, 7)):
    for fn in functions:
        df = pd.read_csv(ROOT / "data" / f"function_{fn}.csv")
        cols = [c for c in df.columns if c.startswith("x")]
        X, y = df[cols].values, df["output"].values
        full = fit(X, y)
        best = X[y.argmax()]
        # distance from the best point, measured in length-scale units per dimension
        r = np.linalg.norm((X - best) / full, axis=1)

        print(f"\n=== f{fn}  ({len(cols)}D, n={len(X)}) ===")
        print(f"full-data length scales: {np.round(full, 3)}")
        print(f"{'radius':>8} {'n':>4}   mean |log ratio| vs full")
        for radius in RADII:
            keep = r <= radius
            if keep.sum() < max(6, len(cols) + 3):
                print(f"{radius:>7.1f} l {keep.sum():>4}   (too few points to fit)")
                continue
            if keep.all():
                print(f"{radius:>7.1f} l {keep.sum():>4}   -- (whole dataset)")
                continue
            dev = np.abs(np.log(fit(X[keep], y[keep]) / full)).mean()
            print(f"{radius:>7.1f} l {keep.sum():>4}   {dev:.2f}")


if __name__ == "__main__":
    print("=" * 68)
    print("1. Analytic: where the length-scale information lives")
    print("=" * 68)
    analytic()
    print()
    print("=" * 68)
    print("2. Empirical: length scales fitted on nested neighbourhoods")
    print("=" * 68)
    empirical()
