"""What kind of design are the supplied initial points?

The course supplied the initial design for each function without saying how it
was generated. This script tests the data against the candidates, so the
datasheet can record what is known rather than what was assumed.

Three tests:

1. Dyadic structure. An unscrambled Sobol sequence places every coordinate at
   k / 2**m, so a large fraction of coordinates should land exactly on a dyadic
   grid.

2. One-dimensional stratification. Sobol and Latin hypercube designs both
   spread their points evenly along each axis. Splitting an axis into n equal
   bins, a Latin hypercube fills all n, a scrambled Sobol sequence fills about
   80%, and i.i.d. uniform points fill about 1 - 1/e = 63%.

3. L2-star discrepancy, the standard measure of space-filling quality, with the
   observed design placed as a percentile of 500 uniform-random draws of the
   same size. A genuinely space-filling design should sit in the bottom few
   percent; a uniform-random one sits anywhere.

Run from the repository root:  python scripts/initial_design_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir())

# initial design sizes supplied by the course, per function
N_INIT = {1: 10, 2: 10, 3: 15, 4: 30, 5: 20, 6: 20, 7: 30, 8: 40}
N_REF = 60     # reference designs per comparison
N_DRAWS = 500  # uniform draws for the discrepancy percentile
EPS = 1e-9


def load(fn):
    df = pd.read_csv(ROOT / "data" / f"function_{fn}.csv")
    cols = [c for c in df.columns if c.startswith("x")]
    return df[cols].values[: N_INIT[fn]], len(cols)


def dyadic_fraction(X):
    """Largest fraction of coordinates lying exactly on a k/2**m grid, m <= 12."""
    return max(
        np.isclose(X * 2**m, np.round(X * 2**m), atol=1e-9).mean()
        for m in range(1, 13)
    )


def stratification(X):
    """Mean fraction of the n equal bins per axis that contain at least one point."""
    n = len(X)
    return np.mean([
        len(set(np.clip((X[:, j] * n).astype(int), 0, n - 1))) / n
        for j in range(X.shape[1])
    ])


def discrepancy(X):
    return qmc.discrepancy(np.clip(X, EPS, 1 - EPS), method="L2-star")


def main():
    rng = np.random.default_rng(1)

    print("Test 1 — dyadic structure (unscrambled Sobol would be close to 1.00)\n")
    for fn in N_INIT:
        X, _ = load(fn)
        print(f"  f{fn}: {dyadic_fraction(X):.1%} of coordinates on a dyadic grid")

    print("\nTest 2 — 1-D stratification (fraction of n bins per axis occupied)\n")
    print(f"  {'':<6} {'observed':>9} {'Sobol':>9} {'LHS':>9} {'uniform':>9}")
    for fn in N_INIT:
        X, d = load(fn)
        n = len(X)
        sob = np.mean([stratification(qmc.Sobol(d, scramble=True, seed=s).random(n))
                       for s in range(N_REF)])
        lhs = np.mean([stratification(qmc.LatinHypercube(d, seed=s).random(n))
                       for s in range(N_REF)])
        uni = np.mean([stratification(rng.random((n, d))) for _ in range(N_REF)])
        print(f"  f{fn:<5} {stratification(X):9.3f} {sob:9.3f} {lhs:9.3f} {uni:9.3f}")

    print("\nTest 3 — L2-star discrepancy, lower is better\n")
    print(f"  {'':<6} {'observed':>10} {'Sobol':>10} {'uniform':>10} {'pctile':>8}")
    for fn in N_INIT:
        X, d = load(fn)
        n = len(X)
        obs = discrepancy(X)
        sob = np.mean([discrepancy(qmc.Sobol(d, scramble=True, seed=s).random(n))
                       for s in range(N_REF)])
        draws = np.array([discrepancy(rng.random((n, d))) for _ in range(N_DRAWS)])
        pct = 100 * (draws < obs).mean()
        print(f"  f{fn:<5} {obs:10.3e} {sob:10.3e} {draws.mean():10.3e} {pct:7.0f}%")

    print("\nReading: the designs are worse than scrambled Sobol on every function")
    print("and sit at an unremarkable percentile of uniform-random draws, so they")
    print("are not Sobol and not a Latin hypercube. The evidence is consistent with")
    print("i.i.d. uniform sampling, but the generator itself was never disclosed.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # Sobol warns when n is not a power of 2
    main()
