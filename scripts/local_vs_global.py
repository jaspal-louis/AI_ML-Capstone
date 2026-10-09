"""
Test whether fitting the GP only to points near the optimum would have beaten
fitting it to everything.

The question this answers: by the later weeks the search was concentrated in one
region of each domain, so what were the distant observations still contributing?
Would a model fitted only to the local cluster have predicted the region of
interest better?

Method. For each function, take the m observations nearest the best observed
point -- the exploitation region. Leave each of them out in turn and predict it
three ways:

  global    fit on all remaining observations, hyperparameters fitted to them
  local     fit on the k nearest remaining observations, hyperparameters
            re-fitted on that subset
  hybrid    hyperparameters from the full dataset, but conditioned only on the
            k nearest observations

Report RMSE over the m held-out points. The three differ only in what the model
was allowed to learn from, so the comparison isolates the value of the distant
points.

Caveats. Restart count is lower than the weekly runs (10 vs 50), so there is
some optimiser noise; and leave-one-out on adaptively-sampled data is optimistic
in absolute terms. Both apply equally to all three columns.

Run from the repository root:

    python scripts/local_vs_global.py
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

RESTARTS = 10
SEED = 0                            # fixed so the table below reproduces exactly
LOG = {1: -300.0, 5: None}          # fn -> negative floor (None = log, no floor)


def build_kernel(d, fn):
    lo = 0.05 if fn == 1 else 1e-3          # f1 needs a floor or the fit collapses
    hi = 10.0 if fn == 8 else 5.0           # f8 has eight dimensions
    return (C(1.0, (1e-3, 1e3)) * Matern(np.ones(d), (lo, hi), nu=2.5)
            + WhiteKernel(1e-6, (1e-10, 1e-1)))


def fit(X, y, fn, restarts=RESTARTS):
    return GPR(kernel=build_kernel(X.shape[1], fn), alpha=1e-6,
               n_restarts_optimizer=restarts, normalize_y=True,
               random_state=SEED).fit(X, y)


def load(fn):
    df = pd.read_csv(f"data/function_{fn}.csv")
    cols = [c for c in df.columns if c != "output"]
    X, y = df[cols].values, df["output"].values
    if fn in LOG:
        floor = LOG[fn]
        yt = np.log10(np.abs(y) + 1e-320)
        if floor is not None:
            yt = np.where(y > 0, yt, floor)
    else:
        yt = y.copy()
    return X, yt


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def compare(fn):
    X, yt = load(fn)
    n = len(yt)
    best = X[int(np.argmax(yt))]
    m = min(12, n // 3)                     # size of the exploitation region tested
    k = min(10, n // 3)                     # neighbours the local models may use
    near = np.argsort(np.linalg.norm(X - best, axis=1))[:m]

    gp_full = fit(X, yt, fn)                # hyperparameters from the whole dataset
    e_global, e_local, e_hybrid = [], [], []

    for i in near:
        keep = np.arange(n) != i
        Xk, yk = X[keep], yt[keep]

        e_global.append(fit(Xk, yk, fn).predict(X[i:i + 1])[0] - yt[i])

        loc = np.argsort(np.linalg.norm(Xk - X[i], axis=1))[:k]
        e_local.append(fit(Xk[loc], yk[loc], fn).predict(X[i:i + 1])[0] - yt[i])

        hyb = GPR(kernel=gp_full.kernel_, optimizer=None, alpha=1e-6,
                  normalize_y=True).fit(Xk[loc], yk[loc])
        e_hybrid.append(hyb.predict(X[i:i + 1])[0] - yt[i])

    return n, m, rmse(e_global), rmse(e_local), rmse(e_hybrid)


if __name__ == "__main__":
    print(f'{"":>3} {"n":>3} {"m":>3} | {"global":>10} {"local":>10} {"hybrid":>10} | winner')
    for fn in range(1, 9):
        n, m, g, l, h = compare(fn)
        winner = ["global", "local", "hybrid"][int(np.argmin([g, l, h]))]
        print(f"f{fn:<2} {n:>3} {m:>3} | {g:>10.4g} {l:>10.4g} {h:>10.4g} | {winner}")
