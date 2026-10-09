"""
Generate the per-function figures used in docs/f1.md ... docs/f8.md.

For each function this produces two PNGs in figures/:

  f{n}_surface.png   the GP mean after week 13, sliced through the two most
                     sensitive dimensions (by ARD length scale) with the rest
                     held at the best observed point. Left panel: full domain.
                     Right panel: a zoom scaled to the plotted axes' length
                     scales. Observations close enough to the slice in every
                     held dimension -- within half an ARD length scale, so the
                     GP mean is still strongly correlated with what is drawn --
                     are filled and coloured on the surface's own scale; the
                     rest are faint rings.

  f{n}_progress.png  the best observed value by week, with the initial design's
                     best as a baseline. For f1, weeks returning a negative
                     output are marked on a separate strip, since they have no
                     position on a log scale.

Run from the repository root:

    python scripts/make_figures.py 1 2 3 4 5 6 7 8

The GP fitted here matches the notebook's canonical configuration (Matern 2.5
with ARD, 50 restarts, free WhiteKernel), so the surfaces correspond to the
week-13 diagnostics in runs/week_13/ rather than being a separate model.
"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

INK, ACCENT, NEG = "#1F1F1F", "#8A6D3B", "#B03A2E"
N_INIT = {1:10,2:10,3:15,4:30,5:20,6:20,7:30,8:40}
LOG = {1: -300.0, 5: None}          # fn -> negative floor (None = log, no floor)
# "near the slice": within half an ARD length scale in every held dimension, i.e. close
# enough that the GP mean is still strongly correlated with the plotted slice.
TOL_LS  = 0.5
TOL_MIN = 0.05
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5})

def load(fn):
    df = pd.read_csv(f"data/function_{fn}.csv")
    cols = [c for c in df.columns if c != "output"]
    X, y = df[cols].values, df["output"].values
    if fn in LOG:
        floor = LOG[fn]
        yt = np.log10(np.abs(y) + 1e-320)
        if floor is not None: yt = np.where(y > 0, yt, floor)
        unit = "log$_{10}$"
    else:
        yt, unit = y.copy(), ""
    return df, cols, X, y, yt, unit

def fit(fn, X, yt, d):
    lo = 0.05 if fn == 1 else 1e-3
    hi = 10.0 if fn == 8 else 5.0
    k = C(1.0,(1e-3,1e3))*Matern(np.ones(d), (lo, hi), nu=2.5) + WhiteKernel(1e-6,(1e-10,1e-1))
    gp = GPR(kernel=k, alpha=1e-6, n_restarts_optimizer=50, normalize_y=True,
             random_state=0).fit(X, yt)   # fixed so the committed figures reproduce
    return gp, np.atleast_1d(gp.kernel_.k1.k2.length_scale)

def surface(fn):
    df, cols, X, y, yt, unit = load(fn)
    d = len(cols); gp, ls = fit(fn, X, yt, d)
    order = np.argsort(ls); a, b = sorted(order[:2])          # two most sensitive dims
    best = X[int(np.argmax(yt))]
    held = [i for i in range(d) if i not in (a, b)]
    near = np.ones(len(X), bool)
    for i in held: near &= np.abs(X[:, i] - best[i]) <= max(TOL_MIN, TOL_LS * ls[i])
    far = ~near

    res = 300; g = np.linspace(0, 1, res); G1, G2 = np.meshgrid(g, g)
    P = np.tile(best, (res*res, 1)); P[:, a] = G1.ravel(); P[:, b] = G2.ravel()
    mu = gp.predict(P).reshape(res, res)
    vmin, vmax = np.percentile(mu, 5), yt.max()

    # zoom window scaled to the ARD length scale of each plotted axis, so the panel shows
    # real curvature rather than a flat patch; shifted (not clipped) when the best point
    # sits on a boundary.
    def window(i):
        p = float(np.clip(0.6 * ls[i], 0.04, 0.30))
        lo_, hi_ = best[i] - p, best[i] + p
        if lo_ < 0: lo_, hi_ = 0.0, min(1.0, 2 * p)
        if hi_ > 1: lo_, hi_ = max(0.0, 1 - 2 * p), 1.0
        return (lo_, hi_)
    zx, zy = window(a), window(b)
    fig, (p1, p2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=120)
    for ax, lim in ((p1, None), (p2, (zx, zy))):
        cf = ax.contourf(G1, G2, np.clip(mu, vmin, vmax), levels=40, cmap="plasma",
                         vmin=vmin, vmax=vmax, extend="max")
        ax.scatter(X[far, a], X[far, b], s=16, facecolors="none", edgecolors="white",
                   linewidths=0.8, alpha=0.45, zorder=3)
        ax.scatter(X[near, a], X[near, b], s=40, c=np.clip(yt[near], vmin, vmax), cmap="plasma",
                   vmin=vmin, vmax=vmax, edgecolors="white", linewidths=1.0, zorder=4)
        ax.scatter(best[a], best[b], marker="*", s=260, c="white", edgecolors="0.1",
                   linewidths=1.0, zorder=6, clip_on=False)
        ax.set_xlabel(cols[a])
        # small outward margin so a best point pinned to a domain wall is fully visible
        (xl, yl) = lim if lim else ((0.0, 1.0), (0.0, 1.0))
        mx, my = 0.025*(xl[1]-xl[0]), 0.025*(yl[1]-yl[0])
        ax.set_xlim(xl[0]-mx, xl[1]+mx); ax.set_ylim(yl[0]-my, yl[1]+my)
    p1.set_ylabel(cols[b]); p1.set_title("full domain"); p2.set_title("around the best point")
    p1.indicate_inset_zoom(p2, edgecolor="white", alpha=0.9)
    cb = fig.colorbar(cf, ax=[p1, p2], pad=0.05, fraction=0.045)
    cb.set_label(f"GP mean $\\mu$ {('('+unit+')') if unit else ''}", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    handles = [Line2D([],[],marker="o",ls="",mfc="0.6",mec="white",ms=7,
                      label=f"on this slice, coloured by output ({near.sum()} of {len(X)})")]
    if far.any():
        handles.append(Line2D([],[],marker="o",ls="",mfc="none",mec="0.45",ms=5.5,
                              label=f"sampled elsewhere in the domain ({far.sum()})"))
    handles.append(Line2D([],[],marker="*",ls="",mfc="white",mec="0.1",ms=12,label="best observed"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.46, -0.10),
               ncol=3, frameon=False, handletextpad=0.5, columnspacing=2.0)
    held_txt = ("" if not held else
                "  |  " + ", ".join(f"{cols[i]}={best[i]:.3g}" for i in held) + " held")
    fig.suptitle(f"f{fn} — GP mean after week 13   ls {cols[a]}={ls[a]:.3f}, {cols[b]}={ls[b]:.3f}{held_txt}",
                 fontsize=11.5, y=0.99)
    fig.savefig(f"figures/f{fn}_surface.png", bbox_inches="tight"); plt.close(fig)
    return near.sum(), len(X), cols[a], cols[b], held

def progress(fn):
    df, cols, X, y, yt, unit = load(fn)
    n = N_INIT[fn]; wk = yt[n:]; base = yt[:n].max(); weeks = np.arange(1, len(wk)+1)
    floored = (fn == 1) & (wk <= -300)
    pos = ~floored
    run = np.maximum(np.maximum.accumulate(np.where(pos, wk, -np.inf)), base)
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120)
    lo_v = min(wk[pos].min(), base)
    if floored.any():
        strip = lo_v - 5
        ax.axhspan(strip-1.4, strip+1.4, color="0.95", zorder=0)
        ax.scatter(weeks[floored], np.full(floored.sum(), strip), marker="x", s=52,
                   color=NEG, lw=1.8, zorder=4, label="negative output")
        ylo = strip - 1.8
    else:
        ylo = lo_v - 0.06*abs(lo_v if lo_v else 1)
    ax.axhline(base, color="0.55", ls="--", lw=1.1, zorder=1)
    ax.step(weeks, run, where="post", color=ACCENT, lw=2.2, zorder=3, label="best so far")
    ax.scatter(weeks[pos], wk[pos], s=42, color=INK, zorder=4, label="weekly result")
    ax.text(len(wk)+0.35, base, f"initial best\n{base:.4g}", fontsize=9, color="0.4", va="center")
    ax.set_xlim(0.4, len(wk)+2.1); ax.set_ylim(ylo, max(wk[pos].max(), base) + 0.08*abs(max(wk[pos].max(), base) or 1))
    ax.set_xticks(weeks); ax.set_xlabel("week")
    ax.set_ylabel(f"output {('('+unit+')') if unit else ''}")
    ax.set_title(f"f{fn} — best observed value by week", fontsize=12, pad=30)
    ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
              handletextpad=0.5, columnspacing=1.8)
    fig.savefig(f"figures/f{fn}_progress.png", bbox_inches="tight"); plt.close(fig)

for fn in [int(a) for a in sys.argv[1:]]:
    k, tot, ca, cb, held = surface(fn); progress(fn)
    print(f"f{fn}: axes {ca}/{cb}; {len(held)} dims held; {k} of {tot} observations on the slice")
