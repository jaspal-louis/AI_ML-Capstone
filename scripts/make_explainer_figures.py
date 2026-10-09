"""
Generate the two explanatory figures used in the README's hyperparameter section.

  figures/length_scale.png  the same eight observations fitted at three different
                            length scales, showing under- and over-smoothing.
  figures/restarts.png      120 independent single-start optimiser runs on one real
                            dataset (f7 as it stood entering week 5), showing that
                            some land in worse modes -- including on the upper bound,
                            where the fit reports a dimension as irrelevant.

Run from the repository root:

    python scripts/make_explainer_figures.py
"""

import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
INK,ACC="#1F1F1F","#8A6D3B"
plt.rcParams.update({"font.size":10.5,"axes.titlesize":11.5,"axes.labelsize":10.5,
                     "xtick.labelsize":9,"ytick.labelsize":9})
rng=np.random.default_rng(3)
f=lambda x: np.sin(2*np.pi*x*1.1)+0.45*np.sin(6.0*np.pi*x)
Xtr=np.array([0.04,0.17,0.29,0.46,0.58,0.71,0.83,0.96])[:,None]; ytr=f(Xtr.ravel())
g=np.linspace(0,1,400)[:,None]
fig,axes=plt.subplots(1,3,figsize=(11,3.3),dpi=120,sharey=True)
for ax,ls,lab in zip(axes,[0.02,0.14,1.5],
        ["ℓ = 0.02  — too short","ℓ = 0.14  — about right","ℓ = 1.5  — too long"]):
    gp=GPR(kernel=C(1.0,"fixed")*Matern([ls],"fixed",nu=2.5),alpha=0.04,normalize_y=False).fit(Xtr,ytr)
    mu,sd=gp.predict(g,return_std=True)
    ax.fill_between(g.ravel(),mu-2*sd,mu+2*sd,color=ACC,alpha=0.16,lw=0)
    ax.plot(g,f(g.ravel()),color="0.72",lw=1.3,ls="--",zorder=2)
    ax.plot(g,mu,color=ACC,lw=2.1,zorder=3)
    ax.scatter(Xtr,ytr,s=34,color=INK,zorder=5)
    ax.set_title(lab); ax.set_xlabel("x"); ax.set_ylim(-2.6,2.6)
    ax.grid(color="0.93",lw=0.7); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("output")
axes[0].text(0.03,-2.4,"snaps back to the mean\nbetween observations",fontsize=8.5,color="0.35",va="bottom")
axes[1].text(0.03,-2.4,"follows the shape and\ninterpolates sensibly",fontsize=8.5,color="0.35",va="bottom")
axes[2].text(0.03,-2.4,"too stiff to follow\nthe real shape",fontsize=8.5,color="0.35",va="bottom")
fig.suptitle("The same eight observations, three different length scales",fontsize=12.5,y=1.02)
fig.text(0.5,-0.08,"solid: GP mean   shaded: ±2σ   dashed grey: the function being learned",
         ha="center",fontsize=9,color="0.4")
fig.savefig("figures/length_scale.png",bbox_inches="tight"); print("ok")

# --------------------------------------------------------------------------
# Figure 2: why restarts are needed
# --------------------------------------------------------------------------

import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
INK,ACC,NEG="#1F1F1F","#8A6D3B","#B03A2E"
plt.rcParams.update({"font.size":10.5,"axes.titlesize":11.5,"axes.labelsize":10.5,
                     "xtick.labelsize":9,"ytick.labelsize":9})
N=34                       # f7's data as it stood entering week 5
d=pd.read_csv("data/function_7.csv"); cols=[c for c in d.columns if c!="output"]
X=d[cols].values[:N]; y=d["output"].values[:N]; dim=len(cols)
rng=np.random.default_rng(0); res=[]
for _ in range(120):
    ls0=10**rng.uniform(-2,np.log10(5.0),dim)
    k=C(1.0,(1e-3,1e3))*Matern(ls0,(1e-3,5.0),nu=2.5)+WhiteKernel(1e-6,(1e-10,1e-1))
    gp=GPR(kernel=k,alpha=1e-6,n_restarts_optimizer=0,normalize_y=True).fit(X,y)
    res.append((np.atleast_1d(gp.kernel_.k1.k2.length_scale)[5], gp.log_marginal_likelihood_value_))
res=np.array(res); best=res[:,1].max(); good=res[:,1]>best-0.5

fig,ax=plt.subplots(figsize=(9.4,4.0),dpi=120)
j=lambda v: v*10**(rng.normal(0,0.012,len(v)))
ax.scatter(j(res[~good,0]),res[~good,1],s=34,facecolors="none",edgecolors=NEG,lw=1.2,
           alpha=0.85,zorder=3,label=f"stopped at a worse fit  ({(~good).sum()} of 120)")
ax.scatter(j(res[good,0]),res[good,1],s=36,color=ACC,alpha=0.7,zorder=4,
           label=f"reached the best fit  ({good.sum()} of 120)")
ax.set_xscale("log"); ax.axhline(best,color="0.6",ls="--",lw=1,zorder=1)
ax.annotate(f"the answer you want:\nx6 matters, ℓ ≈ {res[good,0].mean():.2f}",
            xy=(res[good,0].mean(),best), xytext=(0.0016,best-5.5), fontsize=9, color="0.3",
            arrowprops=dict(arrowstyle="->",color="0.55",lw=1))
cap=res[res[:,0]>4.5]
if len(cap):
    ax.annotate(f"stopped on the upper bound:\n\"x6 is irrelevant\"  ({len(cap)} starts)",
                xy=(cap[:,0].mean(),cap[:,1].mean()), xytext=(0.30,cap[:,1].mean()+7.5),
                fontsize=9, color=NEG, arrowprops=dict(arrowstyle="->",color=NEG,lw=1))
ax.set_xlabel("length scale fitted for x6   (log scale)")
ax.set_ylabel("log marginal likelihood")
ax.set_title("One dataset, 120 optimiser runs from random starting points  (f7, entering week 5)",pad=8)
ax.grid(color="0.93",lw=0.7); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="center left",framealpha=0.9,fontsize=9.5)
fig.savefig("figures/restarts.png",bbox_inches="tight")
print(f"best {best:.2f} | good {good.sum()}/120 | at-cap {len(cap)} | ls(best) {res[good,0].mean():.3f}")

# ── figures/prior_posterior.png ───────────────────────────────────────────────
# A GP is a distribution over whole functions. Left: curves drawn from the prior,
# before any data. Right: the same draws conditioned on four observations, with
# their average — the posterior mean — in black.

rng = np.random.default_rng(7)
gx = np.linspace(0, 1, 300)[:, None]
kern = C(1.0, "fixed") * Matern([0.14], "fixed", nu=2.5)

K = kern(gx) + 1e-9 * np.eye(len(gx))
prior = rng.multivariate_normal(np.zeros(len(gx)), K, size=8)

Xo = np.array([0.12, 0.35, 0.62, 0.88])[:, None]
yo = np.array([0.55, -0.95, 0.35, 1.15])
gp = GPR(kernel=kern, alpha=1e-8, normalize_y=False).fit(Xo, yo)
mu, sd = gp.predict(gx, return_std=True)
post = gp.sample_y(gx, n_samples=8, random_state=7).T

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=120, sharey=True)
pal = ["#4E7A8A", "#8A6D3B", "#6B5B7B", "#5F7A5F", "#A0685A",
       "#7A8CA0", "#937B52", "#5E6E7A"]   # muted, matching the repo palette

axes[0].fill_between(gx.ravel(), -2, 2, color=ACC, alpha=0.12, lw=0)
for c, col in zip(prior, pal):
    axes[0].plot(gx, c, lw=1.4, color=col, alpha=0.9)
axes[0].axhline(0, color="0.6", lw=1.1, ls="--")
axes[0].set_title("Before any data — the prior")
axes[0].text(0.03, -2.75, "every curve is a function the model\n"
             "considers plausible before seeing anything",
             fontsize=8.5, color="0.35", va="bottom")

axes[1].fill_between(gx.ravel(), mu - 2 * sd, mu + 2 * sd, color=ACC, alpha=0.16, lw=0)
for c, col in zip(post, pal):
    axes[1].plot(gx, c, lw=1.4, color=col, alpha=0.9)
axes[1].plot(gx, mu, color=INK, lw=2.6, zorder=6)
axes[1].scatter(Xo, yo, s=58, color=INK, zorder=7)
axes[1].set_title("After four observations — the posterior")
axes[1].text(0.03, -2.75, "only curves passing through the data survive;\n"
             "black line = their average = the GP mean",
             fontsize=8.5, color="0.35", va="bottom")

for ax in axes:
    ax.set_xlabel("x"); ax.set_ylim(-3, 3)
    ax.grid(color="0.93", lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("output")
fig.suptitle("A Gaussian process is a probability distribution over whole functions",
             fontsize=12.5, y=1.02)
fig.savefig("figures/prior_posterior.png", bbox_inches="tight")
print("ok  figures/prior_posterior.png")
