# Local versus global fitting

A retrospective investigation, carried out after the campaign closed. It started as a practical question about these eight functions and turned into a measurement of where the information about a length scale actually lives. Both parts are reproducible: `scripts/local_vs_global.py` and `scripts/length_scale_support.py`.

The short version: refitting on only the local cluster is worse almost everywhere, and the reason it is worse also says what the right alternative looks like.

---

## The question

By the later weeks the search had concentrated in one small region of each domain, and the model was still being refitted on everything — including points sampled a dozen length scales away in week 1 and never revisited. What were those points still contributing? It seems plausible that a model fitted only to the local cluster, even on fewer observations, would describe the region of interest better.

## The test

For each function, take the *m* observations nearest the final best point — the exploitation region — and leave-one-out predict each one three ways:

- **global** — fit on all the remaining data;
- **local** — fit on the *k* ≈ 10 nearest, hyperparameters refitted on that subset;
- **hybrid** — hyperparameters from the full dataset, but conditioned only on the *k* nearest.

RMSE over the held-out points, so lower is better:

| | n | global | local | hybrid | best |
|---|---|---|---|---|---|
| f1 | 23 | 88.19 | 126.6 | **85.55** | hybrid |
| f2 | 23 | **0.0482** | 0.0673 | 0.0489 | global |
| f3 | 28 | **0.0058** | 0.0205 | 0.0086 | global |
| f4 | 43 | **0.286** | 0.353 | 0.326 | global |
| f5 | 33 | **0.147** | 0.280 | 0.218 | global |
| f6 | 33 | 0.0802 | **0.0779** | 0.1172 | local |
| f7 | 43 | **0.229** | 0.412 | 0.332 | global |
| f8 | 53 | **0.0257** | 0.0385 | 0.0425 | global |

Global wins six of eight, and pure local fitting is worse than global on seven of the eight — by around a factor of two on f5 and f7, and by three and a half on f3. The hypothesis is largely refuted, and the reason is instructive.

Two caveats on the table. It uses 10 restarts rather than the weekly 50, and leave-one-out on adaptively-sampled data is optimistic in absolute terms. Both apply equally to all three columns, so the ranking holds. The fitting seed is fixed, so the figures above reproduce exactly.

## Why: distant points play two separate roles

The argument for dropping them confuses the two.

**In the prediction they already contribute almost nothing.** The posterior mean is a distance-weighted sum of the observations, and a point several length scales away carries negligible weight — that is what a length scale means. Dropping it barely changes the prediction, because it was never really being used.

**In the hyperparameter estimation they do most of the work** — but not because more data is always better. That turns out to be the more interesting half of the answer, and it needs its own section.

## Where length-scale information lives

The information about a length scale lives in pairs of points separated by roughly that length scale.

A pair much closer than ℓ is almost perfectly correlated whatever ℓ is; a pair much further apart is uncorrelated whatever ℓ is. Neither responds when ℓ changes, so neither constrains it. Only pairs at intermediate separation are sensitive. Differentiating the Matérn 2.5 correlation with respect to ℓ puts the peak at a separation of about **1.2 ℓ** (for RBF, 1.41 ℓ), falling away in both directions:

| separation | 0.25 ℓ | 0.5 ℓ | 1 ℓ | 1.2 ℓ | 2 ℓ | 3 ℓ | 5 ℓ |
|---|---|---|---|---|---|---|---|
| relative information | 0.15 | 0.48 | 0.95 | 1.00 | 0.69 | 0.23 | 0.01 |

That cuts both ways, which is why subsetting fails for the wrong reason as well as the right one.

Ten tightly clustered observations cannot say how far you must travel before the function decorrelates, because none of them travel far enough. The estimate collapses onto the cluster's own spacing instead. This was visible during the campaign: f4's week-7 local fit returned a length scale for x2 of 0.0065, which is not a measurement of anything.

But points on the far side of the domain are nearly as mute, for the opposite reason. What an estimate needs is not the whole domain; it is a neighbourhood a few length scales wide.

Fitting on nested neighbourhoods of the final best point shows roughly where that threshold sits. Radius is in length-scale units, and the figure is the mean absolute log ratio of the fitted length scales against the full-data fit:

| radius | f4 | f6 | f7 |
|---|---|---|---|
| 1 ℓ | 0.67 | 3.10 | 1.17 |
| 1.5 ℓ | 0.24 | 1.99 | 0.32 |
| 2 ℓ | 0.17 | 1.35 | 0.25 |
| 4 ℓ | — | 0.13 | — |

At one length scale the fit is unusable: f7's first two length scales run to the ceiling of their allowed range, f4's come back at about half their true size. By two length scales most are within 20–30%. f6 needs four, which is consistent with it having one genuinely long length scale. The dashes are where the neighbourhood had already swallowed the entire dataset, so these runs cannot demonstrate the upper half of the story — that part rests on the analytic table above.

So subsetting the training data discards exactly what the distant points were providing, and saves nothing that was being paid for. That the hybrid column beats pure local in six of eight makes the same point from the other side: where localisation helps at all, it helps by restricting the *data used for prediction*, not by re-estimating the *hyperparameters* locally.

## The instinct is nevertheless right about the underlying defect

Two things are true at once here, and they are easy to mistake for a contradiction.

*Estimating* a length scale needs points spread over a few multiples of it — more than a tight cluster, less than the whole domain. But a single length scale per input then *claims* that one decorrelation distance describes the entire domain. That is the stationarity assumption, and for several of these functions it is plainly false.

The first is a statement about where the evidence lives; the second about what the model does with it. So a global fit is not wrong to use data from well beyond the region of interest — it needs some of it. It is wrong in reporting a single compromise value averaged across regions of genuinely different smoothness, which need not suit the region that matters.

That is why pure local fitting fails. It attacks a real problem by discarding the evidence needed to estimate anything at all. **The defect is in the model, not in the amount of data fed to it.**

f1 violates stationarity most grossly — a razor-thin channel in one small region and a flat −300 floor everywhere else — and f1 is the one function where restricting the data genuinely helps. That was resolved at the time by the more radical route of abandoning the model altogether and stepping empirically ([f1](docs/f1.md)); this measurement is the retrospective justification for it.

## The proper fixes

Neither is subsetting. Two stand out, and neither was used during the campaign.

**Trust-region Bayesian optimisation** (TuRBO and relatives) restricts the region that is *searched* while continuing to fit hyperparameters on all the data. It delivers the localisation without discarding the evidence that identifies the length scales, and it would have suited the later weeks of every function here. The measurements above also say what size such a region should be: scaled to a few length scales per dimension, it is both local enough to be worth having and wide enough to estimate the length scales it is scaled by.

**A non-stationary kernel, or input warping**, attacks the assumption directly by letting smoothness vary across the domain. This is the principled answer for f1 and f4, the two functions whose defining feature was a region far sharper than the rest of the space.

---

## Reproducing

```bash
python scripts/local_vs_global.py          # the RMSE table
python scripts/length_scale_support.py     # both length-scale measurements
```

Fitted length scales vary slightly between runs, since the restart seeds are not fixed, but not enough to change any of the conclusions above.
