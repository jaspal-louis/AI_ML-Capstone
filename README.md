# Bayesian Optimisation of Eight Black-Box Functions

A thirteen-week Bayesian optimisation challenge on eight hidden functions of 2 to 8 dimensions. One query per function per week, without any information on gradients, functional form, or feedback on proximity to the goal — 104 evaluations in total across the eight.

This repository contains the code, the complete dataset, the week-13 model state and the write-up.

---

## Non-technical explanation

Imagine eight sealed machines. Each machine has a row of dials — ranging from two on the first one up to eight on the last — and a single output number on a display. You want the dial settings that make that output as large as possible.

You cannot open the machine or see inside it (black-box). You may set the dials **once a week** and read the output display. Thirteen weeks, thirteen readings.

Trial and error is hopeless: eight dials with ten settings each is a hundred million combinations. So instead of guessing, you build a statistical model of what the machine probably does — a multi-dimensional surface fitted through every reading so far, which also reports how confident it is anywhere you have not tried. Each week you ask it the same question: which single setting is most worth spending a reading on?

That is Bayesian optimisation in principle. In practice it ran from the statistical theory, through a Python implementation, to a set of purpose-built diagnostics — and then thirteen rounds of the same cycle: refit the model, work out what it was actually claiming, spend the week's single query, account for what came back. The model was confidently wrong more than once, in ways that were not obvious at the time, and the write-up records how those errors were caught and what was learnt in the process.

---

## Results

| | Dims | Initial best | Final best | Improvement | Found | Character |
|---|---|---|---|---|---|---|
| [**f1**](docs/f1.md) | 2 | 7.7 × 10⁻¹⁶ | **1.73 × 10⁻⁵** | +10.4 orders of magnitude | wk 12 | Sign-changing; spans ~140 orders of magnitude |
| [**f2**](docs/f2.md) | 2 | 0.611205 | **0.658525** | +7.7% | wk 5 | Sharp ridge, near-flat along it |
| [**f3**](docs/f3.md) | 3 | −0.034835 | **−0.000604** | 98% of the gap to zero | wk 11 | Smooth; one dimension untested for 11 weeks |
| [**f4**](docs/f4.md) | 4 | −4.025540 | **0.741945** | negative → positive | wk 10 | Sharp isolated peak; exact x1/x4 symmetry |
| [**f5**](docs/f5.md) | 4 | 1088.86 | **8662.41** | 8.0× | wk 7 | Monotone to the corner; exact x2/x4 symmetry |
| [**f6**](docs/f6.md) | 5 | −0.714265 | **−0.224110** | 69% of the gap to zero | wk 10 | Genuine interior optimum in 5D |
| [**f7**](docs/f7.md) | 6 | 1.364968 | **2.07553** | +52% | wk 13 | Best-behaved; near-uniform curvature |
| [**f8**](docs/f8.md) | 8 | 9.598482 | **9.956340** | +3.7% | wk 12 | Flat optimum; 8D on 13 queries |

Each function has its own page with the final surface, the week-by-week trend, and an account of the approach and the problems encountered.

### How this compared

The exercise ran across a cohort of 34 participants, with each function scored separately. Rankings were published only after the project completed, and the output values behind them were never published at all. Everything below that compares actual results is therefore post-hoc, reconstructed from repositories the cohort published after the close. Averaging the eight per-function placings, I finished with an overall position of **8th of 34** — inside the top quartile. The two best results by function were **f4**, which placed first, and **f3**, third.

f4 is one worth dwelling on, because it is the function where the model was overridden most often and most deliberately. Despite seemingly fitting well, the global model smoothed over a narrow high-performing region, which I identified and addressed by using local analytics to override the model recommendations.

The weaker results, on **f7** and **f2**, both come down to a region that was never probed rather than to functions that were especially hard. f7 is the instructive one, and the finding is a paradox: it was the best-behaved of the eight by every internal measure, and that is close to the reason it went wrong. A surface that never misbehaves never gives you a reason to doubt the model fitted to it. Mine was steadily confident about the two inputs I had effectively frozen, and because nothing ever broke, nothing ever forced the question. My three best placings all came from functions that misbehaved early and were interrogated hard as a result. Peer outcomes show a materially better region existed, and refitting with the extra observations the cohort published afterwards says my model had the slope on one of those inputs backwards for eleven weeks. On f2 the ridge found in week 5 absorbed the remaining budget while a second region went unexplored. Both are described in the individual function analyses.

**A note on the improvement column.** Each figure is expressed in whatever unit suits its own function, because no single measure survives contact with all eight: f1's outputs span about 140 orders of magnitude in absolute value and change sign, f5's span nearly five, f8's top five observations differ by 0.0069. A common "% improvement" would make f1 unreadable and f4 undefined.

For a comparable reading, the table below expresses each gain as a multiple of what the initial design itself achieved — that is, the improvement over the initial best, divided by the distance from the initial best down to the initial design's median. It answers "how much again did thirteen queries add, on top of what the 10–40 supplied starting points already found?" (in the modelling scale, so log₁₀ for f1 and f5):

| f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---|---|---|---|---|---|---|
| 0.16 | 0.12 | 0.48 | 0.40 | 0.73 | 0.67 | 0.55 | 0.21 |

On that basis f5, f6 and f7 gained most relative to their starting point, and f2 and f8 least — f2 because it was essentially solved by week 5, f8 because eight dimensions on thirteen queries leaves very little room. f1's 0.16 understates the achievement badly, which is a fair illustration of why the column above is in natural units: its initial design median sits on the −300 floor, so the denominator is enormous.

---

## What I learned

### Overarching lessons

**Learning Bayesian optimisation in practice.** I came in with a basic overall understanding of Bayesian optimisation and left with a detailed understanding of what the components actually do — what a kernel assumes, what a length scale means, how hyperparameters are chosen by maximising the log marginal likelihood, and why that criterion charges for flexibility rather than rewarding it. Just as useful was learning where the standard diagnostics stop being informative. A model fitted with near-zero noise reproduces its own observations almost exactly, so a high R² is largely a given rather than evidence: f3 returns an R² of 0.99996 alongside a Q² — the same measure on points held back from the fit — of 0.26, and f1's Q² is **−0.25**, meaning its predictions were worse than simply guessing the average. [`model-background.md`](model-background.md) sets this out in detail.

**Optimisation strategy with scarce resources.** The binding constraint was the budget: thirteen queries per function, across eight functions running at once, with no way to buy more information than one point a week. Where the project went well, the budget was allocated deliberately — queries spent on establishing a key fact rather than on chasing a gain, with the evaluation goal defined before the result was known. Where it went badly, typically the budget was misallocated: six weeks refining a point already known to be good, or a dimension left untested for thirteen weeks because nothing forced the question. The single decision with the largest effect on the outcome was not which kernel to use but what each week's query was *aiming to achieve*.

**Visualisation.** I developed an approach that plots four dimensions at once as a grid of heatmaps — two axes within each panel, two more across the rows and columns ([example](runs/week_13/slice_f7_mean_w13.png)) — and it became the main instrument for reading these surfaces. It made the structure legible in a way the fitted numbers alone did not, and it showed directly where the data was thin and a query was worth spending. Used alongside the model's own output it also settled several disagreements in both directions: sometimes the picture overturned a numerical proposal, sometimes the reverse. What I took from that is that neither view is always right, and that the disagreements were worth more than either view on its own — they were consistently where the real structure turned out to be.

**Leveraging existing risk expertise.** More transferred from my own field than I expected. Understanding how the output responds to each input was central to the whole exercise, which is directly analogous to risk analytics: measuring sensitivity to one factor while holding the others fixed (delta), how that sensitivity itself changes (gamma), and how two factors interact (cross-gamma). Similarly attributing a change in output to individual inputs is analogous to P&L attribution.

The habit that mattered most, though, was model-risk discipline. The task throughout was to understand the model well enough to tell a conclusion it could support from one it could not, and to act on that distinction even when the model was recommending otherwise. That distinction is what most of the lessons below come back to.

**Where the difficulty actually lay.** The method worked: all eight functions improved on their starting designs, several by a wide margin, and the set finished inside the top quartile of the cohort. What is more striking in hindsight is where the failures came from. Not one of them was a coding or fitting error. Every expensive mistake in thirteen weeks was an issue of *interpretation* — reading a parameter stuck at its bound as a finding, or a prediction in an unsampled region as evidence. The lessons below are almost entirely about that.

---

### Specific lessons learnt

Nine things I would carry into the next problem of this kind. Each is stated generally; the bracketed links go to the function page where it played out, with the evidence.

**1. A fitted parameter sitting on its bound is not an estimate.** When the fitting routine pushes a length scale setting to the edge of its allowed range and stops there, the edge is not a measurement. It means *at least this much, and the data cannot say more* — i.e. an absence of evidence, not evidence of absence. Reading it the other way was the most expensive error of the project. ([f2](docs/f2.md), [f3](docs/f3.md), [f7](docs/f7.md), [f8](docs/f8.md))

**2. A prediction where there is no data is an assumption, not evidence.** The model answers every question put to it, and answers just as smoothly in regions it has never sampled as in regions it knows well. The two answers look identical and are not. ([f5](docs/f5.md), [f8](docs/f8.md))

**3. To learn what one input does, change only that input.** Points that differ in several inputs at once cannot say which difference mattered. A starting design that varies every input at once contains no such controlled pair anywhere, so that evidence has to be bought deliberately with a query. ([f5](docs/f5.md), [f4](docs/f4.md))

**4. Direction and distance are separate questions.** When a move of an input parameter fails to improve the output, the instinct is to abandon the direction. More often the direction was right and the step was simply too long — in which case the fix is to shorten it, not to turn around. ([f6](docs/f6.md))

**5. Surface sharpness and noise are assumptions, and the model cannot reliably distinguish the two.** Given two nearby points with very different outputs, a model can conclude the function turns sharply or that the readings are noisy. Noise is usually the easier explanation, so it will often choose that even where the function is known to be deterministic — and a kernel assuming more smoothness than the function has will quietly average away the feature that matters. Neither is a setting to leave alone: the fitted noise is worth reading every week, because a large value is the model reporting that its own assumptions cannot represent what it is being shown. ([f1](docs/f1.md), [f4](docs/f4.md))

**6. Uncertainty is not opportunity.** The standard rules for choosing where to look next reward uncertainty, and uncertainty is largest simply where there is no data — so a point can score well purely for being far from everything. Before accepting one, check what the model actually expects to find there. ([f2](docs/f2.md), [f7](docs/f7.md))

**7. Look at the picture and the numbers, and treat disagreement as information.** Plots caught errors the statistics missed, and the statistics caught errors the plots missed. Neither view is always right, and where they disagreed was almost always where the real structure was. ([f4](docs/f4.md), [f6](docs/f6.md))

**8. Know when to stop using the model.** Occasionally the right response is not to re-tune it but to set it aside and work directly from the observations. The function with the largest relative gain of the eight was optimised that way, with the model forecasting failure at every step. ([f1](docs/f1.md))

**9. Spend residual budget on information — but choose what to measure.** Only the best result counts, so once a strong one is recorded it cannot be lost, which makes a deliberate probe cheap and often the best value available. The harder question is what to point it at: six queries can be spent on exactly the right *kind* of investigation, but on entirely the wrong target. ([f3](docs/f3.md), [f7](docs/f7.md), [f5](docs/f5.md))

---

## Data

The dataset is 104 weekly evaluations across the eight functions, plus the initial designs supplied at the start of the project.

| | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---|---|---|---|---|---|---|---|
| Dimensions | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 |
| Initial design | 10 | 10 | 15 | 30 | 20 | 20 | 30 | 40 |
| Weekly queries | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 |
| **Total** | 23 | 23 | 28 | 43 | 33 | 33 | 43 | 53 |

`data/function_1.csv` … `function_8.csv` hold the complete record: columns `x1`…`xd` and `output`, with the initial design first and the thirteen weekly submissions appended in order. Inputs are normalised to [0, 1] in every dimension.

The initial designs were supplied by the course, which did not say how they were generated. Testing them against the obvious candidates rules out a Sobol sequence and a Latin hypercube; the evidence is consistent with independent uniform sampling ([`scripts/initial_design_check.py`](scripts/initial_design_check.py)). What matters for this project is a property they have either way: every input varies at once, so no two points differ in one dimension alone. That is precisely why so many of the relevance questions above could only be settled by spending a weekly query on a controlled probe.

Outputs are untransformed apart from two cases. **f1** and **f5** are modelled on log₁₀, because their outputs span roughly 140 and 5 orders of magnitude respectively. f1 additionally takes negative outputs to a floor of −300 in log space: a signed-log or log|y| transform would let a large-magnitude negative appear as a strong result during maximisation, while flooring retains the spatial information about *where* the negatives are without promoting them. These choices are recorded per function in `config/params.csv`.

### Provenance

The functions themselves are not public and are not reproduced here — only the coordinates submitted and the values returned. The initial designs were supplied as part of the course exercise.

The code, the written analysis, the figures and the coordinates I chose are released under the MIT licence. The initial design points and the returned output values originate with the course and are not mine to licence; they are included because the work cannot be reproduced without them. [`NOTICE`](NOTICE) states the split precisely, and [`datasheet.md`](datasheet.md) gives the full description of collection, composition and permitted use.

---

## Model

A Gaussian process regressor (`sklearn.gaussian_process.GaussianProcessRegressor`), refitted from scratch each week on the full cumulative dataset for that function.

**Kernel.** A constant amplitude term multiplying either a Matérn kernel with ν = 2.5 or a squared-exponential (RBF) kernel, in both cases with **ARD** — a separate length scale per input dimension — plus a white-noise term:

```
C(1.0) × Matern(length_scale=[1.0]*d, nu=2.5)  +  WhiteKernel()
C(1.0) × RBF(length_scale=[1.0]*d)             +  WhiteKernel()
```

Matérn 2.5 is the primary choice. RBF assumes the function is infinitely differentiable, which is a strong claim about an unknown black box and a poor one for f1 and f4; Matérn 2.5 assumes only twice-differentiable and handles sharper structure. Both were run every week, and the comparison was itself informative — persistent disagreement between them was a reliable signal that the data did not constrain the fit.

**Why a GP rather than anything else.** The requirement is not just a prediction but a *calibrated* prediction: the acquisition functions need σ as much as μ, and they need it to mean something. A GP gives a closed-form posterior over functions, so σ is not an add-on. An SVR was fitted alongside for f4 as a cross-check and reproduced the GP's error, which was informative in a different way — it confirmed the problem was the smoothness assumption both models share, not one model's optimiser.

**Acquisition.** Expected improvement (EI) as the primary, upper confidence bound (UCB) as a contrast. The acquisition surface is maximised in two stages: a scrambled Sobol screen over the whole domain (5,000 candidates in 2–3 dimensions, 10,000 in 4–5, 20,000 in 6 or more), then L-BFGS-B refinement started from the ten best candidates, keeping whichever finish is highest. The acquisition surface has its own local optima, so a single gradient run from one starting point is unreliable; multi-starting it from the best screened points is the same device as the fifty restarts used on the kernel fit, applied to a different optimisation.

**Noise.** Two different things carry that name. A fixed jitter on the covariance diagonal (`alpha`, tightened to 1 × 10⁻⁸ from week 12) is numerical housekeeping. A `WhiteKernel` whose level is *fitted* is the term that decides whether the model interpolates, and it was left free deliberately — partly because the brief describes f2 as noisy, but mainly because a GP's noise term absorbs whatever the kernel cannot represent. Its fitted value is model discrepancy rather than randomness, which makes it a diagnostic rather than a setting:

| | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---|---|---|---|---|---|---|---|
| fitted noise | **0.1** | 0.023 | 5.8 × 10⁻⁴ | 3.9 × 10⁻⁵ | 10⁻¹⁰ | 8.4 × 10⁻³ | 10⁻¹⁰ | 10⁻¹⁰ |

Only f5, f7 and f8 interpolate. f1's value sits on its *upper* bound, which by lesson 1 is not an estimate of anything — it is the model reporting that it cannot represent that function, and it reported it in every weekly fit from the start. Nobody read it, and f1's surface stayed in use until week 4, when it predicted a value fourteen orders of magnitude above anything ever observed.

[`model-background.md`](model-background.md) sets out why the term belongs there, and shows that refitting with the noise pinned and comparing the two surfaces flags all four functions where smoothing across sharp structure caused real errors while staying silent on the other four. It also corrects the record on week 12: the intended switch to a near-interpolating fit changed the jitter, not the free noise term, so it never took effect on five of the eight.

---

## Hyperparameter optimisation

There are two optimisations here. The **outer** one is the task itself — find the inputs that maximise the hidden function, one query a week. The **inner** one runs every time the model is refitted: given the data so far, what shape should I assume this function has? This section is about the inner one.

The model has one **length scale** per input dimension, plus an amplitude and a noise level. A length scale is the distance you must move along that input before the output stops being predictable from where you started — so it also answers which inputs matter, since a very long one means that input barely changes anything. These are not set by hand. They are fitted by maximising the **log marginal likelihood**, which scores how probable the observed data is under a given set of assumptions, and which penalises over-flexible models automatically.

**[`model-background.md`](model-background.md) sets this out in detail** — Gaussian processes, the model's key assumptions, length scales, the likelihood, the optimiser and restarts, the bounds, and the Q² diagnostic. What follows here is only what was chosen and why.

### What was used

| Setting | Value | Why |
|---|---|---|
| Kernel | Matérn ν = 2.5 (primary), RBF (contrast) | Matérn 2.5 assumes the function is twice differentiable; RBF assumes infinitely differentiable, which is too strong a claim for an unknown box and wrong for f1 and f4 |
| ARD | on | one length scale per dimension, so relevance falls out of the fit rather than being assumed |
| `n_restarts_optimizer` | 50 | the fitting routine is a local optimiser on a surface with several peaks; see below |
| Length-scale bounds | per function | f1 needs a lower bound of 0.05 or the fit collapses onto its own points; f8 an upper bound of 10 given its eight dimensions |
| Noise | `alpha` = 1 × 10⁻⁸ from week 12 (diagonal jitter), plus a `WhiteKernel` fitted freely in [10⁻¹⁰, 10⁻¹] | the jitter is numerical housekeeping; the free term is what lets the model register structure the kernel cannot express, and its fitted value is a diagnostic rather than a setting — see above |
| Diagnostics | log marginal likelihood, Q² against R², interpolation residual | R² is near 1.000 wherever the fit interpolates, which is most functions, and says very little; Q² measures whether the fit predicts a point it has not seen |

### The two settings that mattered

**Fifty restarts, not ten.** The likelihood surface has more than one peak, and the peaks are competing stories about the same data — *this dimension matters and the readings are noisy* versus *this dimension is irrelevant and the readings are clean*. Which one a single fit finds depends on where it happens to start. In **f7**, x6's length scale flipped between about 0.12 and the top of its range across consecutive weeks on essentially unchanged data. Restarts run the fit repeatedly from random starting points and keep the best. Raising the count from 10 to 50 in week 5 removed the instability for the rest of the project.

**A length scale sitting on its bound is not an estimate.** When the data cannot constrain a dimension, the fitted value walks to the edge of its allowed range and stops there. What it means is "at least this large, and the data cannot say more" — which is *not* the same as "this dimension does not matter". Reading it as the latter cost multiple weeks on f2, f3, f7 and f8, and it is the first lesson in the list above.

### Choosing between runs each week

Four runs were fitted per function per week — `rbf_ard_ei`, `matern_ard_ei`, `matern_ard_ucb`, `rbf_ard_ucb` — and compared on their fit statistic, on whether the predicted value was plausible against the range actually observed, and on whether the four agreed with each other.

Selection was not automatic. A run with the better fit statistic was overridden where direct evidence from the diagnostic grids contradicted it (f8, week 2); all four were overridden together where the predicted surface was not credible (f1, from week 5). Persistent disagreement between the Matérn and RBF fits was itself treated as a signal, usually that the data did not constrain the fit rather than that one kernel was right.

---

## What I would do differently

Three changes are summarised below, with references to further detailed descriptions.

- **Run the nine lessons above as a checklist from week 1, rather than discovering them.** Every one of them was learned by paying for it. The cheapest to act on early and the most expensive to retrofit is lesson 3: spend a query establishing a controlled pair in each input before the search narrows, because once it has narrowed there is no budget left for it. Two functions carried an untested input all the way to week 13.

- **Read the fitted noise every week, and fit both ways.** Leaving the noise term free was the right call on an unknown function, but its value was never looked at — and on f1 it sat pinned at its ceiling from the start, which is the model saying it cannot represent the function at all. Running a free fit alongside a genuinely pinned one turns the gap between them into a number: on this data it flags all four functions where smoothing across sharp structure caused real errors, and stays silent on the other four. The measurement is in [`model-background.md`](model-background.md).

- **Use a method that matches the problem.** Everything here was a textbook GP with a stationary kernel, refitted globally each week. Two better-suited families went unused: **trust-region Bayesian optimisation** (TuRBO and relatives), which concentrates the *search* without discarding the data that identifies the length scales, and **non-stationary kernels or input warping**, which let smoothness vary across the domain instead of assuming one value fits all of it. The second is the principled answer for f1 and f4, whose defining feature was a region far sharper than the rest of the space.

  The retrospective test behind that bullet is in [`local-vs-global-fitting.md`](local-vs-global-fitting.md), along with the measurement that sets a trust region's size: length-scale information lives in pairs of points separated by roughly one length scale, so a region a few length scales wide is both local enough to be useful and wide enough to estimate the length scales it is scaled by. It also records one hypothesis I tested and had refuted — that refitting on only the local cluster would describe the exploitation region better. It does not; it is worse on six of the eight.

---

## Repository layout

```
data/            function_1..8.csv — complete 13-week dataset
config/          params.csv — the week-13 configuration for all 8 functions
notebooks/       bayesian_opt.ipynb — the working notebook, runnable end to end
scripts/         make_figures.py — per-function surface and trend figures
                 make_explainer_figures.py — figures for model-background.md
                 local_vs_global.py — the local-vs-global fitting test
                 length_scale_support.py — where length-scale information lives
                 initial_design_check.py — what kind of design the seed points are
                 make_quality_figure.py — Q2/R2 and the model card figure
runs/week_13/    week-13 model state: acquisition surfaces, slice grids,
                 ARD length scales, recommendations and results
figures/         the surface and trend figures used in docs/
docs/            f1.md … f8.md — function-by-function detail
datasheet.md     dataset documentation
model_card.md    model documentation
model-background.md
                 how the surrogate model works, from first principles
local-vs-global-fitting.md
                 retrospective: local vs global fitting, and where
                 length-scale information comes from
requirements.txt the verified environment — pinned versions
LICENSE          MIT
NOTICE           what the MIT licence covers, and what it cannot
```

## Reproducing this

```bash
pip install -r requirements.txt
jupyter notebook notebooks/bayesian_opt.ipynb      # Restart & Run All
python scripts/make_figures.py 1 2 3 4 5 6 7 8     # from the repository root
python scripts/make_explainer_figures.py           # figures for model-background.md
python scripts/make_quality_figure.py              # Q2/R2 table and the model card figure
python scripts/local_vs_global.py                  # the local-vs-global fitting test
python scripts/initial_design_check.py             # what the seed designs are
python scripts/length_scale_support.py             # where length-scale information lives
```

The notebook locates the repository root by searching upward for the `data/` directory, so it runs from wherever it is opened. `CURRENT_WEEK` is set to 13 and `config/params.csv` holds week 13 only; results are written to `runs/week_13/`.

Every script fixes its random seed, so with the pinned versions in [`requirements.txt`](requirements.txt) the tables and figures in this repository reproduce exactly. The notebook does not fix its seed, so its weekly fits move by around 1 × 10⁻⁴ between runs — far too little to change any recommendation or conclusion.

The whole set was last verified end to end in a clean virtual environment on Python 3.11. The pins are not brittle: the same analysis was also run on scikit-learn 1.8.0 against the pinned 1.9.1, and the fitted values agreed to about 1 × 10⁻⁴.

---

## Use of AI tools and other sources

Anthropic's Claude was used to support this project to write the notebook and the scripts in `scripts/`, draft initial documentation and produce additional analysis. Decisions regarding weekly submissions, review and override of the model outcome were fully mine, as were updates to and final wording of all documentation; additionally feedback to correct mistakes / make necessary enhancements to code base.

The post-hoc comparison in [How this compared](#how-this-compared) draws on repositories other participants published after the close. No participant is named and no peer output value appears here.
