# Datasheet — eight black-box function evaluations

Following the datasheet framework of Gebru et al., *Datasheets for Datasets* (2018).

## Motivation

**For what purpose was the dataset created?**

It was created as the working record of a Bayesian optimisation exercise set as the capstone for the Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence. Eight functions of unknown form, with 2 to 8 input dimensions, were made available through a submission portal. Each participant received a fixed initial design per function and could submit one further set of input coordinates per function per week for thirteen weeks, receiving the corresponding output in return.

The dataset therefore exists to support a single task: choosing the next query. It was not designed as a general-purpose benchmark, and it carries all the structure of a sequential decision process — later points are not independent of earlier ones, because each was chosen in response to what came before.

**Who created the dataset and on whose behalf?**

The functions, the initial designs and the evaluation service were provided by the course. The 104 weekly query points were chosen by me, and the corresponding outputs were returned by the course's evaluation service.

## Composition

**What do the instances represent?**

Each instance is one evaluation of one function: a point in the unit hypercube together with the scalar the function returned there. There are eight independent sets, one per function, and no relationship is claimed between them — f1 and f2 are both two-dimensional, but nothing suggests they are related.

**How many instances are there?**

| | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | Total |
|---|---|---|---|---|---|---|---|---|---|
| Dimensions | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 | — |
| Initial design | 10 | 10 | 15 | 30 | 20 | 20 | 30 | 40 | 175 |
| Weekly queries | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 104 |
| **Total** | 23 | 23 | 28 | 43 | 33 | 33 | 43 | 53 | **279** |

**Does the dataset contain all possible instances, or is it a sample?**

It is a sample, and a very small and deliberately non-random one. The domain is continuous, so the population is infinite; 53 points in eight dimensions is under two per axis if laid out as a grid, and splitting each axis merely in half gives 256 cells for 53 points.

The two halves of each file are samples of quite different kinds, and the distinction matters for anyone using the data:

- The **initial design** was supplied by the course and varies every input at once. It carries no bias towards any region of the domain, but it contains no two points differing in a single dimension — which is exactly why questions about individual dimensions could not be answered from it. How it was generated was never stated; see the sampling-strategy answer below.
- The **weekly queries** are adaptive. They concentrate heavily near whatever looked promising at the time, and several were chosen specifically to be uninformative about the optimum and informative about a structural question instead.

**What data does each instance consist of?**

Raw numeric values only. Columns `x1` … `xd` give the input coordinates, normalised to [0, 1] in every dimension; the column `output` gives the returned scalar. No derived features, no metadata, no timestamps.

**Is there a label or target?**

`output` is the quantity being maximised. There is no classification label.

**Is any information missing?**

Nothing is missing within the dataset itself. What is absent by design is the functions' identity and form — they are black boxes, and were never disclosed. Note also that the row ordering encodes the week (the initial design first, then weeks 1 to 13 in order) but the week number is not stored as a column.

**Are there any errors, sources of noise, or redundancies?**

The evaluation service appears to be deterministic: near-identical inputs return near-identical outputs, and in two cases outputs identical in every digit stored where an exact symmetry exists (f4 in x1/x4 returning 0.696302 twice, f5 in x2/x4 returning 7786.316251 twice). No point was ever submitted twice on purpose — a deliberate replicate would have spent one of the thirteen queries to confirm something nothing in the brief had put in doubt — so these two pairs are the only direct evidence of it in the dataset.

Apparent noise was diagnosed at one point in f2 and later attributed to model misspecification rather than to the data — see the README. Anyone modelling this data should treat it as noise-free.

**Does the dataset contain confidential or sensitive data?**

No. It contains no personal data, no identifying information, nothing about people at all.

## The eight functions

Each function stands for a different real-world optimisation problem, as set out in the problem brief. Every one is a maximisation.

| | Dims | What it represents |
|---|---|---|
| **f1** | 2 | Detecting contamination sources — a radiation field, say — across a two-dimensional area, where only proximity produces a non-zero reading. Both strong and weak sources must be found |
| **f2** | 2 | A black-box ML model returning a log-likelihood score from two inputs. The brief states the output is **noisy** and that local optima should be expected |
| **f3** | 3 | Drug discovery: three compound quantities per experiment, output the number of adverse reactions. Framed as maximising the negative, so zero side effects is the ceiling |
| **f4** | 4 | Warehouse product placement for a high-volume online retailer. Four hyperparameters of a fast ML approximation to an expensive biweekly calculation; the output is the gap to that baseline. The brief warns of many local optima |
| **f5** | 4 | The yield of a chemical process in a factory, from four inputs. The brief describes it as typically unimodal with a single peak |
| **f6** | 5 | A cake recipe across five ingredients, scored by an expert taster on flavour, consistency, calories, waste and cost — all penalties, so the score is negative by design |
| **f7** | 6 | Tuning six hyperparameters of an ML model — learning rate, regularisation strength, layer count and similar — maximising a performance score such as accuracy or F1 |
| **f8** | 8 | Eight-dimensional hyperparameter tuning. The brief's example includes two numerically-encoded categorical inputs, activation function and optimiser type |

## Collection process

**How was the data acquired?**

Directly observed, in the sense that each output was returned by the evaluation service in response to a submitted input. No inference or estimation is involved in any recorded value.

**What mechanisms or procedures were used?**

Coordinates were submitted through the course portal each week and the returned values were transcribed into the cumulative CSV files. The values in this repository have been checked row by row against the portal record.

**What was the sampling strategy?**

For the initial designs, unknown. They were supplied by the course without a stated generator, and the data does not support guessing one. Tested against the obvious candidates (`scripts/initial_design_check.py`), they are **not** a Sobol sequence and **not** a Latin hypercube: coordinates show no dyadic structure, each axis leaves about 36% of its bins empty where a Latin hypercube would leave none and a scrambled Sobol sequence about 19%, and L2-star discrepancy is worse than scrambled Sobol on all eight functions while sitting at an unremarkable percentile of uniform-random draws (2nd to 94th across the eight). The evidence is consistent with independent uniform sampling, and that is as far as it goes.

For the weekly queries, Bayesian optimisation: a Gaussian process fitted to all data available at the time, with candidates scored by expected improvement or upper confidence bound, and the final choice made by me — frequently overriding the model's recommendation. The README documents where and why.

The practical consequence is that **the weekly points are not an unbiased sample of the domain and must not be treated as one.** They are dense where the function appeared good and where a specific question needed answering, and empty elsewhere.

**Over what timeframe was the data collected?**

Thirteen consecutive weeks in 2026, one submission per function per week. The initial designs were provided at the outset.

**Were any ethical review processes conducted?**

Not applicable. The data involves no human or animal subjects.

## Preprocessing, cleaning and labelling

**Was any preprocessing done?**

The CSV files hold the raw submitted coordinates and returned outputs, unmodified.

Two transformations are applied at modelling time only, and are recorded per function in `config/params.csv` rather than baked into the data:

- **f1 and f5** are modelled on log₁₀ of the output, because they span roughly 140 and 5 orders of magnitude respectively. A GP fitted to raw f1 values is dominated entirely by a handful of large-magnitude points.
- **f1** additionally floors negative outputs at −300 in log space. Its outputs change sign, and neither log(|y|) nor a signed-log transform is safe under maximisation: both let a large-magnitude *negative* result appear as a strong one. Flooring keeps the spatial information about where negative regions are — which turned out to be the most important structure in f1 — without letting them compete for the optimum.

**Was the raw data saved?**

Yes. `data/function_1.csv` … `function_8.csv` *are* the raw data; the transforms are applied in the notebook at fit time and are reversible.

## Uses

**What has the dataset been used for?**

Only this project: fitting Gaussian process surrogates and selecting weekly queries.

**What other tasks could it be used for?**

It is a reasonable small-scale testbed for surrogate modelling, for acquisition-function comparison, and in particular for studying how ARD length-scale estimates behave under very sparse and adaptively-chosen data — which is the phenomenon this project spent most of its time on. f1 and f4 are useful stress cases for kernels that assume smoothness.

**Is there anything about the composition or collection that might affect future uses?**

Three cautions, in decreasing order of importance.

1. **The adaptive sampling is confounded with the outcome.** The points are where they are *because of* the values already observed. Any analysis that treats the rows as an i.i.d. sample — cross-validation included — will be optimistic, since neighbouring points were chosen to be neighbours. Leave-one-out Q² was used in this project with that caveat understood.

2. **Dimension-wise conclusions drawn from this data are unreliable unless the specific comparison is controlled.** This is the substantive finding of the whole project. Several dimensions have no pair of observations anywhere in the dataset differing in that dimension alone, and a fitted length scale for such a dimension reflects the absence of evidence, not its content. Anyone drawing relevance conclusions should check for controlled pairs first.

3. **The sample is tiny relative to the dimensionality.** f8 has 53 points in eight dimensions. Conclusions about its structure are weakly supported, and this repository says so where it applies.

**Are there tasks for which the dataset should not be used?**

It should not be used to characterise the underlying functions globally, nor to benchmark optimiser performance in any competitive sense, since the query points reflect one participant's decisions rather than an algorithm run to a protocol.

## Distribution

**How is the dataset distributed?**

As CSV files in this repository.

**Is it distributed under a licence, and are there restrictions?**

The submitted coordinates, the code and the analysis are my own work, released under the MIT licence with the repository.

The initial designs and the returned values originate with the Imperial College London course. Including them here is taken as intended rather than as an exception: the course requires the work to be reproducible, which is not possible without the underlying data, and it encourages participants to publish the result on a public GitHub repository as part of building a portfolio. Those two requirements together imply that the data is meant to travel with the write-up.

**Are there IP-based or other restrictions?**

The functions themselves are not disclosed here and are not mine to disclose. Only the coordinates queried and the values returned appear — 279 points, which reveal the shape of each function in the neighbourhoods explored and nothing about how it is computed.

## Maintenance

**Who maintains the dataset, and will it be updated?**

I do. It is complete and closed: the exercise ran for thirteen weeks and has finished, so no further evaluations will be added. Corrections to transcription errors would be made if any were found, as commits to this repository.

**Is there an erratum?**

None at present. One correction was made during preparation: the cumulative files were rebuilt and verified row by row against the portal record, which resolved a discrepancy between the week labels used in the contemporaneous working notes and the submission order recorded by the portal. The CSV ordering is authoritative.

**Will older versions continue to be supported?**

The repository's git history serves that purpose; no separate versioning is maintained.
