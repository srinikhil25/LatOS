# Stage 6 — Reliability-gated closed loop

**Status:** ✅ complete (Phase 1 of the ionic-thermoelectric campaign work)
**Sub-stages covered:** thermovoltage analyzer, per-datapoint variance, stopping verdict, workbook ingestion, campaign rehearsal, one-command cycle
**Commits:** `9f9368a`, `5250406`, `4f02e21`, `1d2f9de`, `765476f`, `6d98f8e`, `922b88f`
**Date range:** 2026-08-19 – 2026-08-21

> **Note on numbering.** `THESIS_OUTLINE.md` reserved Stage 6 for "Bayesian
> optimization of synthesis parameters", which is what this delivers. Stages 4
> and 5 (cross-modal correlation, vision-language inference) have partial
> implementations in `reporting/correlation.py` and elsewhere but no stage doc;
> `RESULTS_LOG.md` and `BENCHMARKS.json` likewise stop at Stage 2. That gap is
> real and is recorded here rather than papered over.

## 1. Goal

Make the loop closeable for a prospective experiment: read what the bench
recorded, judge how well each sample is known, decide whether another experiment
is worth running, and freeze the prediction before the sample exists — in one
command, so it happens consistently rather than when someone remembers.

## 2. Motivation

Two problems, one measured and one structural.

**The reliability claim was one bit per datapoint.** `optimize()` accepted a
single campaign-wide `measured_noise` scalar and a boolean `unreliable` mask. So
"per-datapoint reliability travels into the optimizer", the project's central
claim, was implemented as *flagged or not*. Meanwhile the analysis layer was
already computing better numbers and discarding them: a fitted slope carries a
standard error, a derived quantity carries propagated uncertainty, and the XPS
check produces a spread across background models.

**The tool would not say when to stop.** An audit had recorded that convergence
never fires below ten points. Probing it showed something worse: on a
single-peak objective sampled at six points *including the peak*, the engine
reported `P(within ε) = 0.992`, signal exhausted — then `converged=False` and a
recommendation at the far edge of the search space. Every number was right and
the advice was wrong. For a campaign budgeted at ten experiments that spends the
back half walking away from an answer already found.

## 3. Design decisions

- **Decision:** the Seebeck coefficient is a *fitted slope* across three or more
  ΔT values, not a single reading divided by a single ΔT.
  - Alternatives considered: one reading per sample, as the technique is usually
    reported.
  - Why this won: in a thermodiffusive cell the measured voltage is
    `ΔV = S·ΔT + ΔV_electrode`, and one point cannot separate the two. Fitting
    exposes the electrode term as the intercept and yields the standard error
    that Stage 6's per-point variance depends on. It costs three measurements
    instead of one.

- **Decision:** per-observation variance multiplies with the physics flag rather
  than replacing it.
  - Alternatives considered: let a supplied σ override the flag.
  - Why this won: they are different claims. A σ says how repeatable a
    measurement was; a flag says the value contradicts something that must hold
    regardless. A precisely-measured impossible number deserves both penalties —
    the precision is what makes it worth distrusting.

- **Decision:** report the disagreement between stopping signals rather than
  resolving it silently.
  - Alternatives considered: let `epsilon_delta_met` override the
    data-sufficiency gate, or leave `converged` as the only answer.
  - Why this won: the probabilistic regret bound and the sufficiency grade are
    independent lines of evidence, and when they disagree neither "stop" nor
    "keep exploring" is honest. `CONFIRM` names the third option — repeat the
    incumbent, which costs one experiment where continued exploration costs
    several.

- **Decision:** the workbook schema is declared once and read by both the writer
  and the parser.
  - Alternatives considered: keep the template generator as a standalone script.
  - Why this won: it was standalone, in an untracked directory, with its own
    copy of every column name. A rename would have produced empty measurements
    rather than an error.

- **Decision:** a rehearsal reports per-shape results but computes its headline
  only from shapes whose optimum is interior.
  - Alternatives considered: average over all shapes.
  - Why this won: when the optimum sits at an endpoint the seed design finds it
    before any model runs, so those shapes cannot separate one strategy from
    another. Averaging them in is how 54.5 % of an early Starrydata benchmark
    came to mean nothing.

## 4. Methods / algorithms

- **Ordinary least squares with parameter standard errors** — `ΔV = S·ΔT + b`
  fitted in closed form. With `n` points and two parameters the residual
  variance carries `n − 2` degrees of freedom; a two-point fit therefore has
  none, and its standard errors are reported as undetermined rather than zero.
  [\[draper1998\]](../references.md#draper1998)

```math
\mathrm{se}(S) = \sqrt{\frac{\sum r_i^2}{(n-2)\,S_{xx}}},
\qquad
S_{xx} = \sum (\Delta T_i - \overline{\Delta T})^2
```

- **Extra-sum-of-squares F test for curvature** — R² cannot detect curvature on
  a short series: a deliberately quadratic five-point series scores 0.963.
  Comparing a quadratic fit against the linear one takes its noise estimate from
  the quadratic residual, where the curvature has been absorbed rather than
  counted as noise. [\[draper1998\]](../references.md#draper1998)

- **Heteroscedastic Gaussian-process regression** — per-observation variances
  enter as the diagonal `alpha` term, so each point contributes
  `(σᵢ / std(y))²`. In log space the conversion is per observation, since an
  absolute σ is a different fraction at each magnitude.
  [\[rasmussen2006\]](../references.md#rasmussen2006)

- **Probabilistic regret bound** — `P(the incumbent is within ε of the optimum)`
  by Monte Carlo over joint posterior draws. Already present; this stage makes
  it the headline of a recommendation rather than a buried field.
  [\[wilson2024\]](../references.md#wilson2024)

- **Single-parabolic-band thermodiffusion relations** — the ionic Seebeck
  coefficient of a dual-ion electrolyte as a weighted difference of ion heats of
  transport, `S = (w₊Q₊* − w₋Q₋*)/(eT)`. Used as a diagnostic against the
  measured curve, never as a prior. [\[sun2023\]](../references.md#sun2023)

## 5. Implementation summary

| File | What it owns |
|---|---|
| `analysis/thermovoltage/slope.py` | `ΔV`-vs-`ΔT` fit; Seebeck slope, electrode intercept, curvature test |
| `ingestion/ite_workbook_template.py` | the workbook schema and the writer for it |
| `ingestion/parsers/ite_workbook.py` | one `ParsedData` per sample, carrying its ΔT series |
| `optimization/engine.py` | `point_noise`, `_noise_model`, `StoppingVerdict` |
| `optimization/rehearsal.py` | campaign simulation and prior audition |
| `campaign_cycle.py` | the four steps in order, plus the report |
| `__main__.py` | `python -m latos next` and `latos rehearse` |

Key invariants:

- A pre-registration is written *before* the recommended sample exists, or the
  cycle says plainly that the prediction cannot later be presented as one.
- `BoConfig.point_noise_used` is recorded and serialised, so two runs cannot
  carry identical frozen configs having weighed their observations differently.
- Missing Tier-1 workbook fields are reported per row against their own field
  name and never defaulted.

## 6. Validation

- **Tests:** 124 new across the six modules (1600 total in the unit suite)
- **Quality gates:** ruff lint, ruff format, mypy strict — all clean on 108
  source files
- **Real-data behaviour:** none. Every number in this stage comes from synthetic
  objectives with known optima. See Limitations.
- **Numerical accuracy:** planted slopes and intercepts recovered to within
  measurement noise; the rehearsal independently reproduces the mixing-law
  `HARMS` verdict from inside the package.

```mermaid
flowchart LR
    W[recording workbook] --> P[ite-workbook parser]
    P -->|ΔT, ΔV per sample| S[slope fit]
    S -->|S ± σ_S| O[surrogate, heteroscedastic]
    O --> V[stopping verdict]
    O --> R[pre-registration]
    V --> N[next composition]
```

## 7. Limitations

- **No real measurements anywhere.** Every threshold shipped here is a reasoned
  guess validated only against synthetic data: `_R_SQUARED_WARN = 0.99`,
  `_CURVATURE_F = 10.0`, the 10 % offset warning, `_DEGENERATE_SIGMA_FRACTION`.
  They should be re-checked against real replicate scatter before being trusted.
- **A three-point standard error is biased low.** With one degree of freedom the
  residual variance follows a chi-squared whose median is 0.455 of its mean, so
  `σ_S` understates the truth more often than not — by roughly 2.4× at the
  median. A three-ΔT campaign hands the surrogate an over-confident noise
  estimate. Four ΔT values is the cheap fix; the behaviour is pinned by a test.
- **`converged` and the recommended `x` are unchanged.** The stopping verdict is
  additive, because frozen pre-registrations must keep replaying to the same
  numbers. So the engine can still recommend an exploration point while the
  verdict says `CONFIRM`.
- **The instrument is unknown**, so no parser exists for the raw `V(t)` trace.
  The transient analyzer described in the development plan is not built.
- **Phase 2 is deliberately not started** — campaign-level humidity
  confounding, endpoint drift control, and the campaign-record export all need
  real runs before they mean anything.

## 8. Thesis mapping

| Thesis section | What this stage feeds |
|---|---|
| Chapter 8 — Bayesian optimization of synthesis parameters | the heteroscedastic surrogate, the stopping verdict, and the pre-registration record |
| Chapter 9 — Reliability-gated closed-loop discovery | the whole cycle, and the measured negative results on physics priors |
| Methods — measurement validation | the ΔV-vs-ΔT linearity and intercept test as the ionic analogue of Wiedemann-Franz |

## See also

- [`RESULTS_LOG.md`](../../RESULTS_LOG.md) — chronological detail + bug log
- [`BENCHMARKS.json`](../../BENCHMARKS.json) — structured metrics
- [`references.md`](../references.md) — citations
- [`glossary.md`](../glossary.md) — terminology
