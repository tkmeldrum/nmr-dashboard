# Plan: uncertainty for T₂–D (a few-population model, fitted and sampled)

**Status:** decisions made (no offset; K automatic with override; D restricted to the resolvable window; free diffusion). Step (a), the model and fit, is built and merged. Step (b), MCMC and bootstrap (`nmrdash/t2d_uncertainty.py`), is built. Step (c), the page, figures, saving, params file and batch, is next.

## Why a second model

The T₂–D inversion gives a regularised *map*, not parameters. A single solve takes 20 s to 3 min in the browser, so a bootstrap of the map
(hundreds of solves) is out of reach, and the map's "regions" are summaries, not fitted quantities with intervals. The map also cannot say
what it does not know: on both sample datasets the fast-D region lies outside the diffusion range the b-values can resolve, and a regularised
solution happily puts weight there.

A model with a small number of populations fixes both. It is cheap to evaluate (milliseconds), so least squares, MCMC and a bootstrap are all
practical, and each population gets a T₂, a D and an amplitude with an interval, which is how results are usually reported. The regularised
map stays as the exploratory view that tells us how many populations to try.

## The model

For echo time *t* and b-value *b* (the data's own `b_vec`, in the solver's 10⁻⁹ s/m² units; D is reported back in m²/s):

    S(t, b) = Σₖ Aₖ · exp(−t / T₂ₖ) · exp(−b · Dₖ)          k = 1 … K, Aₖ ≥ 0   (optionally + a constant)

Free diffusion in each population, one T₂ each. Noise σ comes from the imaginary channel, exactly as for the map (the map fits to χ²/point
≈ 0.98–1.00 on both samples, which is the bar the model must reach). Populations are ordered by T₂ (then D) so that they have identities.

## Choosing K

1. Start from the regularised map: the number of regions above the threshold (3 on M1115, 5 on S3115) and their mean T₂ and D are the starting points.
2. Fit K = 1 … 4 (or up to the region count + 1) by least squares from several starts; report ΔAIC / ΔBIC like the 1D fit, and the χ²/point of each.
3. The page proposes the K that BIC prefers; the user can override. Populations the data cannot support get the same "not supported by the data" warning as 1D.

## Diffusion range

By default D is restricted to the resolvable window (1/b_max … 5/b_min, as on the map). A population that wants to sit outside it is reported as
**a bound only** ("D above ~8×10⁻¹⁰ m²/s"), not as a value with an interval, and flagged on the map and in the table. The user can widen the window.
Posteriors that pile up on a limit are flagged, as for T₂ in 1D.

## Uncertainty

| Method | What it is | Cost (25 000 points, K = 3) |
|---|---|---|
| Least-squares covariance | Student-t intervals and delta-method fractions, as in 1D | instant |
| MCMC (emcee) | New `Mcmc2D`, same interface as `Mcmc`: vectorised log-probability, ordered ln T₂ / ln D priors, start from the fit, chunked with progress and Stop | about 5 ms per evaluation of 32 walkers: roughly 1 min for the default 6000 steps |
| Bootstrap | Parametric or residual resampling of the 2D data, refit the model each time | about 0.3–1 s per replicate: a few minutes for 200 |
| The regularised map | **No intervals** (stated in the page); it is a view, not an estimate | — |

Derived quantities with intervals: population fractions, mean T₂ and mean D.

## What the user sees

- Uncertainty tab for T₂–D: number of populations (auto / set), D range (resolvable window / widen), MCMC and bootstrap settings (the existing controls).
- Results: a table per population (value, least-squares, bootstrap and MCMC intervals; the "Methods compared" outlier highlight applies), and a **posterior scatter of every population in the (T₂, D) plane drawn over the regularised map**, which shows at a glance where the two analyses agree. Corner plot, traces, residual map.
- Saved with everything else (tables, figures, provenance); `[t2d_model]` section in the params file; batch mode then gives a **time series of each population's T₂, D and fraction**.

## Validation (written first, as tests)

1. Synthetic data with known populations (inside the window, one near an edge, two close together): parameters recovered within their intervals; **coverage** of the 95% intervals over many noise seeds (as for the 1D MCMC).
2. K recovered by BIC when the truth is K = 1, 2, 3; no phantom population when K is too large (warning fires).
3. On M1115 and S3115: χ²/point at the noise level; the model's populations land on the map's regions; the fast-D population is reported as a bound.
4. Ordering and label stability across MCMC chains; speed budget checked.

## Risks

- **Degeneracy:** two populations with similar T₂ and D trade off. Intervals will be wide or multimodal; the corner plot and warnings must say so.
- **The model is exp(−bD) only.** Restricted diffusion or non-exponential decay would be absorbed into extra populations; K selection could then over-count. State this in the page.
- **Browser time:** MCMC with K = 4 on a long dataset could take several minutes; the existing estimate-then-confirm and Stop apply. If it is too slow, the sufficient-statistics form of the residual (SVD of the data, exact here because there are only 25 b-values) cuts the cost by an order of magnitude.

## Decisions (recommendation first)

1. **Constant offset in the model?** No by default, to match the map; an option.
2. **Choose K automatically?** Yes, from the map's regions and BIC, with a manual override.
3. **D prior = resolvable window?** Yes, widenable; out-of-window populations reported as bounds.
4. **Scope.** Three pull requests: (a) model, fit, K selection and the validation tests; (b) MCMC and bootstrap with coverage tests; (c) page, figures, saving, params file and batch.
