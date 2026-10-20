# nmr-dashboard

Browser-based processing of Magritek Kea single-sided NMR data (CPMG, saturation recovery, T₂–D). Drop a data folder onto the page, look at the
parameters and raw echoes, run multi-exponential fits, ILTs and T₂–D analyses (optionally spatially resolved), and read off
fit statistics and uncertainties. **Developers and new contributors: start with [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).**

**All processing happens in your browser.** The page loads Python (numpy/scipy, via [Pyodide](https://pyodide.org))
into a Web Worker and runs the code in `nmrdash/` locally. Your data is never uploaded; this repository holds
only code. The libraries are served from the site itself and a Content-Security-Policy in `index.html` allows connections to the site only
(see "Self-hosted libraries" below); the page has no analytics.

Live site: https://dashboard.meldrumlab.org/ (use **Explore test data** to try it without any files).

**Tecmag data (`.tnt`).** `nmrdash` reads Tecmag files too (CPMG, T1 saturation recovery, stimulated-echo T₂–D), using the sequence's own variables and delay tables, so the
same analyses apply. Drop a folder with `.tnt` files (each file is one dataset; a batch can mix Kea folders and `.tnt` files) or drop `.tnt` files onto the drop zone directly (results then come as a .zip);
the command-line tool takes `.tnt` files and folders too. The gradient is not in a Tecmag file: `ft.G` and `t2d.gradient` default to `"auto"`, which means 6.59 T/m for Tecmag data
and 23.67 T/m for Kea data (set a number to override). The experiment date stored in the file is the time stamp. What is read and the assumptions made are in
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §9.

## Workflow (the tabs run left to right)

0. **The sequence decides the analysis.** The `experiment` entry in `acqu.par` is read (spelling variants tolerated): **CPMG**
   → exponential decay (T₂), with FT, ILT and per-position fits; **T1Sat** → saturation recovery (T₁); **SSE-T2** → T₂–D
   correlation (a 2D inversion plus a few-population model). The page shows the detected analysis on the Load tab and offers only the
   controls that apply.
1. **Load**: drag a folder in or click to browse. A parent folder with several experiments gets a dropdown.
   Shows the curated parameters (sequence, `data.2d` timestamp, pulse length with 90°/180° amplitudes in dB, echo
   time, echoes, points, dwell, repetition time, scans, data directory); everything else is under "Show all parameters".
2. **Raw data**: echo-magnitude heat map, a single echo (real/imaginary/magnitude) with an echo slider whose y-range
   is fixed from the second echo + 10%, and the summed decay.
3. **Process**:
   - *Echo processing*: **integrate each echo** (autophase, sum the points), or **FT each echo** (signal vs position z).
   - *Decay analysis*: **multi-exponential fit** (1–4 components, optional offset) and/or
     **ILT**. In FT mode every analysis is run at every z position. The model equations are shown.
   - *Advanced* (collapsed): next to the multi-exponential fit, **compare 1–4 components** (on by default, hidden); next to the ILT, the **α sweep**
     (L-curve, off). At the end of the tab, **Advanced settings** (closed by default): the confidence level for the fit intervals (68 / 95 / 99%), and optional
     **MCMC** and **bootstrap** on the whole decay or on one z position. They run after the processing, with their progress on the Results tab, once you have seen a timed
     estimate (a prompt appears only for runs over about 20 seconds; a yes is remembered for the session for runs up to 1.5× as long) and confirmed; a progress bar and a **Stop (keep partial results)** button are shown while they run. Loading a params file that turns one of these on opens its block.
4. **Results**: spatial profile and echo map (FT mode), spatially resolved fits (T₂(z) with error bars, amplitude(z),
   reduced χ²(z), a linked ILT cross-section viewer, per-peak ILT T₂ and signal fraction vs z), the decay with fits and
   residuals, fit tables, fit-quality statistics and the ILT spectrum.

## Saturation recovery (T1Sat)

Kea writes the recovery curve to `data.dat` (delay in s, amplitude); `nmrdash` reads that, not `data.2d`. The model is
M(t) = c + Σᵢ Aᵢ(1 − e^(−t/T₁,ᵢ)) with **one component by default and two on request** (a second component is rare and
needs good data); the constant offset c is **on by default** because Kea recovery curves rarely start at zero (on the sample
data leaving it out raises the residual from 38 to 305). There is no ILT: recovery curves are too sparse (11–15 points) for it.
The same fit statistics, confidence intervals (68/95/99%), model comparison (ΔAIC/ΔBIC for 1 vs 2 components), **MCMC and
bootstrap**, params file, batch, provenance and exports apply. These data have **no imaginary channel**, so there is no
measured noise σ: reduced χ² is not available, MCMC samples σ, and the bootstrap resamples residuals. Params file section: `[t1]`.
Batch tables keep `T2_*` (CPMG) and `T1_*` (T1Sat) columns apart, with a `kind` column and `timeseries_T1.png`.

## T₂–D correlation (SSE-T2)

The sequence name `SSE-T2` selects a 2D inverse Laplace transform of the echo × diffusion-delay data. The solver is the one in
`MeldrumLabCode/pythonic/run_2dilt.py` **as it is** (a MATLAB-matched, SVD-compressed Kronecker solve with curvature regularisation); only
unit handling, the summaries and the interface are new. Files read: `acqu.par`, `data2D_Cp.2d` (or `data2D.dat`), `data.dat`, and
`DiffAxis.dat` / `T2Axis.dat` when present; the imaginary channel after phasing gives a measured noise σ and a χ² per point.

- **Defaults (as you specified):** T₂ 1e-5 to 1 s, D 1e-12 to 2.2e-9 m²/s (the upper limit is about the self-diffusion coefficient of water; it was 1e-8 until 2026-10-19), α = 1e9, 50 T₂ × 25 D points. D is entered and reported in m²/s; the
  solver works on D × 10⁹ (and b × 10⁻⁹) and every result is converted back. Params file section `[t2d]`; `exclude_first` applies.
- **Cost:** one dense problem of (T₂ points × D points) unknowns: native times of 1.9 s (30 × 25), 8 s (50 × 25), 54 s (100 × 25) and ~0.9 GB for the
  last. In the browser 50 × 25 takes about 20 s and 100 × 25 about 3 minutes; the defaults are 100 T₂ × 20 D points (about 2 minutes in the browser; D points never exceed the number of diffusion delays). Before a long solve the page times a small trial, shows an estimate and asks;
  there is no progress to report during a solve, so a **Stop** button restarts Python (data and settings are kept).
- **What the data can resolve (important).** A diffusion coefficient is only measurable if the signal decays over the b-values used: roughly from
  1/b_max to 5/b_min (for the synthetic T₂–D sets, b = 6.5e9 – 2.9e12 s/m², i.e. **3.5e-13 to 7.7e-10 m²/s**). The regularisation (a curvature penalty) does
  not discourage large amplitudes where the kernel is ~0, so outside that window the solution can hold a lot of "signal" that the data say nothing
  about: with the grid reaching 1e-8 a large part of the spectrum's weight can sit above the window (the 2.2e-9 default for D max shortens that tail), and a synthetic test with known populations shows the same
  drift. The page therefore shades the unresolvable band, flags each region as constrained or not, reports each region's share of the *resolvable*
  part, and offers **"Use the range the b-values can resolve"** (params: `d_min = d_max = "auto"`). Within that window the slow population of the
  synthetic test is recovered to ~10% in T₂ and D; the fast one's T₂ is good but its D can be off by ~2× and its share by ~20% (and shifts with α).
- **Uncertainty:** the regularised map has no intervals. A **few-population model** (1–4 populations by BIC unless set, no offset, D limited to the resolvable window) is fitted as
  well, and MCMC and the bootstrap sample it (`docs/PLAN_T2D_UNCERTAINTY.md`). The **b-values are calculated** from the gradient setting (default 23.67 T/m), Δ and τ in `acqu.par`
  (b = (γG)²τ²(Δ + 2τ/3)); the spectrometer's own axis files are only a cross-check.
- **Outputs:** `t2d_spectrum.csv` (D rows × T₂ columns), `t2d_peaks.csv`, projections, `t2d_statistics.csv`, `t2d_data_fit_residual.csv`, the spectrum
  and data/fit/residual figures, `summary.md`, provenance. Batch tables gain `t2d_peak{1-3}_*` columns.

## What it computes

| Step | Details |
|---|---|
| Integrate | zero-order autophase, then sum all points of every echo (`nmrdash/echoes.py`; matches `kea_io.prepare_t2_data`) |
| FT | echo centre moved to t = 0, zero-filling inserted in the middle (no phase ramp), FFT, autophase; z = f / (γ·G) in µm; + z = + frequency = closer to the magnet (down); the echo map's z axis is flipped to match |
| Multi-exponential | `Σ Aᵢ exp(−t/T₂,ᵢ) [+ c]`, multi-start least squares, components sorted fast→slow, Student-t intervals from the covariance, delta-method errors for fractions and mean T₂, AIC/BIC model comparison, warnings for unresolved components (`nmrdash/multiexp.py`) |
| ILT | regularised NNLS, a copy of `MeldrumLabCode/pythonic/run_1dilt.py` (`nmrdash/ilt.py`); peak summary and optional α sweep in `ilt_tools.py`; max grid points is user-set (default 100, never more than the echo count) |
| Per-position fits | `nmrdash/spatial.py`: exponential fits at the natural resolution by default, ILT at every zero-filled point so the heat map is smooth; positions below a signal threshold (default 5% of the profile maximum) are skipped; ILT peaks are found on the position-summed spectrum and followed as bands |
| MCMC | `nmrdash/uncertainty.py`, [emcee](https://emcee.readthedocs.io) ensemble sampler on the multi-exponential model. Sampled: amplitudes (flat, ≥ 0), ln T₂ (log-uniform, ordered fast→slow so components cannot swap), offset, optionally ln σ; σ otherwise fixed from the imaginary channel. Output: credible intervals, trace and corner plots, acceptance, autocorrelation time and effective samples (with warnings if the chain is under ~50 τ), a posterior-predictive band on the decay |
| Bootstrap | refits surrogate decays (fitted curve + resampled noise): the multi-exponential model (warm-started) and/or the ILT. Noise source: the **measured imaginary-channel noise** (preferred, model-free), the fit residuals, or Gaussian with the measured σ. Output: percentile intervals, parameter histograms, an ILT band, per-peak T₂ and fraction intervals |
| Statistics | RSS, MSE, residual SD, **residual SD / fitted signal range** and **reduced χ²** (independent of amplitude and echo count), noise σ taken from the imaginary channel after phasing (model-free), ΔAIC/ΔBIC (raw AIC/BIC only mean something as differences) |

Conventions: times are in seconds; value ± uncertainty pairs share one base exponent (`1.99e-2 ± 0.35e-2`); T₂ axes
use decade ticks (`1e-5, 1e-4, …`).

## Assumptions to confirm

- **¹H vs ¹⁹F** is preset from `b1Freq`: **19.44 MHz is ¹H on this magnet** (confirmed; so ¹⁹F ≈ 18.30 MHz). One constant:
  `REF_B1FREQ_1H_MHZ` in `nmrdash/constants.py`. The lab has a **second magnet** run by a non-Kea (Tecmag) spectrometer; when
  Tecmag reading is added, that magnet's reference frequency must be set separately. No ¹⁹F dataset has been tested.
- Default gradient **G = 23.67 T/m** (`nmrdash/physics.py`; adjustable: FT tab and T₂–D settings; the T₂–D b-values are calculated from it, Δ and τ), zero-fill ×2⁴, z-window **−200 … +200 µm** (`nmrdash/constants.py`).
- ILT defaults: α = 1e8 and the T₂ range from the echo times (min = decade ≤ first echo, max = decade ≥ 10 × last echo). The page takes α and the T₂ limits as log₁₀ values (log₁₀ T₂ = −5 is 10 µs, 0 is 1 s); the params file keeps α in plain form and T₂ in seconds.
- The per-position phase is a zero-order correction per z; positions closer than the natural resolution
  (1 / (points × dwell), converted to µm) are interpolations of each other, not independent measurements.

## Repository layout

```
index.html, style.css, *.js      the page (plain ES modules, no build step)
  app.js        worker plumbing, folder loading, tab flow     tabs.js      tab bar
  raw.js        raw-data plots                                process.js   settings form and run
  results.js    results, fit tables, linked ILT viewer        equations.js model equations
  plotutil.js   plot styling, number formatting               state.js     shared state
worker.js       Pyodide in a Web Worker (load / process messages, progress)
nmrdash/        the Python package (also runs on a desktop: pip install -e .; its command is `nmrdash`)
  physics.py inference.py     the physics models and the estimation formulas (everything else calls these)
  kea_io.py sequences.py t2d.py echoes.py multiexp.py ilt.py ilt2d.py ilt_tools.py spatial.py pipeline.py constants.py display_params.py
  t2d_model.py uncertainty.py t2d_uncertainty.py (MCMC, bootstrap)   params.py (the params file)   runner.py cli.py webbatch.py (run, batch)
  outputs.py figures.py formatting.py (results on disk)   provenance.py export.py
tests/          pytest: synthetic recovery tests, statistics invariance tests, every sample dataset
test_data/      synthetic sample datasets (see below)
tools/          build_site.py (assembles the deployable site), dev_server.py (local server)
docs/DEVELOPMENT.md  the development log and guide (read first)   docs/PLAN*.md  the plans   docs/REFERENCES.md  references to check
```

`nmrdash/` is the canonical home of this code. It started as copies of `MeldrumLabCode/pythonic/` (`kea_io.py`,
`run_1dilt.py`); the pythonic scripts have **not** yet been switched to import from `nmrdash`.

## Where the physics is

A reviewer who wants to check the models can read two short files, listed with equations, references and links to the lines on the site's About page:

- `nmrdash/physics.py`: the signal models (exponential and multi-exponential decay, saturation recovery, diffusion attenuation, the T₂–D signal and the b-value of the stimulated echo), the inversion kernels, and the position relations of the static-gradient profile. NumPy only.
- `nmrdash/inference.py`: how numbers are estimated from them (noise from the imaginary channel, least-squares measures, AIC/BIC, Student-*t* intervals, the regularisation of the inversions).

Every other module calls these functions; the formulas are not repeated elsewhere. Adding a function to either file without listing it in its `MODELS` / `METHODS` registry fails a test, and `tests/test_physics.py` / `tests/test_inference.py` check each one against independent calculations.

## Self-hosted libraries

The site serves everything it needs itself: Pyodide (Python in the browser) with NumPy, SciPy and Matplotlib, Plotly.js and the emcee
package, about 50 MB in all. They are downloaded at build time by `python tools/fetch_vendor.py` into `vendor/` (not committed) and checked
against the checksums in `tools/vendor.lock.json`; the page's content-security policy allows connections to this site only. To upgrade a
library, change its version in `tools/fetch_vendor.py`, run it with `--update`, review the new checksums, commit `vendor.lock.json`, and
update the version in `about.html` (a test checks they agree). Run `python tools/fetch_vendor.py` once before using `tools/dev_server.py`.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"        # includes emcee (MCMC tests) and pytest-xdist; run the tests with: pytest -q -n auto
pytest                          # ~90 s: runs every sample dataset through the pipeline
python tools/dev_server.py      # http://localhost:8000  (generates the manifests on the fly)
python tools/build_site.py site # what CI deploys; versions every asset URL so browsers never serve stale code
```

Deploys: every push to `main` runs the tests, builds the site and publishes it to GitHub Pages
(`.github/workflows/pages.yml`).

**Test data is synthetic.** `test_data/` holds only generated datasets (made by `tools/make_synthetic_data.py`, with known true values listed in
`test_data/README.md`), bundled into the live site as "Explore test data". No real measurements are kept in this repository: add new test cases to the generator, not
recorded data.

## Status and roadmap

**Done:** load and raw display; T₂–D 2D ILT (branch `t2d`); saturation recovery (T1Sat); MCMC and bootstrap (branch `uncertainty`, not yet merged); command-line tool, params file, batch/time-series and provenance (branch `script-export`, built on top of it); integrate and FT; multi-exponential fit with intervals and model comparison; ILT with peak
tables and α sweep; spatially resolved fits with a linked cross-section viewer; fit-quality statistics; tabbed workflow;
sample-data explorer; CI and Pages deploy.

**Next (see `docs/PLAN.md` for the original phasing):**

1. **Further regularised-inversion options** (planned, details to follow once the underlying methods are published): more than one
   penalty to choose from and compare on one plot, with the page, params file, bootstrap and provenance treating them alike.
2. (reserved)
3. ~~Uncertainty tab: MCMC and bootstrap~~ **done on branch `uncertainty`** (see the pull request); still to do: bootstrap
   bands for the planned inversion options, and MCMC/bootstrap at *every* z position (today: one position at a time).
4. ~~Plot controls and saving~~ **done on branch `save-and-plot-controls`**: a Save card on the Results tab writes tables (CSV),
   figures (PNG/SVG/PDF, matplotlib), `params_used.toml`, `provenance.json` and `summary.md` into a new
   `processed_<timestamp>/` folder (inside the data folder, or another folder, in Chrome/Edge; a `.zip` elsewhere).
   Plot settings (font size, line and marker scale, colour map, figure size, titles none/custom, legends, time units)
   apply to the page's plots and the saved figures; time units and figure size are saved-figure only. Still to do:
   per-figure limits and titles are in (`[figures.<name>]`), and the params file has `[style]`; the page's Export button writes both, and T₂–D data also get a few-population model (a handful of populations with intervals, `[t2d_model]` in the params file) that MCMC and the bootstrap sample. **Load params…** on the Process tab fills the page's settings from a params file. When a folder holds several datasets, a banner says so and the Process tab gets a **Batch** card that runs the same settings on all of them (same sequence), writing per-dataset folders plus a time-series summary, or one zip. *The real File System Access path
   (permission prompt, a real folder) has only been tested against a stand-in directory; try it in Chrome.*
5. ~~Export a processing script (batch and time series)~~ **done on branch `script-export`** (see the pull request). Still to do:
   import a params file into the page, a time-resolved T₂(z, t) map for FT batches, and a SLURM example.
6. ~~T₂–D (SSE-T2)~~ **done on branch `t2d`**. Still to do: a parametric few-component T₂–D model with MCMC, and a bootstrap of the 2D map (opt-in; each replicate is a full solve). Original notes: The 2D ILT exists in `MeldrumLabCode/pythonic/` (`run_2dilt.py`, `kea_t2d_io.py`, MATLAB-matched); the sample folders in `test_data/` hold `data2D_Cp/Re/Im.2d`. MCMC applies to a parametric few-component T₂–D model, not to the full 2D ILT map (the bootstrap does).
7. **Tecmag data** (second magnet, own Python reader exists): add a reader beside `kea_io.py` and check the Larmor frequency.
8. Smaller items: an option to skip or down-weight a distorted first echo; non-CPMG data (T1 saturation recovery and
   others) and the NN-decays / T2-D formats from `pythonic/`; make `pythonic/` import from `nmrdash`; slim CI time.

## Run it on your own computer (command line, params file, batch)

The same code the page runs is a normal Python package with a command-line tool, so an analysis can be repeated on local
hardware, on many datasets, or on a cluster.

```bash
pip install "nmrdash[mcmc] @ git+https://github.com/tkmeldrum/nmr-dashboard"     # once (or pip install -e . in a checkout)
nmrdash params -o params.toml                  # a fully commented params file to edit
nmrdash run   params.toml  data/CPMG/1         # one dataset: a folder with acqu.par + data.2d
nmrdash batch params.toml  data/series --jobs 8   # every dataset under a folder (a time series), 8 at a time
nmrdash batch params.toml  data/series --list     # just list the datasets and their timestamps
```

- **From the page:** the **Export params + script** button (Process tab) downloads a `params.toml` holding
  exactly the settings currently on the page, plus `process_nmr.py`, a short runner (`python process_nmr.py DATA_FOLDER`, add
  `--batch` for a folder of datasets). Use `defaults` in place of a params file to run with the built-in defaults.
- **Excluding the first points:** `[processing] exclude_first = n` (or the "Exclude the first n echoes" field on the Process tab; for T1 it counts
  delays) leaves the first n points out of **every** analysis: multi-exponential fit, ILT, per-position fits, MCMC, bootstrap and the noise estimate.
  They are still plotted (open markers) and written to `decay.csv` with `included_in_fits = 0`. The first echo is often distorted on real instruments; one scaled
  by 0.3 in the synthetic CPMG set (`test_excluding_a_distorted_first_echo_improves_the_fit`) wrecks the FT-sum fit, and excluding it restores the true T₂.
- **Params file** (`nmrdash/params.py` is the single definition): sections `processing`, `ft`, `multiexp`, `ilt`,
  `uncertainty`, `mcmc`, `bootstrap`, `output`, `batch`; every setting is commented in the generated file, unknown names and
  out-of-range values are rejected with a message naming the setting, and `"auto"` means the same rule the page uses.
- **Where results go:** a `processed_<timestamp>` folder **inside each data folder** (raw data is never touched; re-running
  never overwrites, it adds `-2`, `-3`, …), or under `output.directory` / `--out`. It holds CSV tables (decay, fit components
  with intervals, fit quality, model comparison, ILT spectrum / peaks / α sweep, FT profile and per-position tables, MCMC and
  bootstrap summaries and samples), figures (`png`/`pdf`/`svg`; titles can be switched off), **`report.pdf`** (a 2–4 page US-Letter report: settings, key numbers with intervals, every warning, the main figures, the physics that applies and the provenance; `[output] report = false` to skip; a batch also writes `batch_report.pdf`), `summary.md` (readable, same
  number formatting as the page), `params_used.toml` (the resolved settings, runnable again) and `provenance.json`.
- **Batch:** every folder under the root holding `acqu.par` is a dataset (results folders are skipped), sorted by its
  `data.2d` modification time. `batch_summary.csv` has a row per dataset with `timestamp_utc`, `timestamp_local`,
  **`elapsed_min`** (from the first dataset, a date-time you set, or none: `[batch] t0`), status, fit results and ILT peaks;
  `timeseries.png` plots the T₂ components and amplitude against elapsed time. A dataset that cannot be processed (for
  example non-CPMG data) is recorded as skipped or failed and never stops the rest. If copying files reset the
  modification times, give the real ones in a CSV (`folder,timestamp`) with `[batch] timestamps_file`.
- **Provenance** (script **and** page): `provenance.json` records the input files with **SHA-256**, the fully resolved
  settings, the exact params file text, the `acqu.par` values that matter, random seeds, the `nmrdash` version and commit
  (flagged if the checkout had uncommitted changes), Python / numpy / scipy / matplotlib / emcee versions and the platform
  (for the page: Pyodide and the browser), the command line, and warnings raised. On the page, **Results → How these results
  were produced** shows it and downloads it; it is refreshed after MCMC / bootstrap runs.
- **Reproducibility note:** the page and a local run give the same result up to the numerical libraries' rounding
  (the page runs numpy 2.0 / scipy 1.14 under Pyodide). On a test dataset the fitted T₂ agrees to ~3×10⁻⁷ relative, far below
  its uncertainty; the recorded versions explain any such difference.

## How the long jobs work

MCMC and bootstrap are small chunked objects (`step(k)` then `result()`), driven from the page in ~0.4 s chunks, so the
Python worker stays responsive: **Stop** simply stops sending chunks and the partial chain is summarised (burn-in is cut
if needed and a warning says so). Their time estimates come from a short timed trial in the same browser. `emcee` is not
bundled with Pyodide; it is fetched once from PyPI with `micropip` the first time MCMC is used (the page's
Content-Security-Policy already allows `pypi.org` and `files.pythonhosted.org`), so MCMC needs a connection that first time.

## Not yet verified

Everything was checked in headless Chromium against the built site (files served from disk), never against the live URL in
a normal browser. Not tested at all: the real File System Access save path (only a stand-in directory was), Safari/Firefox, ¹⁹F data, and the `emcee` download from the real PyPI inside a real browser (in tests the same files are served from a local cache).
