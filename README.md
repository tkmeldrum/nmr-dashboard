# nmr-dashboard

Browser-based processing of Magritek Kea CPMG NMR data. Drop a data folder onto the page, look at the
parameters and raw echoes, run multi-exponential fits and ILTs (optionally spatially resolved), and read off
fit statistics and uncertainties.

**All processing happens in your browser.** The page loads Python (numpy/scipy, via [Pyodide](https://pyodide.org))
into a Web Worker and runs the code in `nmrdash/` locally. Your data is never uploaded; this repository holds
only code. A Content-Security-Policy in `index.html` restricts network access to the CDN hosts that serve Pyodide
and Plotly, and the page has no analytics.

Live site: https://tkmeldrum.github.io/nmr-dashboard/ (use **Explore test data** to try it without any files).

## Workflow (the tabs run left to right)

1. **Load**: drag a folder in or click to browse. A parent folder with several experiments gets a dropdown.
   Shows the curated parameters (sequence, `data.2d` timestamp, pulse length with 90°/180° amplitudes in dB, echo
   time, echoes, points, dwell, repetition time, scans, data directory); everything else is under "Show all parameters".
2. **Raw data**: echo-magnitude heat map, a single echo (real/imaginary/magnitude) with an echo slider whose y-range
   is fixed from the second echo + 10%, and the summed decay.
3. **Process**:
   - *Echo processing*: **integrate each echo** (autophase, sum the points), or **FT each echo** (signal vs position z).
   - *Decay analysis*: **multi-exponential fit** (1–4 components, optional offset, optional 1–4 model comparison) and/or
     **ILT**. In FT mode every analysis is run at every z position. The model equations are shown.
4. **Uncertainty**: confidence level for the fit intervals (68 / 95 / 99%). MCMC and bootstrap are placeholders.
5. **Results**: spatial profile and echo map (FT mode), spatially resolved fits (T₂(z) with error bars, amplitude(z),
   reduced χ²(z), a linked ILT cross-section viewer, per-peak ILT T₂ and signal fraction vs z), the decay with fits and
   residuals, fit tables, fit-quality statistics and the ILT spectrum.

## What it computes

| Step | Details |
|---|---|
| Integrate | zero-order autophase, then sum all points of every echo (`nmrdash/echoes.py`; matches `kea_io.prepare_t2_data`) |
| FT | echo centre moved to t = 0, zero-filling inserted in the middle (no phase ramp), FFT, autophase; z = f / (γ·G) in µm; + z = + frequency = closer to the magnet (down); the echo map's z axis is flipped to match |
| Multi-exponential | `Σ Aᵢ exp(−t/T₂,ᵢ) [+ c]`, multi-start least squares, components sorted fast→slow, Student-t intervals from the covariance, delta-method errors for fractions and mean T₂, AIC/BIC model comparison, warnings for unresolved components (`nmrdash/multiexp.py`) |
| ILT | regularised NNLS, a copy of `MeldrumLabCode/pythonic/run_1dilt.py` (`nmrdash/ilt.py`); peak summary and optional α sweep in `ilt_tools.py`; max grid points is user-set (default 100, never more than the echo count) |
| Per-position fits | `nmrdash/spatial.py`: exponential fits at the natural resolution by default, ILT at every zero-filled point so the heat map is smooth; positions below a signal threshold (default 5% of the profile maximum) are skipped; ILT peaks are found on the position-summed spectrum and followed as bands |
| Statistics | RSS, MSE, residual SD, **residual SD / fitted signal range** and **reduced χ²** (independent of amplitude and echo count), noise σ taken from the imaginary channel after phasing (model-free), ΔAIC/ΔBIC (raw AIC/BIC only mean something as differences) |

Conventions: times are in seconds; value ± uncertainty pairs share one base exponent (`1.99e-2 ± 0.35e-2`); T₂ axes
use decade ticks (`1e-5, 1e-4, …`).

## Assumptions to confirm

- **¹H vs ¹⁹F** is preset from `b1Freq`, assuming **19.44 MHz is ¹H on this magnet** (so ¹⁹F ≈ 18.30 MHz). One constant:
  `REF_B1FREQ_1H_MHZ` in `nmrdash/constants.py`. No ¹⁹F dataset has been tested.
- Default gradient **G = 23.7 T/m**, zero-fill ×2⁴, z-window **−200 … +200 µm** (`nmrdash/constants.py`).
- ILT defaults: α = 1e8, T₂ range from the echo times (min = decade ≤ first echo, max = decade ≥ 10 × last echo).
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
nmrdash/        the Python package (also runs on a desktop: pip install -e .)
  kea_io.py echoes.py multiexp.py ilt.py ilt_tools.py spatial.py pipeline.py constants.py display_params.py
tests/          pytest: synthetic recovery tests, statistics invariance tests, every sample dataset
test_data/      sample Kea datasets (PUBLIC, see below)
tools/          build_site.py (assembles the deployable site), dev_server.py (local server)
docs/PLAN.md    the original build plan
```

`nmrdash/` is the canonical home of this code. It started as copies of `MeldrumLabCode/pythonic/` (`kea_io.py`,
`run_1dilt.py`) and `PED/`; the pythonic scripts have **not** yet been switched to import from `nmrdash`.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest                          # ~90 s: runs every sample dataset through the pipeline
python tools/dev_server.py      # http://localhost:8000  (generates the manifests on the fly)
python tools/build_site.py site # what CI deploys; versions every asset URL so browsers never serve stale code
```

Deploys: every push to `main` runs the tests, builds the site and publishes it to GitHub Pages
(`.github/workflows/pages.yml`).

**Test data is public.** Anything under `test_data/` is in a public repo and is bundled into the live site (only
`acqu.par` and `data.2d` are shipped). Commit only data you are happy to publish.

## Status and roadmap

**Done:** load and raw display; integrate and FT; multi-exponential fit with intervals and model comparison; ILT with peak
tables and α sweep; spatially resolved fits with a linked cross-section viewer; fit-quality statistics; tabbed workflow;
sample-data explorer; CI and Pages deploy.

**Next (see `docs/PLAN.md` for the original phasing):**

1. **PED analysis**: port `PED/t2_analysis_real_data.py` (`PED_LBFGS`, `T2Analyzer`) from PyTorch to numpy/scipy L-BFGS,
   keeping the regularisers (None, L1, L2, PED, ElasticNet, SQRLasso), with a **parity test against the PyTorch
   original** (needs PyTorch locally; not run in CI).
2. **Uncertainty tab**: MCMC (`emcee`) for the multi-exponential fits with corner and trace plots; residual bootstrap for
   ILT/PED spectra. Long runs get a time estimate, a confirmation, a progress bar and a Cancel button.
3. **Plot controls and saving**: common plot settings (axis scale and limits, time units, fonts, line widths, colormap,
   figure size, a "no title" option for publication figures). Save numeric results (CSV) and figures (PNG/SVG/PDF,
   rendered with matplotlib) into a `processed_<timestamp>/` folder inside the data folder (Chrome/Edge via the File
   System Access API, or another folder the user picks); other browsers get a `.zip`. A `processing.json` records
   every setting plus the `nmrdash` version for reproducibility. *Needs testing in a real Chrome.*
4. **Export a processing script**: generate a standalone Python script from the settings chosen in the browser, so the
   same analysis can run on local hardware (faster, no browser limits), and extend it to **batch and time-series** data.
   The settings are already one JSON object and `nmrdash` is pip-installable, so the script is mostly that JSON plus
   `load()` / `process()` calls in a loop.
5. Smaller items: an option to skip or down-weight a distorted first echo; non-CPMG data (T1 saturation recovery and
   others) and the NN-decays / T2-D formats from `pythonic/`; make `pythonic/` import from `nmrdash`; slim CI time.

## Not yet verified

Everything was checked in headless Chromium against the built site (files served from disk), never against the live URL in
a normal browser. Not tested at all: the real File System Access save path, Safari/Firefox, ¹⁹F data.
