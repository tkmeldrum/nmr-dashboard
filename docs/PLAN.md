> **Status:** this was the original plan. For what is built and what comes next, see [README.md](../README.md).

# Plan: Browser-based NMR processing dashboard (GitHub Pages, local-only processing)

## Context
The lab's Kea CPMG processing lives in Python scripts and notebooks under `MeldrumLabCode/pythonic/`. Using them means a working Python environment and editing paths in notebooks. The goal is a static site on GitHub Pages: a user drops a Kea data folder into the browser, sees its parameters and raw data, picks processing options, and saves results. **All computation runs in the user's browser. No data ever leaves the machine**, and the repo holds only code.

Decisions made:
- **Engine: Pyodide.** The existing numpy/scipy code runs in WebAssembly, so the site, notebooks, and CLI share one Python codebase.
- **Repo: new repo** (e.g. `tkmeldrum/nmr-dashboard`), deployed to Pages with GitHub Actions.
- **Saving:** Chrome/Edge write straight into the data folder (File System Access API). Firefox/Safari get a `.zip` download.
- **v1 formats:** Kea `data.2d` / `data.1d` CPMG + `acqu.par` only. NN-decays series and T2-D come later.

## Architecture
```
index.html ── main thread (UI, Plotly charts, file I/O)
     │  postMessage (settings, file bytes)  ▲ results, progress
     ▼                                      │
worker.js ── Pyodide + numpy/scipy/matplotlib (+ emcee via micropip)
     └── imports the package  nmrdash/  (plain Python, also testable with CPython/pytest)
```
- **Build-free static site**: plain ES modules, Pyodide and Plotly.js from jsDelivr with pinned versions. Nothing to compile, so it's easy for students to maintain.
- **Pyodide runs in a Web Worker**, so the UI never freezes during ILT or MCMC, and progress can stream back.
- **Pyodide starts loading on page open**, so it is usually warm by the time a folder is dropped (about 30 MB the first time, then browser-cached).
- **Two plot layers:** Plotly for interactive on-screen plots, and matplotlib (in Pyodide) for saved publication figures (PNG/SVG/PDF). Both read one shared "plot settings" object, so on-screen and saved figures match.
- **Privacy guard:** a `<meta http-equiv="Content-Security-Policy">` restricts `connect-src` to the CDN hosts, so the page cannot send data anywhere even by accident. There is no analytics. The README says data stays local.

## Repo layout (new repo)
```
index.html, app.js, worker.js, ui/ (panels: load, params, raw, process, results, save), style.css
nmrdash/                      # Python package loaded into Pyodide
  kea_io.py        ← copy of pythonic/kea_io.py (read_kea_binary, read_kea_params, _autophase)
  echoes.py        ← integrate each echo (from prepare_t2_data) + FT each echo (cleaned-up echoFT)
  multiexp.py      ← n-component exp fits (generalizes kea_decay_series_io.fit_decay / exp_decay / biexp_decay)
  ilt.py           ← copy of pythonic/run_1dilt.py (MATLAB-matched; use this, not processing_funcs.rmea1d)
  uncertainty.py   ← covariance CIs, emcee MCMC, ILT bootstrap
  figures.py       ← matplotlib figure builders driven by the plot-settings dict
  export.py        ← CSV/JSON writers, processing-settings record
  pipeline.py      ← thin entry points the worker calls (load, process, export)
tests/             ← pytest on CPython with synthetic decays (no real data committed)
.github/workflows/ ← pytest + Pages deploy
```
**Canonical code location: `nmrdash/` in the new repo.** It becomes the single home for the CPMG I/O, echo processing, fitting, ILT, and uncertainty code. It is an installable Python package (`pyproject.toml`), so the same code runs in the browser (Pyodide loads it from the site) and on a desktop (`pip install -e path/to/nmr-dashboard`, or `pip install git+https://github.com/tkmeldrum/nmr-dashboard`). Once a module moves into `nmrdash`, the matching `pythonic/` file becomes a thin wrapper that imports from `nmrdash`, keeping its CLI and notebook interface. For example, `pythonic/run_1dilt.py` becomes `from nmrdash.ilt import *`. That leaves one version of the math. Code not yet in the dashboard (T2-D, the decay series, wood/gel) stays in `pythonic/` until it's needed.

## Features → implementation

**1. Load folder (drag/drop or browse)**
- Drop: `DataTransferItem.getAsFileSystemHandle()` in Chromium, which also gives a writable handle for saving. Other browsers use `webkitGetAsEntry()`. Browse: `showDirectoryPicker({mode:'readwrite'})`, falling back to `<input webkitdirectory>`.
- Copy `acqu.par` + `data.2d`/`data.1d` into Pyodide's in-memory FS (`/data/`) and call `pipeline.load()`.

**2. Show parameters immediately.** Run `read_kea_params` and show a short curated list of CPMG-relevant parameters. **Tyler will supply the list, working from sample data**, since most of `acqu.par` isn't relevant. The list lives in one config dict (`nmrdash/display_params.py`) so it's easy to edit. The full `acqu.par` sits behind a collapsed "Show all parameters" toggle. Then check whether the data is CPMG (echoTime and nrEchoes present, a 2D file with yDim = nrEchoes, or a 1D echo train). If not, grey out the analysis options with the message "non-CPMG data: not yet supported".

**3. Raw data display (as collected)**
- Heatmap of |signal| over points within an echo × echo number.
- A selected echo shown as real, imaginary, and magnitude, with an echo-index slider.
- Quick summed decay (sum over points per echo) on linear/log axes.

**4. Processing options**
- *Echo processing* (choose one):
  - **Integrate**: autophase (`_autophase`), then sum all points in each echo. No integration window.
  - **FT each echo**: zero-fill, FFT, and fftshift to a z-axis using G and γ. G defaults to 23.7 T/m (`pythonic/constants.py`). A nucleus toggle sets γ: **¹H (proton), 42.577 MHz/T** (default) or **¹⁹F, 40.078 MHz/T**. Inputs: z-range and either z-bins or "every z point". Output: a profile and one decay per z-bin. Rewrite `processing_funcs.echoFT` cleanly: it uses snake_case param keys that don't match `kea_io`'s camelCase, and its 1D branch has an indexing bug.
- *Decay analysis* (applied to the bulk decay or to each z-bin):
  - **Multi-exponential fit**: 1–4 components, optional offset, bounds and initial guesses (p0) set automatically. Report the amplitudes, T2s, reduced χ², and residuals for each component.
  - **ILT**: `run_1dilt`, with inputs T2 min/max, number of steps, and α. Includes an optional α sweep with an L-curve plot.

**5. Uncertainties**
- **Fit confidence intervals**: from the `curve_fit` covariance, as t-distribution 68/95% CIs, with error propagation for derived values (mean T2, fractions).
- **MCMC**: `emcee` (pure Python, installed with micropip) on the multi-exponential model. Log-uniform priors on T2 and positive amplitudes, noise σ either estimated from the tail of the decay or sampled. Settings: walkers, steps, burn-in. Outputs: posterior medians with 16/84 percentiles, a corner plot, and a trace plot.
- **Long-run handling (MCMC and bootstrap):** before starting, show a confirmation dialog warning that the run may take a while, with a rough time estimate from a short timed trial. The run only starts once the user confirms. While it runs, the worker sends progress updates that drive a progress bar (step k of N plus elapsed and remaining time). A Cancel button stops the worker and restarts Pyodide.
- **ILT**: MCMC doesn't suit a regularized inversion, so offer residual bootstrap instead (refit N resampled decays) and show a median spectrum with a 5–95% band. Say so in the UI.

**6. Display and plot controls.** For each figure: axis lin/log, axis limits, time units (**default s**, ms optional), font size, line width, marker size, colormap, figure size/DPI, title text with a **"No title" checkbox** for publication figures, and legend on/off. Settings are saved in `localStorage` as per-user defaults.

**7. Save**
- Default target: a `processed_YYYYMMDD-HHMMSS/` subfolder inside the dropped data folder. Never overwrite the raw files. A "Choose another location…" button opens `showDirectoryPicker`. Non-Chromium browsers download a `.zip` built with JSZip.
- Contents:
  - `decay.csv` (t, raw, fit, residual)
  - `spectrum.csv` (T2, amplitude, and band if bootstrapped)
  - `fit_results.csv` (params ± CI)
  - `mcmc_samples.npz/csv` (optional)
  - `zprofile.csv` / `t2_vs_z.csv` (if FT)
  - figures in the chosen format(s)
  - `processing.json`: every setting, the acquisition params, the nmrdash version and git SHA, and a timestamp, so any result can be reproduced.

## Phased build
0. **Scaffold**: repo, Pages Action, worker that loads Pyodide and runs `numpy` "hello". Check the deployed URL.
1. **Load + params + raw display** (`kea_io`, panels 1–3).
2. **Integrate → multi-exp fit + ILT**, with Plotly results and covariance CIs.
3. **FT-each-echo** path, z-binning, T2(z) map.
4. (reserved)
5. **MCMC** (emcee) and ILT bootstrap.
6. **Plot controls + matplotlib export + save** (FS Access + zip fallback).
7. **Polish**: error messages, a "how to use" page, cleanup of the `pythonic/` cross-references.

## Verification
- **pytest on CPython (CI)**:
  - Synthetic mono-, bi-, and tri-exponential decays with noise. Fits recover T2s within their CIs.
  - The ILT peak lands at the true T2.
  - FT of a synthetic box profile gives the expected width.
  - Round trip of `read_kea_binary` on a generated file for each dataType.
- **Regression against existing scripts**: on one real dataset, `nmrdash` output must match `pythonic/run_kea_ilt.py` (spectrum, RMSE) and `kea_decay_series_io.fit_decay` numerically. This runs locally only, with the data path supplied as an argument and nothing committed.
- **Browser end-to-end**: open the Pages URL in Chrome and drop a real `data.2d` folder. Check that the params show, the raw plots render, each processing path runs, and the `processed_*` subfolder appears in the data folder. Repeat in Safari/Firefox and check that the zip downloads.
- **Privacy check**: watch the DevTools Network tab during a full run. The only requests should be CDN fetches for Pyodide, packages, and Plotly, with no request carrying data.

## Test data
Tyler will upload sample datasets after the repo is created. **If the repo is public, anything committed to it is public too**, and that conflicts with "the repo holds only processing code". Two options:
- **Public repo:** add only datasets cleared for publication, under `tests/data/`.
- **Unpublished data:** keep it in a separate private repo (e.g. `nmr-dashboard-testdata`). CI can fetch it with a token, or it can sit in a gitignored local folder.

## Open items to confirm during build
- The curated list of CPMG parameters to display (Tyler, from sample data).
- Exact `acqu.par` key names for acqTime, dwell time, and point count (check against a real file before writing the FT path).
- Default ILT α and T2 range for the UI (`run_kea_ilt.py` uses α = 1e8 and [1e-4, 10] s).
