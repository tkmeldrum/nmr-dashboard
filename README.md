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
4. **Uncertainty**: confidence level for the fit intervals (68 / 95 / 99%), and optional **MCMC** and **bootstrap** on the
   whole decay or on one z position. They run after the processing, on the Results tab, once you have seen a timed
   estimate and confirmed; a progress bar and a **Stop (keep partial results)** button are shown while they run.
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
| MCMC | `nmrdash/uncertainty.py`, [emcee](https://emcee.readthedocs.io) ensemble sampler on the multi-exponential model. Sampled: amplitudes (flat, ≥ 0), ln T₂ (log-uniform, ordered fast→slow so components cannot swap), offset, optionally ln σ; σ otherwise fixed from the imaginary channel. Output: credible intervals, trace and corner plots, acceptance, autocorrelation time and effective samples (with warnings if the chain is under ~50 τ), a posterior-predictive band on the decay |
| Bootstrap | refits surrogate decays (fitted curve + resampled noise): the multi-exponential model (warm-started) and/or the ILT. Noise source: the **measured imaginary-channel noise** (preferred, model-free), the fit residuals, or Gaussian with the measured σ. Output: percentile intervals, parameter histograms, an ILT band, per-peak T₂ and fraction intervals |
| Statistics | RSS, MSE, residual SD, **residual SD / fitted signal range** and **reduced χ²** (independent of amplitude and echo count), noise σ taken from the imaginary channel after phasing (model-free), ΔAIC/ΔBIC (raw AIC/BIC only mean something as differences) |

Conventions: times are in seconds; value ± uncertainty pairs share one base exponent (`1.99e-2 ± 0.35e-2`); T₂ axes
use decade ticks (`1e-5, 1e-4, …`).

## Assumptions to confirm

- **¹H vs ¹⁹F** is preset from `b1Freq`: **19.44 MHz is ¹H on this magnet** (confirmed; so ¹⁹F ≈ 18.30 MHz). One constant:
  `REF_B1FREQ_1H_MHZ` in `nmrdash/constants.py`. The lab has a **second magnet** run by a non-Kea (Tecmag) spectrometer; when
  Tecmag reading is added, that magnet's reference frequency must be set separately. No ¹⁹F dataset has been tested.
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
nmrdash/        the Python package (also runs on a desktop: pip install -e .; its command is `nmrdash`)
  kea_io.py echoes.py multiexp.py ilt.py ilt_tools.py spatial.py pipeline.py constants.py display_params.py
  uncertainty.py (MCMC, bootstrap)   params.py (the params file)   runner.py cli.py (run, batch)
  outputs.py figures.py formatting.py (results on disk)   provenance.py export.py
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
pip install -e ".[test]"        # includes emcee, used by the MCMC tests
pytest                          # ~90 s: runs every sample dataset through the pipeline
python tools/dev_server.py      # http://localhost:8000  (generates the manifests on the fly)
python tools/build_site.py site # what CI deploys; versions every asset URL so browsers never serve stale code
```

Deploys: every push to `main` runs the tests, builds the site and publishes it to GitHub Pages
(`.github/workflows/pages.yml`).

**Test data is public.** Anything under `test_data/` is in a public repo and is bundled into the live site (only
`acqu.par` and `data.2d` are shipped). Commit only data you are happy to publish.

## Status and roadmap

**Done:** load and raw display; MCMC and bootstrap (branch `uncertainty`, not yet merged); command-line tool, params file, batch/time-series and provenance (branch `script-export`, built on top of it); integrate and FT; multi-exponential fit with intervals and model comparison; ILT with peak
tables and α sweep; spatially resolved fits with a linked cross-section viewer; fit-quality statistics; tabbed workflow;
sample-data explorer; CI and Pages deploy.

**Next (see `docs/PLAN.md` for the original phasing):**

1. **PED analysis** (deferred at Tyler's request): port `PED/t2_analysis_real_data.py` (`PED_LBFGS`, `T2Analyzer`) from PyTorch to numpy/scipy L-BFGS,
   keeping the regularisers (None, L1, L2, PED, ElasticNet, SQRLasso), with a **parity test against the PyTorch
   original** (needs PyTorch locally; not run in CI).
2. ~~Uncertainty tab: MCMC and bootstrap~~ **done on branch `uncertainty`** (see the pull request); still to do: bootstrap
   bands for PED spectra once PED exists, and MCMC/bootstrap at *every* z position (today: one position at a time).
3. **Plot controls and saving**: common plot settings (axis scale and limits, time units, fonts, line widths, colormap,
   figure size, a "no title" option for publication figures). Save numeric results (CSV) and figures (PNG/SVG/PDF,
   rendered with matplotlib) into a `processed_<timestamp>/` folder inside the data folder (Chrome/Edge via the File
   System Access API, or another folder the user picks); other browsers get a `.zip`. A `processing.json` records
   every setting plus the `nmrdash` version for reproducibility. *Needs testing in a real Chrome.*
4. ~~Export a processing script (batch and time series)~~ **done on branch `script-export`** (see the pull request). Still to do:
   import a params file into the page, a time-resolved T₂(z, t) map for FT batches, and a SLURM example.
5. **Tecmag data** (second magnet, own Python reader exists): add a reader beside `kea_io.py` and check the Larmor frequency.
6. Smaller items: an option to skip or down-weight a distorted first echo; non-CPMG data (T1 saturation recovery and
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

- **From the page:** the **Export params + script** button (Process and Uncertainty tabs) downloads a `params.toml` holding
  exactly the settings currently on the page, plus `process_nmr.py`, a short runner (`python process_nmr.py DATA_FOLDER`, add
  `--batch` for a folder of datasets). Use `defaults` in place of a params file to run with the built-in defaults.
- **Params file** (`nmrdash/params.py` is the single definition): sections `processing`, `ft`, `multiexp`, `ilt`,
  `uncertainty`, `mcmc`, `bootstrap`, `output`, `batch`; every setting is commented in the generated file, unknown names and
  out-of-range values are rejected with a message naming the setting, and `"auto"` means the same rule the page uses.
- **Where results go:** a `processed_<timestamp>` folder **inside each data folder** (raw data is never touched; re-running
  never overwrites, it adds `-2`, `-3`, …), or under `output.directory` / `--out`. It holds CSV tables (decay, fit components
  with intervals, fit quality, model comparison, ILT spectrum / peaks / α sweep, FT profile and per-position tables, MCMC and
  bootstrap summaries and samples), figures (`png`/`pdf`/`svg`; titles can be switched off), `summary.md` (readable, same
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
a normal browser. Not tested at all: the real File System Access save path, Safari/Firefox, ¹⁹F data, and the `emcee` download from the real PyPI inside a real browser (in tests the same files are served from a local cache).
