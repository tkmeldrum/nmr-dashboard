# Plan: the physics in one place, listed on the About page, and checkable by a reviewer

**Status:** plan, for review. Nothing here is built yet.

## The goal

A scientific reviewer should be able to open **one short file**, read the signal models the tool assumes, and judge each directly ("yes, that is the
decay I mean"), without wading through fitting, plotting or page code. The About page should list those models, say what each describes and where it comes
from, and link to the exact lines. Tests should check each model against independent calculations.

## Where the physics is today (scattered)

| Model or relation | Where it lives now |
|---|---|
| T₂ decay `exp(−t/T₂)`, multi-exponential `Σ Aᵢ exp(−t/T₂ᵢ) [+ c]` | `multiexp.py` (`KERNELS`, `model`) |
| Saturation recovery `1 − exp(−t/T₁)` | `multiexp.py` (`KERNELS["saturation"]`) |
| 1D ILT kernel `K[i,j] = exp(−tᵢ/τⱼ)` | `ilt.py` (`build_kernel`) |
| Diffusion factor `exp(−b D)` and the 2D ILT kernel (a Kronecker product of the two) | `ilt2d.py` (`build_kernel_diffusion`, `run_2dilt`) |
| Few-population T₂–D model `Σ Aₖ exp(−t/T₂ₖ) exp(−b Dₖ)` | `t2d_model.py` (`basis`, `predict`) |
| Position from frequency `z = f / (γ G)`; resolution `Δz = 1 / (N Δt γ G)`; nucleus γ | `echoes.py` (`ft`, `native_resolution_um`), `constants.py` |
| Echo phasing and integration | `echoes.py` (`integrate`) |
| Which diffusion coefficients the b-values can resolve, `[1/b_max, 5/b_min]` | `ilt2d.py` (`sensitive_d_window`) |
| **b-values themselves** | **not computed here**: read from the spectrometer's `DiffAxis.dat` / `data.dat` (units 10⁻⁹ s/m²) |

Several of these are the same relation written more than once (the exponential decay appears in three places).

## Proposal

### 1. `nmrdash/physics.py`: the signal models and measurement relations, and nothing else

- **Only models.** Plain functions of numbers and arrays; imports NumPy and nothing from the rest of the package. No fitting, no I/O, no plotting,
  no regularisation. About 200 lines, so it can be read in one sitting.
- **Each function states, in its docstring:** the equation, every symbol with units, the assumptions, and where the relation comes from.
- **Single source of truth.** `multiexp`, `ilt`, `ilt2d`, `t2d_model`, `echoes` and `uncertainty` call these functions; the duplicates are deleted. What a
  reviewer reads is what runs.
- **A registry** at the bottom of the file lists the models (name, equation as text, what it describes, source, function). The About page is generated
  from it, so the list and the code cannot drift apart.

Contents:

| Function | Models |
|---|---|
| `exp_decay(t, T2)` | `exp(−t/T₂)` |
| `multi_exp(t, A, T2, offset=0)` | `Σ Aᵢ exp(−t/T₂ᵢ) + c` (CPMG) |
| `saturation_recovery(t, T1)` and `multi_saturation(...)` | `1 − exp(−t/T₁)` and its sums (T₁) |
| `diffusion_attenuation(b, D)` | `exp(−b D)` |
| `t2d_signal(t, b, A, T2, D)` | `Σ Aₖ exp(−t/T₂ₖ) exp(−b Dₖ)` |
| `t2_kernel(t, tau)`, `diffusion_kernel(b, D)` | the matrices the inversions use, built from the same factors; the 2D kernel is their Kronecker product |
| `z_from_frequency(f, gamma, G)`, `native_resolution(n, dwell, gamma, G)` | position axis and resolution of the static-gradient profile |
| `resolvable_diffusion_window(b)` | `[1/b_max, 5/b_min]`, with the rule of thumb stated as such |

### 2. `nmrdash/inference.py`: how numbers are estimated from the models (second file, optional second review)

Noise from the imaginary channel (1.4826 × MAD), the least-squares and χ² objectives, AIC/BIC, Student-*t* and delta-method intervals, the Gaussian
log-likelihood and priors used by MCMC, and the Tikhonov regularisation of the ILT. Kept apart because a reviewer checks *what is modelled* (physics) and
*how it is estimated* (statistics) separately.

### 3. The About page lists them

A "Physics models in this tool" section, generated from the registry at build time: a table of **model, equation, what it describes, where it comes from, and a
link to the lines** (pinned to the deployed commit, so the link shows exactly the code that is running). It also states what the tool does *not* model
(for instance restricted diffusion or non-exponential relaxation) and that **the b-values are supplied by the spectrometer, not computed here**.

### 4. Checks a reviewer can read and run

`tests/test_physics.py`: each model against an independent calculation, never against itself:
- exponential decay: value `1/e` at `t = T₂`, half-life `T₂ ln 2`, monotone, `1` at `t = 0`; multi-exponential equals the sum of single ones; saturation recovery `0` at `0` and `→ 1`;
- T₂–D signal factorises for one population and adds for several; the 1D and 2D kernel matrices equal the model evaluated on the grids; the 2D kernel is exactly the Kronecker product;
- `z = f/(γG)` against a hand calculation (1 kHz, ¹H, 23.7 T/m → 0.99 µm) and the resolution relation;
- the resolvable window against the definition.
Optionally the About page gets a **"Run these checks in your browser"** button that runs the same checks in Pyodide on the deployed code.

### 5. A short reviewer's guide

`docs/PHYSICS.md`: the same table with fuller derivations and references, the assumptions, and how to run the checks.

## Order of work

1. Create `physics.py` with the registry and move the existing formulas in (behaviour unchanged; the full existing suite must stay green and numerically identical).
2. `tests/test_physics.py`, with the independent checks.
3. The About section generated from the registry, with source links; a build test that every registered model exists and is listed.
4. `inference.py` (a second, separate step).
5. The in-browser check button (optional).

## Decisions for you

1. **Two files (physics and inference), or one?** Recommended: two.
2. **Sources.** For each model, which reference should the About page cite? I can supply standard ones (Carr–Purcell–Meiboom–Gill; Venkataramanan et al. 2002 for the inversion; the static-gradient position relation) but you should confirm each, and tell me the reference for the SSE-T₂ diffusion weighting your Kea sequence uses.
3. **b-values.** They come from Kea's axis file. Do you want an *independent cross-check* that recomputes b from the acquisition parameters (needs the formula for your sequence), shown as a warning if they disagree?
4. **Browser check button.** Worth it?
