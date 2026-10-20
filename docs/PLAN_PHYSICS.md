# Plan: the physics in one place, listed on the About page, and checkable by a reviewer

**Status:** decided and built (physics.py and inference.py, the About page section, the tests; PR #33), after the b-values (PR #31). The decisions below are recorded as made.

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
| b-values `b = (γG)² τ² (Δ + 2τ/3)`, the τ grid, the gradient | **now calculated** from `gradient` (a setting, default 23.67 T/m), `DELTA`, `tauMin`, `tauMax`, `nrPntsTau`: `physics.py` and `t2d.py` (PR #31). The spectrometer's axis files are only a cross-check. |

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

## A physical effect the tool does not model (to decide whether to state or to include)

In a static gradient, diffusion also attenuates the **CPMG echo train** itself, so the decay rate is `1/T₂,app = 1/T₂ + γ²G² D tₑ² / 12` (tₑ the echo time). At G = 23.67 T/m:

| echo time | `γ²G²tₑ²/12` (s⁻¹ per m²/s of D) | for D = 10⁻⁹ m²/s (free water) | for D = 10⁻¹¹ |
|---|---|---|---|
| 44 µs (SSE-T₂) | 6.5×10⁹ | 6.5 s⁻¹ | 0.07 s⁻¹ |
| 200 µs (CPMG) | 1.3×10¹¹ | 130 s⁻¹ | 1.3 s⁻¹ |

So for solids and slow diffusers (D ≲ 10⁻¹¹) it is negligible, but for a free liquid at long echo times it can shorten the apparent T₂ appreciably. Today T₂ in the fits is the *apparent* T₂. Options: state this on the About page (cheap), or add an optional correction when D is known (e.g. from the T₂–D result).

## Decisions (made)

1. **Two files** (physics and inference).
2. **Sources:** drafted for the About page; each reference is to be confirmed by the author before the tool is announced beyond the lab.
3. **b-values:** calculated from the gradient, Δ and τ (PR #31); the spectrometer's files only cross-check. The gradient is a setting, default 23.67 T/m.
4. **No in-browser self-check button:** the About page shows the formulas and links to the code; the checks live in `tests/`.
5. **Diffusion during the CPMG echo train:** stated as a footnote on the About page, not corrected by default.
