"""The physics this tool relies on, written to be read and checked on their own.

What the NMR signal is assumed to be (relaxation, diffusion), the kernels the inversions use, and the geometry of the static-gradient
profile. Only NumPy; nothing here fits, loads, plots or knows about the rest of the package. How numbers are *estimated* from these
models (noise, least squares, information criteria, regularisation) is in inference.py.

Each function says what it models, in what units, under which assumptions, and where the relation comes from. MODELS at the bottom
lists them; the About page is generated from that list, with links to these lines.
"""
import numpy as np

# Gyromagnetic ratio / 2*pi, in Hz per tesla.
GAMMA_HZ_PER_T = {"1H": 42.577e6, "19F": 40.078e6}

# The static field gradient of the magnet, T/m: used for the position axis of the profile (z = f / gamma G) and for the diffusion weighting b.
# It is a property of the magnet, so it is a setting you can change, not something read from the data files.
DEFAULT_GRADIENT_T_PER_M = 23.67


# ── relaxation ───────────────────────────────────────────────────────────────

def decay_factor(x):
    """exp(-x): the basic decay, with x = t / T. (Vectorised over x; used by the fits and the MCMC.)"""
    return np.exp(-x)


def recovery_factor(x):
    """1 - exp(-x): the basic recovery of the longitudinal magnetisation toward equilibrium, with x = t / T1."""
    return 1.0 - np.exp(-x)


def exp_decay(t, T2):
    """Transverse relaxation of one population: exp(-t / T2), unit amplitude at t = 0.

    t and T2 in the same unit (s). Assumes mono-exponential decay (a single, homogeneous relaxation time).
    In a static gradient, the echo train is also attenuated by diffusion (see MODELS, "not modelled"); T2 here is the apparent T2.
    Source: Bloch (1946); for the echo train, Carr and Purcell (1954) and Meiboom and Gill (1958).
    """
    return decay_factor(np.asarray(t, float) / np.asarray(T2, float))


def multi_exp(t, A, T2, offset=0.0):
    """Several populations relaxing independently: S(t) = sum_i A_i exp(-t / T2_i) + c.

    A (amplitudes, signal units), T2 (s), offset c (signal units: a baseline). Assumes the populations do not exchange magnetisation
    on the time scale of the measurement (slow exchange), so the decay is a plain sum.
    Source: standard multi-component relaxation analysis, e.g. Whittall and MacKay (1989).
    """
    t = np.asarray(t, float)
    y = np.zeros_like(t, dtype=float)
    for a, T in zip(np.atleast_1d(A), np.atleast_1d(T2)):
        y += a * decay_factor(t / T)
    return y + offset


def saturation_recovery(t, T1):
    """Recovery of the longitudinal magnetisation after saturation: 1 - exp(-t / T1), zero at t = 0, one at long times.

    t and T1 in the same unit (s). Assumes complete saturation by the pulse train and a single T1.
    Source: standard (e.g. Fukushima and Roeder, Experimental Pulse NMR, 1981).
    """
    return recovery_factor(np.asarray(t, float) / np.asarray(T1, float))


def multi_saturation(t, A, T1, offset=0.0):
    """Several T1 populations: S(t) = sum_i A_i (1 - exp(-t / T1_i)) + c. (Amplitude A_i is the recovered signal of population i.)"""
    t = np.asarray(t, float)
    y = np.zeros_like(t, dtype=float)
    for a, T in zip(np.atleast_1d(A), np.atleast_1d(T1)):
        y += a * recovery_factor(t / T)
    return y + offset


# ── diffusion ────────────────────────────────────────────────────────────────

def diffusion_attenuation(b, D):
    """Signal attenuation by free (Gaussian) diffusion: exp(-b D), with b in s/m^2 and D in m^2/s (only the product matters).

    Assumes unrestricted diffusion with one coefficient per population (no restriction, no distribution of D within a population).
    Source: Stejskal and Tanner (1965); Tanner (1970) for the stimulated echo.
    """
    return np.exp(-np.asarray(b, float) * np.asarray(D, float))


def gradient_implied_by_kea(gradient_khz_per_mm):
    """The gradient (T/m) that Kea's `gradient` parameter in acqu.par corresponds to: used only to cross-check, never to calculate.

    ASSUMPTION (it reproduces the spectrometer's own b-values on the sample data to 0.03%, see tests/test_physics.py): the parameter is the
    proton frequency gradient (gamma_H G / 2 pi) in kHz/mm, so G = gradient * 1e6 Hz/m / gamma_H; gradient = 1000 gives 23.49 T/m.
    """
    return float(gradient_khz_per_mm) * 1e6 / GAMMA_HZ_PER_T["1H"]


def geometric_grid(first, last, n):
    """n values from `first` to `last`, equally spaced in the logarithm (how Kea spaces the diffusion delays tau)."""
    return np.geomspace(float(first), float(last), int(n))


def b_value_stimulated_echo(tau_s, delta_s, gradient_t_per_m_, nucleus="1H"):
    """Diffusion weighting b (s/m^2) of a stimulated echo in a static field gradient.

        b = (gamma G)^2 * tau^2 * (Delta + 2 tau / 3),     gamma = 2 pi * GAMMA_HZ_PER_T[nucleus]

    tau   : the encoding delay (s) between the first and second 90-degree pulses, and again between the third pulse and the echo
    Delta : the storage (mixing) time (s) between the second and third pulses
    G     : the static gradient (T/m)

    The signal is attenuated by exp(-b D) for a diffusion coefficient D (m^2/s). Assumes: a constant gradient, free (Gaussian)
    diffusion, instantaneous pulses (no correction for their length) and no contribution from the CPMG echo train that follows
    (its own diffusion weighting acts on the apparent T2, not on b).
    Source: Hahn (1950) and Tanner (1970) for diffusion in a stimulated echo; the form above is the standard static-gradient
    result (e.g. Callaghan, Principles of Nuclear Magnetic Resonance Microscopy, 1991).
    """
    gamma = 2.0 * np.pi * GAMMA_HZ_PER_T[nucleus]
    tau = np.asarray(tau_s, float)
    return (gamma * float(gradient_t_per_m_)) ** 2 * tau ** 2 * (float(delta_s) + 2.0 * tau / 3.0)


def resolvable_diffusion_window(b_solver_units):
    """The diffusion coefficients (m^2/s) a set of b-values can tell apart, as a rule of thumb: [1/b_max, 5/b_min].

    b_solver_units is b in units of 1e9 s/m^2 (the solver's units). A population with D far above 5/b_min has decayed to nothing before the
    first b-value, and one far below 1/b_max has barely decayed by the last, so the data cannot place either: whatever a regularised
    solution puts there is extrapolation, not measured signal. A rule of thumb (exp(-5) < 1 % and exp(-1) is still 37 %), not a sharp limit.
    """
    b = np.asarray(b_solver_units, float) * 1e9
    return 1.0 / b.max(), 5.0 / max(b.min(), 1e-300)


# ── the signal of the T2-D experiment ────────────────────────────────────────

def t2d_factors(t, b, T2, D):
    """The two factors of every population of the T2-D signal: E[:, k] = exp(-t / T2_k) (n_t x K) and G[:, k] = exp(-b D_k) (n_b x K).

    t (s), b (s/m^2), T2 (s), D (m^2/s), each population k with its own T2_k and D_k.
    """
    return exp_decay(np.asarray(t, float)[:, None], np.asarray(T2, float)[None, :]), diffusion_attenuation(np.asarray(b, float)[:, None], np.asarray(D, float)[None, :])


def t2d_signal(t, b, A, T2, D):
    """The T2-D signal of K independent populations, an (n_t x n_b) matrix:

        S(t, b) = sum_k A_k exp(-t / T2_k) exp(-b D_k),     A_k >= 0

    Each population relaxes with its own T2 (during the echo train) and diffuses with its own D (during the diffusion encoding), and the
    two effects separate because they happen in different parts of the sequence. No offset. Assumes free diffusion and a single T2
    per population. Source: the stimulated-echo / CPMG combination of Tanner (1970) and Carr and Purcell (1954); the 2D relaxation-diffusion
    correlation experiment of Hurlimann and Venkataramanan (2002).
    """
    E, G = t2d_factors(t, b, T2, D)
    return (E * np.asarray(A, float)[None, :]) @ G.T


# ── the kernels of the inversions ────────────────────────────────────────────

def t2_kernel(t, T2_grid):
    """The matrix K[i, j] = exp(-t_i / T2_j) that maps a T2 distribution f(T2_j) to a decay: y_i = sum_j K[i, j] f_j.

    t (s) echo times, T2_grid (s). This is the Laplace kernel the 1D inversion inverts. Matches the construction in nnlsmooth1Dsvd.m (Matlab).
    """
    return np.exp(-np.outer(np.asarray(t, float), 1.0 / np.asarray(T2_grid, float)))


def diffusion_kernel(b, D_grid):
    """The matrix K[i, j] = exp(-b_i D_j): linear in b, unlike the T2 kernel. b and D_grid in any units whose product is dimensionless
    (the solver passes b in 1e9 s/m^2 and D x 1e9).

    The 2D (T2-D) inversion inverts the separable kernel K2D = kron(t2_kernel, diffusion_kernel), applied to the data without forming
    the full matrix (ilt2d.run_2dilt compresses each factor with its SVD first; the compression does not change the problem).
    Source: Venkataramanan, Song and Hurlimann (2002); Song et al. (2002).
    """
    return np.exp(-np.outer(np.asarray(b, float), np.asarray(D_grid, float)))


# ── position along the gradient ──────────────────────────────────────────────

def z_um_from_frequency(f_hz, gamma_mhz_per_t, gradient_t_per_m_):
    """Position (micrometres) from the offset frequency (Hz) in a static gradient: z = f / (gamma G).

    gamma in MHz/T (gamma / 2 pi) and G in T/m give f / (MHz/m) = micrometres directly. Positive z is positive frequency, i.e. toward
    the magnet. Source: the definition of frequency encoding (Callaghan, 1991).
    """
    return np.asarray(f_hz, float) / (gamma_mhz_per_t * gradient_t_per_m_)


def native_resolution_um(n_points, dwell_us, gamma_mhz_per_t, gradient_t_per_m_):
    """Spatial resolution (micrometres) set by the acquisition window: delta_z = 1 / (N dwell gamma G).

    N points at `dwell` (us) span a frequency resolution 1 / (N dwell) in Hz; dividing by gamma G converts it to length. Zero-filling
    interpolates but adds no resolution, so neighbouring z points of the FFT are not independent on finer scales than this.
    """
    return 1e6 / (n_points * dwell_us) / (gamma_mhz_per_t * gradient_t_per_m_)


# ── the list the About page is generated from ────────────────────────────────

# Each entry: the model, its equation (plain text), what it describes, where it comes from, and the functions that implement it.
MODELS = [
    {"id": "exp_decay", "name": "Exponential T₂ decay", "equation": "S(t) = A · exp(−t / T₂)",
     "describes": "Transverse relaxation of one population, measured as the decay of the echo amplitudes in a CPMG train.",
     "source": "Carr and Purcell, Phys. Rev. 94, 630 (1954); Meiboom and Gill, Rev. Sci. Instrum. 29, 688 (1958)",
     "functions": [exp_decay, decay_factor]},
    {"id": "multi_exp", "name": "Multi-exponential decay", "equation": "S(t) = Σᵢ Aᵢ · exp(−t / T₂ᵢ) + c",
     "describes": "Several populations relaxing independently (slow exchange), plus a baseline c. The multi-exponential fit.",
     "source": "Whittall and MacKay, J. Magn. Reson. 84, 134 (1989)", "functions": [multi_exp]},
    {"id": "saturation", "name": "Saturation recovery (T₁)", "equation": "S(t) = Σᵢ Aᵢ · (1 − exp(−t / T₁ᵢ)) + c",
     "describes": "Recovery of the longitudinal magnetisation after saturation, for one or two T₁ populations.",
     "source": "Fukushima and Roeder, Experimental Pulse NMR: A Nuts and Bolts Approach (Addison-Wesley, 1981)",
     "functions": [saturation_recovery, multi_saturation, recovery_factor]},
    {"id": "diffusion", "name": "Diffusion attenuation", "equation": "exp(−b · D)",
     "describes": "Attenuation of the signal by free diffusion with coefficient D at diffusion weighting b.",
     "source": "Stejskal and Tanner, J. Chem. Phys. 42, 288 (1965); Tanner, J. Chem. Phys. 52, 2523 (1970)", "functions": [diffusion_attenuation]},
    {"id": "b_value", "name": "Diffusion weighting b of the stimulated echo", "equation": "b = (γG)² · τ² · (Δ + 2τ/3)",
     "describes": "How strongly each diffusion delay τ weights diffusion, for a stimulated echo with storage time Δ in a static gradient G. "
                  "Calculated from the gradient setting, Δ and τ in acqu.par (the echo time does not enter).",
     "source": "Hahn, Phys. Rev. 80, 580 (1950); Tanner, J. Chem. Phys. 52, 2523 (1970); Callaghan, Principles of Nuclear Magnetic Resonance Microscopy (Oxford, 1991)",
     "functions": [b_value_stimulated_echo, geometric_grid, gradient_implied_by_kea]},
    {"id": "t2d", "name": "T₂–D signal", "equation": "S(t, b) = Σₖ Aₖ · exp(−t / T₂ₖ) · exp(−b · Dₖ),  Aₖ ≥ 0",
     "describes": "Several populations, each with its own T₂ and D (the few-population T₂–D model; no offset).",
     "source": "Combines Tanner (1970) and Carr and Purcell (1954); Hürlimann and Venkataramanan, J. Magn. Reson. 157, 31 (2002)",
     "functions": [t2d_signal, t2d_factors]},
    {"id": "kernels", "name": "Inversion kernels", "equation": "K₁D[i, j] = exp(−tᵢ / T₂ⱼ);  K_D[i, j] = exp(−bᵢ Dⱼ);  K₂D = K₁D ⊗ K_D",
     "describes": "The matrices the regularised inversions invert to obtain a T₂ distribution (1D) or a T₂–D map (2D, separable).",
     "source": "Venkataramanan, Song and Hürlimann, IEEE Trans. Signal Process. 50, 1017 (2002); Song et al., J. Magn. Reson. 154, 261 (2002)",
     "functions": [t2_kernel, diffusion_kernel]},
    {"id": "resolvable", "name": "Range of D the b-values can resolve", "equation": "1 / b_max  ≤  D  ≤  5 / b_min",
     "describes": "A rule of thumb for which diffusion coefficients the measured b-values can tell apart; outside it a regularised map is extrapolation.",
     "source": "Rule of thumb (exp(−5) < 1 %; exp(−1) = 37 %), not a sharp limit", "functions": [resolvable_diffusion_window]},
    {"id": "position", "name": "Position along the gradient", "equation": "z = f / (γ·G);   Δz = 1 / (N · dwell · γ·G)",
     "describes": "Frequency to position for the static-gradient profile (FT of each echo), and the spatial resolution set by the acquisition window.",
     "source": "Frequency encoding in a static gradient (Callaghan, 1991)", "functions": [z_um_from_frequency, native_resolution_um]},
]

# What is not modelled (stated on the About page).
NOT_MODELLED = [
    "Restricted diffusion, or a distribution of D within a population (each population has one free-diffusion coefficient).",
    "Non-exponential relaxation within a population, and exchange of magnetisation between populations (the populations are independent).",
    "Finite pulse lengths and imperfect refocusing pulses in the diffusion encoding and in the echo train.",
    "Diffusion during the CPMG echo train in the static gradient. It adds γ²G²D·tₑ²/12 to the decay rate (tₑ is the echo time), so the fitted T₂ is the "
    "apparent one. It is negligible for solids and slow diffusers and can matter for free liquids at long echo times (for D = 10⁻⁹ m²/s and G = 23.67 T/m: "
    "about 6.5 s⁻¹ at tₑ = 44 µs, about 130 s⁻¹ at 200 µs). The tool does not correct for it; only users who know their D and need the true T₂ should.",
]
