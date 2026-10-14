"""The physical relations this tool relies on, written to be read and checked on their own (see docs/PLAN_PHYSICS.md).

Only NumPy; nothing here fits, loads, plots or knows about the rest of the package. Each function says what it models, in what
units, under which assumptions, and where the relation comes from. (The signal models themselves are being moved in here too.)
"""
import numpy as np

# Gyromagnetic ratio / 2*pi, in Hz per tesla.
GAMMA_HZ_PER_T = {"1H": 42.577e6, "19F": 40.078e6}


# The static field gradient of the magnet, T/m: used for the position axis of the profile (z = f / gamma G) and for the diffusion weighting b.
# It is a property of the magnet, so it is a setting you can change, not something read from the data files.
DEFAULT_GRADIENT_T_PER_M = 23.67


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
