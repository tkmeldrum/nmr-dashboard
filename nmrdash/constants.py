"""Instrument and nuclear constants (defaults from MeldrumLabCode/pythonic/constants.py)."""

from . import physics

GAMMA_MHZ_PER_T = {n: g / 1e6 for n, g in physics.GAMMA_HZ_PER_T.items()}   # gyromagnetic ratio / 2π, from the physics module
DEFAULT_G_T_PER_M = physics.DEFAULT_GRADIENT_T_PER_M                          # static gradient, T/m (the Kea magnet); Tecmag data: see tecmag.GRADIENT_T_PER_M


def default_gradient(params):
    """The static gradient to start from for a dataset: this lab's Tecmag magnet (about 6.7 T/m) or the Kea magnet (23.67 T/m). It is a setting; the file does not hold it."""
    return 6.7 if params.get("source") == "tecmag" else DEFAULT_G_T_PER_M
DEFAULT_ZERO_FILL = 4                             # FFT length = nextpow2(nrPnts * 2**zf)
DEFAULT_Z_WINDOW_UM = (-200.0, 200.0)             # sample region; + z = + frequency = closer to the magnet (down)

# ASSUMPTION to confirm: b1Freq = 19.44 MHz in acqu.par is the 1H Larmor frequency of this magnet
# (B0 = 19.44 / 42.577 = 0.4566 T), so 19F would sit at 19.44 * 40.078 / 42.577 = 18.30 MHz.
REF_B1FREQ_1H_MHZ = 19.44
B1FREQ_TOLERANCE_MHZ = 0.3


def nucleus_of(params):
    """(nucleus, reason) for a dataset: the nucleus the file names (Tecmag headers do), else the one nearest b1Freq (Kea)."""
    if params.get("nucleus") in GAMMA_MHZ_PER_T:
        return params["nucleus"], f"{ {'1H': '¹H', '19F': '¹⁹F'}[params['nucleus']] } from the file header"
    return guess_nucleus(params.get("b1Freq"))


def guess_nucleus(b1freq_mhz):
    """Return (nucleus, reason) from acqu.par's b1Freq: whichever nucleus is nearest in Larmor frequency."""
    if b1freq_mhz is None:
        return "1H", "acqu.par has no b1Freq; defaulting to ¹H"
    expect = {n: REF_B1FREQ_1H_MHZ * g / GAMMA_MHZ_PER_T["1H"] for n, g in GAMMA_MHZ_PER_T.items()}
    nuc = min(expect, key=lambda n: abs(expect[n] - b1freq_mhz))
    label = {"1H": "¹H", "19F": "¹⁹F"}[nuc]
    if abs(expect[nuc] - b1freq_mhz) > B1FREQ_TOLERANCE_MHZ:
        return "1H", f"b1Freq {b1freq_mhz:g} MHz matches neither ¹H nor ¹⁹F on this magnet; defaulting to ¹H"
    return nuc, f"b1Freq {b1freq_mhz:g} MHz → {label}"

# T2-D (SSE-T2). D is entered and reported in m^2/s; the solver works on D * 1e9 (see ilt2d.py).
T2D_DEFAULTS = {"gradient": physics.DEFAULT_GRADIENT_T_PER_M, "t2_min": 1e-5, "t2_max": 1.0, "t2_steps": 50, "d_min": 1e-12, "d_max": 2.2e-9, "d_steps": 25, "alpha": 1e9, "peak_threshold_pct": 10.0}

# Saturation recovery: Kea's recovery curves do not start at zero (an offset of a few % of the plateau is typical),
# and a fit without it is much worse on the sample data (RMSE 305 vs 38), so the offset is on by default.
T1_DEFAULT_OFFSET = True
