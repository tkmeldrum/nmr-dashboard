"""Instrument and nuclear constants (defaults from MeldrumLabCode/pythonic/constants.py)."""

GAMMA_MHZ_PER_T = {"1H": 42.577, "19F": 40.078}   # gyromagnetic ratio / 2π
DEFAULT_G_T_PER_M = 23.7                          # static gradient, T/m
DEFAULT_ZERO_FILL = 4                             # FFT length = nextpow2(nrPnts * 2**zf)
DEFAULT_Z_WINDOW_UM = (-200.0, 200.0)             # sample region; + z = + frequency = closer to the magnet (down)

# ASSUMPTION to confirm: b1Freq = 19.44 MHz in acqu.par is the 1H Larmor frequency of this magnet
# (B0 = 19.44 / 42.577 = 0.4566 T), so 19F would sit at 19.44 * 40.078 / 42.577 = 18.30 MHz.
REF_B1FREQ_1H_MHZ = 19.44
B1FREQ_TOLERANCE_MHZ = 0.3


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

# Saturation recovery: Kea's recovery curves do not start at zero (an offset of a few % of the plateau is typical),
# and a fit without it is much worse on the sample data (RMSE 305 vs 38), so the offset is on by default.
T1_DEFAULT_OFFSET = True
