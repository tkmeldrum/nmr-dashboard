"""The synthetic sample datasets in test_data/ (made by tools/make_synthetic_data.py) and what is in them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from make_synthetic_data import TRUTH, write_kea_binary   # noqa: E402,F401  (the values the datasets were made from)

DATA = ROOT / "test_data"

TECMAG_CPMG_CLEAN, TECMAG_CPMG_SNR100 = DATA / "tecmag_cpmg_clean.tnt", DATA / "tecmag_cpmg_snr100.tnt"
TECMAG_T1_CLEAN, TECMAG_T1_SNR100 = DATA / "tecmag_t1_clean.tnt", DATA / "tecmag_t1_snr100.tnt"
TECMAG_T2D_CLEAN, TECMAG_T2D_SNR100 = DATA / "tecmag_t2d_clean.tnt", DATA / "tecmag_t2d_snr100.tnt"
CPMG_CLEAN, CPMG_SNR100 = DATA / "kea_cpmg_clean", DATA / "kea_cpmg_snr100"
T1_CLEAN, T1_SNR100 = DATA / "kea_t1_clean", DATA / "kea_t1_snr100"
T2D_CLEAN, T2D_SNR100 = DATA / "kea_t2d_clean", DATA / "kea_t2d_snr100"


def t2d_with_a_slow_population(d):
    """A copy of the noisy T2-D set with a third population too slow for these b-values to resolve (D = 2.5e-13 m^2/s, below about 3.5e-13): it can only be 'bound' at the low limit."""
    import numpy as np
    from make_synthetic_data import write_t2d
    write_t2d(d, True, np.random.default_rng(7), "kea_t2d_slow_population", T2_s=(0.004, 0.012, 0.04), D_m2s=(1.5e-10, 2e-11, 2.5e-13), fraction=(0.3, 0.3, 0.4))
    return d


def cpmg_single_exponential(d, T2=0.02, noisy=True):
    """A copy of the CPMG set with one T2 everywhere (a decay that a second exponential cannot be told apart from)."""
    import numpy as np
    from make_synthetic_data import write_cpmg
    write_cpmg(d, noisy, np.random.default_rng(3), "kea_cpmg_single", T2_s=(T2,), fraction=(1.0,), layers_um=((-120.0, 120.0),))
    return d


def tecmag_variant(path, kind, noisy=True, **override):
    """A Tecmag file of the given kind ('cpmg', 't1', 't2d') with entries of its TRUTH replaced (and the writer's keyword options kept at their defaults)."""
    import numpy as np
    from make_synthetic_data import write_tecmag_cpmg, write_tecmag_t1, write_tecmag_t2d
    write = {"cpmg": write_tecmag_cpmg, "t1": write_tecmag_t1, "t2d": write_tecmag_t2d}[kind]
    write(path, noisy, np.random.default_rng(11), path.name, **override)
    return path


TECMAG_SERIES_CLEAN, TECMAG_SERIES_SNR30 = DATA / "tecmag_cpmg_power_series_clean.tnt", DATA / "tecmag_cpmg_power_series_snr30.tnt"

# every kind at signal-to-noise ratios of 30 and 10 (SNR100 is the one most tests use); see tools/make_synthetic_data.py for how the SNR is defined
SNR_LEVELS = (100, 30, 10)
KEA = {(kind, snr): DATA / f"kea_{kind}_snr{snr}" for kind in ("cpmg", "t1", "t2d") for snr in SNR_LEVELS}
TECMAG = {(kind, snr): DATA / f"tecmag_{kind}_snr{snr}.tnt" for kind in ("cpmg", "t1", "t2d") for snr in SNR_LEVELS}
