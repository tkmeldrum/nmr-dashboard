"""Run the ILT and summarise it: peaks, area fractions, and an optional alpha sweep."""
import numpy as np
from scipy.signal import find_peaks

from .ilt import run_1dilt, second_derivative_matrix

PEAK_MIN_FRACTION = 0.05   # default: peaks below 5% of the tallest are ignored
DEFAULT_MAX_STEPS = 100    # default for the user-editable 'max grid points' (run_1dilt also caps at the number of echoes)
STEPS_LIMIT = 1000         # safety limit on a typed-in value
DEFAULT_SWEEP = (4, 12)    # alpha sweep, as powers of ten


def summarize_peaks(tau, spectrum, min_fraction=PEAK_MIN_FRACTION):
    """[{T2_peak, T2_mean, fraction}] with one entry per resolved peak.

    Each peak owns the grid points between the valleys on either side; T2_mean is the
    amplitude-weighted geometric mean over that region, fraction its share of the total signal.
    """
    spectrum = np.asarray(spectrum, float)
    total = spectrum.sum()
    if total <= 0:
        return []
    peaks, _ = find_peaks(spectrum, height=min_fraction * spectrum.max())
    if len(peaks) == 0:
        return []
    cuts = [0] + [int(a + np.argmin(spectrum[a:b + 1])) for a, b in zip(peaks[:-1], peaks[1:])] + [len(spectrum)]
    out = []
    for pk, lo, hi in zip(peaks, cuts[:-1], cuts[1:]):
        seg = spectrum[lo:hi]
        out.append({
            "T2_peak": float(tau[pk]),
            "T2_mean": float(np.exp(np.sum(seg * np.log(tau[lo:hi])) / seg.sum())),
            "fraction": float(seg.sum() / total),
        })
    return out


def band_edges(tau, spectrum, min_fraction=PEAK_MIN_FRACTION):
    """Partition the T2 grid into one band per peak of `spectrum`: [(peak_idx, lo, hi)], hi exclusive.

    Boundaries sit at the valleys between neighbouring peaks, so the bands tile the whole grid.
    Pass the position-summed spectrum to get bands that stay fixed across positions.
    """
    spectrum = np.asarray(spectrum, float)
    if spectrum.sum() <= 0:
        return []
    peaks, _ = find_peaks(spectrum, height=min_fraction * spectrum.max())
    if len(peaks) == 0:
        return []
    cuts = [0] + [int(a + np.argmin(spectrum[a:b + 1])) for a, b in zip(peaks[:-1], peaks[1:])] + [len(spectrum)]
    return [(int(pk), lo, hi) for pk, lo, hi in zip(peaks, cuts[:-1], cuts[1:])]


def run(t, y, t2_min, t2_max, n_steps, alpha, sweep=False,
        peak_min_fraction=PEAK_MIN_FRACTION, sweep_range=DEFAULT_SWEEP, sigma=None):
    """n_steps is a maximum: the grid is also capped at the number of echoes."""
    n_steps = min(int(n_steps), STEPS_LIMIT)
    spectrum, tau, pred, resid, rmse = run_1dilt(
        y, t, {"ILTminmax": [t2_min, t2_max], "nrILTSteps": n_steps}, alpha=alpha)
    rss = float(np.sum(np.asarray(resid) ** 2))
    amp = float(np.ptp(pred))                           # fitted signal range over the sampled echoes (as in multiexp)
    out = {
        "rss": rss, "mse": rss / len(t), "signal_range": amp,
        "rmse_over_amp": float(rmse) / amp if amp > 0 else None,
        "noise_sigma": sigma,
        # the regularised fit has no clean parameter count, so this is chi-square per point, not reduced chi-square
        "chi2_per_point": rss / (sigma ** 2 * len(t)) if sigma else None,
        "alpha": alpha, "t2_min": t2_min, "t2_max": t2_max, "n_steps": len(tau),
        "tau": tau, "spectrum": spectrum, "pred": pred, "resid": resid, "rmse": float(rmse),
        "peaks": summarize_peaks(tau, spectrum, peak_min_fraction), "sweep": None,
    }
    if sweep:
        D2 = second_derivative_matrix(len(tau))
        rows = []
        for a in np.logspace(sweep_range[0], sweep_range[1], 9):
            s, _, _, _, r = run_1dilt(y, t, {"ILTminmax": [t2_min, t2_max], "nrILTSteps": n_steps}, alpha=a)
            norm = np.linalg.norm(s)
            rows.append({"alpha": float(a), "rmse": float(r),
                         "roughness": float(np.linalg.norm(D2 @ s) / norm) if norm > 0 else 0.0})
        out["sweep"] = rows
    return out
