"""Run the decay analyses at every spatial position of an FT'd CPMG dataset.

Output is column-oriented (one array per quantity, indexed by position) so the UI can plot
T2(z), amplitude(z) and the ILT map directly. The multi-exponential fits and the ILT use their
own position sets: exponential fits are expensive, so they sit at the natural resolution by
default, while the (cheap) ILT map covers every zero-filled grid point so heat maps look smooth.
"""
import numpy as np

from . import ilt_tools, multiexp
from .ilt import run_1dilt

FAST_STARTS = 5            # multi-exp starting guesses per position (the bulk fit uses more)
NEGLIGIBLE_BAND = 0.01     # a band holding <1% of a position's signal gets no mean T2 there


def _gmean(tau, spec):
    tot = spec.sum()
    return float(np.exp(np.sum(spec * np.log(tau)) / tot)) if tot > 0 else None


def _unpack(pos):
    """(z, Y[, keep[, sigma]]) -> (z, Y, keep, sigma); positions with keep False (no signal) are not fitted."""
    z, Y = pos[0], pos[1]
    keep = np.asarray(pos[2], bool) if len(pos) > 2 and pos[2] is not None else np.ones(len(z), bool)
    sigma = np.asarray(pos[3], float) if len(pos) > 3 and pos[3] is not None else np.full(len(z), np.nan)
    return z, Y, keep, sigma


def _fit_multiexp(t, Y, z, keep, sig, me, progress, done0, total):
    npos, n = len(z), int(me["n"])
    nan = lambda *shape: np.full(shape, np.nan)
    m = {"n": n, "offset": bool(me.get("offset", True)), "ci_level": float(me.get("ci_level", 0.95)), "z_um": z,
         "T2": nan(n, npos), "T2_se": nan(n, npos), "A": nan(n, npos), "A_se": nan(n, npos),
         "fraction": nan(n, npos), "mean_T2": nan(npos), "mean_T2_se": nan(npos),
         "total_amplitude": nan(npos), "rmse": nan(npos), "chi2_red": nan(npos),
         "ok": np.zeros(npos, bool), "flag": np.zeros(npos, bool)}
    done = done0
    for i in range(npos):
        if not keep[i]:
            continue
        try:
            f = multiexp.fit(t, Y[i], n, m["offset"], m["ci_level"], n_starts=FAST_STARTS,
                             sigma=float(sig[i]) if np.isfinite(sig[i]) else None)
            for k, c in enumerate(f["components"]):
                m["T2"][k, i], m["T2_se"][k, i] = c["T2"], c["T2_se"]
                m["A"][k, i], m["A_se"][k, i] = c["A"], c["A_se"]
                m["fraction"][k, i] = c["fraction"]
            m["mean_T2"][i], m["mean_T2_se"][i] = f["mean_T2"], f["mean_T2_se"]
            m["total_amplitude"][i], m["rmse"][i] = f["total_amplitude"], f["rmse"]
            m["chi2_red"][i] = f["chi2_red"] if f["chi2_red"] is not None else np.nan
            m["ok"][i], m["flag"][i] = True, bool(f["warnings"])
        except (ValueError, RuntimeError, FloatingPointError):
            pass
        done += 1
        if progress:
            progress(done, total)
    return m


def _fit_ilt(t, Y, z, keep, sig, il, peak_min_fraction, progress, done0, total):
    npos = len(z)
    params = {"ILTminmax": [float(il["t2_min"]), float(il["t2_max"])],
              "nrILTSteps": min(int(il["n_steps"]), ilt_tools.STEPS_LIMIT)}
    spectra, rmse, tau = [], np.full(npos, np.nan), None
    done = done0
    for i in range(npos):
        if not keep[i]:
            spectra.append(None)
            continue
        try:
            spec, tau, _, _, r = run_1dilt(Y[i], t, params, alpha=float(il["alpha"]))
            spectra.append(spec)
            rmse[i] = r
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            spectra.append(None)
        done += 1
        if progress:
            progress(done, total)
    good = [s for s in spectra if s is not None]
    width = len(good[0]) if good else 0
    S = np.array([s if s is not None else np.full(width, np.nan) for s in spectra]).reshape(npos, width)

    mean_T2 = np.array([_gmean(tau, s) or np.nan if s is not None else np.nan for s in spectra], float)
    # One band per peak of the position-summed spectrum, so each band follows the same feature of the
    # heat map at every position; per position: geometric-mean T2 within the band and its signal share.
    bands = ilt_tools.band_edges(tau, np.nansum(S, axis=0), peak_min_fraction) if good else []
    out_bands, b_mean, b_frac = [], [], []
    for pk, lo, hi in bands:
        out_bands.append({"t2_lo": float(tau[lo]), "t2_hi": float(tau[hi - 1]), "t2_peak": float(tau[pk])})
        means, fracs = np.full(npos, np.nan), np.full(npos, np.nan)
        for i, s in enumerate(spectra):
            if s is None or s.sum() <= 0:
                continue
            seg = s[lo:hi]
            fracs[i] = seg.sum() / s.sum()
            if fracs[i] >= NEGLIGIBLE_BAND:
                means[i] = _gmean(tau[lo:hi], seg)
        b_mean.append(means)
        b_frac.append(fracs)
    return {"z_um": z, "tau": tau, "alpha": float(il["alpha"]), "n_steps": width, "spectra": S,
            "mean_T2": mean_T2, "rmse": rmse, "bands": out_bands, "band_mean_T2": b_mean, "band_fraction": b_frac}


def fit_positions(t, me_pos=None, il_pos=None, me=None, il=None, progress=None, peak_min_fraction=0.05):
    """me_pos / il_pos are (z_um, Y[n_positions, n_echoes][, keep]) for the exponential fits and the ILT.

    Positions with keep False (signal below the threshold) are skipped and come back as NaN/None, so
    noise-only positions do not produce meaningless fits. progress(done, total) counts fitted positions.
    """
    do_me = bool(me and me.get("enabled") and me_pos is not None)
    do_il = bool(il and il.get("enabled") and il_pos is not None)
    mz, mY, mk, msig = _unpack(me_pos) if do_me else (None,) * 4
    iz, iY, ik, isig = _unpack(il_pos) if do_il else (None,) * 4
    total = (int(mk.sum()) if do_me else 0) + (int(ik.sum()) if do_il else 0)
    out = {"multiexp": None, "ilt": None}
    done = 0
    if do_me:
        out["multiexp"] = _fit_multiexp(t, mY, mz, mk, msig, me, progress, done, total)
        done += int(mk.sum())
    if do_il:
        out["ilt"] = _fit_ilt(t, iY, iz, ik, isig, il, peak_min_fraction, progress, done, total)
    return out
