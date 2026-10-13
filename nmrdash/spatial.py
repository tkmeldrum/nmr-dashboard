"""Run the decay analyses at every spatial position of an FT'd CPMG dataset.

Output is column-oriented (one array per quantity, indexed by position) so the UI can plot
T2(z), amplitude(z) and the ILT map directly.
"""
import numpy as np

from . import ilt_tools, multiexp
from .ilt import run_1dilt

FAST_STARTS = 5     # multi-exp starting guesses per position (the bulk fit uses more)


def _gmean_t2(tau, spectrum):
    tot = spectrum.sum()
    return float(np.exp(np.sum(spectrum * np.log(tau)) / tot)) if tot > 0 else None


def fit_positions(t, Y, z_um, me=None, il=None, progress=None):
    """Y: (n_positions, n_echoes) decays at positions z_um. me/il are the multiexp / ilt settings dicts."""
    npos = len(z_um)
    out = {"z_um": z_um, "multiexp": None, "ilt": None}
    do_me = bool(me and me.get("enabled"))
    do_il = bool(il and il.get("enabled"))

    if do_me:
        n = int(me["n"])
        nan = lambda *shape: np.full(shape, np.nan)
        m = {"n": n, "offset": bool(me.get("offset", True)), "ci_level": float(me.get("ci_level", 0.95)),
             "T2": nan(n, npos), "T2_se": nan(n, npos), "A": nan(n, npos), "A_se": nan(n, npos),
             "fraction": nan(n, npos), "mean_T2": nan(npos), "mean_T2_se": nan(npos),
             "total_amplitude": nan(npos), "rmse": nan(npos),
             "ok": np.zeros(npos, bool), "flag": np.zeros(npos, bool)}
    if do_il:
        n_steps = min(int(il["n_steps"]), ilt_tools.STEPS_LIMIT)
        params = {"ILTminmax": [float(il["t2_min"]), float(il["t2_max"])], "nrILTSteps": n_steps}
        spectra, gm, peak, rm = [], np.full(npos, np.nan), np.full(npos, np.nan), np.full(npos, np.nan)
        tau = None

    for i in range(npos):
        y = Y[i]
        if do_me:
            try:
                f = multiexp.fit(t, y, n, m["offset"], m["ci_level"], n_starts=FAST_STARTS)
                for k, c in enumerate(f["components"]):
                    m["T2"][k, i], m["T2_se"][k, i] = c["T2"], c["T2_se"]
                    m["A"][k, i], m["A_se"][k, i] = c["A"], c["A_se"]
                    m["fraction"][k, i] = c["fraction"]
                m["mean_T2"][i], m["mean_T2_se"][i] = f["mean_T2"], f["mean_T2_se"]
                m["total_amplitude"][i], m["rmse"][i] = f["total_amplitude"], f["rmse"]
                m["ok"][i], m["flag"][i] = True, bool(f["warnings"])
            except (ValueError, RuntimeError, FloatingPointError):
                pass
        if do_il:
            try:
                spec, tau, _, _, r = run_1dilt(y, t, params, alpha=float(il["alpha"]))
                spectra.append(spec)
                gm[i], rm[i] = _gmean_t2(tau, spec) or np.nan, r
                peak[i] = tau[int(np.argmax(spec))] if spec.max() > 0 else np.nan
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                spectra.append(None)
        if progress:
            progress(i + 1, npos)

    if do_me:
        out["multiexp"] = m
    if do_il:
        good = [s for s in spectra if s is not None]
        width = len(good[0]) if good else 0
        out["ilt"] = {"tau": tau, "alpha": float(il["alpha"]), "n_steps": width,
                      "spectra": [s if s is not None else np.full(width, np.nan) for s in spectra],
                      "mean_T2": gm, "peak_T2": peak, "rmse": rm}
    return out
