"""Nonnegative 2D (T2-D) inverse Laplace transform.

`run_2dilt` is a port of MeldrumLabCode/pythonic/run_2dilt.py (itself a port of ILT Scripts/run_T2DILT.m), kept as it is:
a single-orientation, SVD-compressed Kronecker-product solve with curvature regularisation, via scipy's lsq_linear. See that
file's docstring for how it deviates from MATLAB. Here it keeps the original unit convention: D in units of 1e-9 m^2/s and
b-values in units of 1e-9 s/m^2 (so b*D is unchanged and the numbers stay near 1). `run` wraps it so callers work in m^2/s.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import lsq_linear

from .ilt import build_kernel, build_tau_grid, second_derivative_matrix

D_SCALE = 1e9                      # processed D = D[m^2/s] * 1e9; converted back for every result
EXPONENT_FOR_TIME = 2.6            # measured: solve time grows ~ (unknowns)^2.6 (750 -> 1.9 s, 1250 -> 8 s, 2500 -> 54 s)
TRIAL_CORRECTION = 1.65            # the small timed trial under-predicts big solves by this factor (5.1 s predicted vs 8.3 s measured at 50 x 25)
PEAK_MIN_FRACTION = 0.10           # 2D regions are those above this fraction of the spectrum maximum


def build_kernel_diffusion(b_vec, D_grid):
    """Eb[i, j] = exp(-b_i * D_j): linear in b, unlike the T2 kernel."""
    return np.exp(-np.outer(b_vec, D_grid))


def run_2dilt(data, echo_vec, b_vec, params, alpha=None):
    """Nonnegative 2D ILT. data (n_echo, n_bval); params T2minmax/T2_steps/Dminmax/D_steps(/alpha).

    Returns spectrum (n_D, n_T2), T2_grid, D_grid, pred (n_bval, n_echo), resid, rmse.
    """
    data = np.asarray(data, dtype=float)
    echo_vec = np.asarray(echo_vec, dtype=float).ravel()
    b_vec = np.asarray(b_vec, dtype=float).ravel()

    if alpha is None:
        if "alpha" not in params:
            raise ValueError("Alpha must be provided either as argument or in params.")
        alpha = params["alpha"]
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")

    T2_min, T2_max = map(float, params["T2minmax"])
    n_T2 = int(min(params["T2_steps"], len(echo_vec)))
    D_min, D_max = map(float, params["Dminmax"])
    n_D = int(min(params["D_steps"], len(b_vec)))

    T2_grid = build_tau_grid(T2_min, T2_max, n_T2)
    D_grid = build_tau_grid(D_min, D_max, n_D)

    pas = (np.log10(T2_max) - np.log10(T2_min)) / (n_T2 - 1)
    pasdata = (np.log10(echo_vec[-1]) - np.log10(echo_vec[0])) / (len(echo_vec) - 1)
    weighting = 1.0 / np.sqrt(pasdata**2 * pas**3 * alpha)

    Ea = build_kernel(echo_vec, T2_grid)            # (n_echo, n_T2)
    Eb = build_kernel_diffusion(b_vec, D_grid)       # (n_bval, n_D)

    ua, _, _ = np.linalg.svd(Ea, full_matrices=False)
    ub, _, _ = np.linalg.svd(Eb, full_matrices=False)
    Ea_c = ua.T @ Ea
    Eb_c = ub.T @ Eb

    data_bt = data.T
    data_c = ub.T @ data_bt @ ua

    Xdim = n_T2 * n_D
    E_c = np.kron(Ea_c, Eb_c)
    b_flat = data_c.flatten(order="F")

    D2 = second_derivative_matrix(Xdim)
    A = np.vstack([E_c, weighting * D2])
    b_aug = np.concatenate([b_flat, np.zeros(Xdim - 2)])

    result = lsq_linear(A, b_aug, bounds=(0, np.inf), method="trf")
    spectrum = result.x.reshape((n_D, n_T2), order="F")

    pred = Eb @ spectrum @ Ea.T
    resid = data_bt - pred
    rmse = np.sqrt(np.mean(resid**2))
    return spectrum, T2_grid, D_grid, pred, resid, rmse


# ── everything below is new: unit handling, summaries, time estimate ─────────────────────────

def grid_sizes(n_echo, n_b, t2_steps, d_steps):
    """The grid the solver will really use (it never has more points than echoes / b-values)."""
    return int(min(t2_steps, n_echo)), int(min(d_steps, n_b))


def estimate_seconds(n_T2, n_D, seconds_per_unit):
    """Rough solve time from the grid size, scaled by a timed trial on this machine (see trial_seconds)."""
    return TRIAL_CORRECTION * seconds_per_unit * (n_T2 * n_D) ** EXPONENT_FOR_TIME


def trial_seconds(echo_vec, b_vec):
    """Time a small solve here and return (seconds per unknown^2.6): the machine's speed for this solver."""
    import time
    n = 12
    e = np.asarray(echo_vec)[np.linspace(0, len(echo_vec) - 1, 60).astype(int)]
    bb = np.asarray(b_vec)[np.linspace(0, len(b_vec) - 1, min(len(b_vec), 10)).astype(int)]
    rng = np.random.default_rng(0)
    data = np.abs(rng.normal(1, 0.1, (len(e), len(bb))))
    nd = min(len(bb), n)
    t0 = time.perf_counter()
    run_2dilt(data, e, bb, {"T2minmax": [1e-5, 1.0], "T2_steps": n, "Dminmax": [1e-3, 10.0], "D_steps": nd, "alpha": 1e9})
    return (time.perf_counter() - t0) / (n * nd) ** EXPONENT_FOR_TIME


def sensitive_d_window(b_vec):
    """The diffusion coefficients (m^2/s) the b-values can tell apart, as a rule of thumb: [1/b_max, 5/b_min].

    b_vec is in the solver's units (1e-9 s/m^2), so b = b_vec * 1e9 s/m^2. A species with D far above 5/b_min has decayed to
    nothing before the first b-value, and one far below 1/b_max has barely decayed by the last, so the inversion cannot place
    either: whatever the regularised solution puts there is extrapolation, not measured signal.
    """
    b = np.asarray(b_vec, float) * D_SCALE
    return 1.0 / b.max(), 5.0 / max(b.min(), 1e-300)


def run(data, echo_vec, b_vec, t2_min, t2_max, t2_steps, d_min, d_max, d_steps, alpha, sigma=None, peak_min_fraction=PEAK_MIN_FRACTION):
    """The 2D ILT with D in m^2/s on the way in and the way out (the solver itself runs on D * 1e9)."""
    spec, T2, Dscaled, pred, resid, rmse = run_2dilt(
        data, echo_vec, b_vec, {"T2minmax": [t2_min, t2_max], "T2_steps": t2_steps, "Dminmax": [d_min * D_SCALE, d_max * D_SCALE], "D_steps": d_steps}, alpha=alpha)
    D = Dscaled / D_SCALE
    rss = float(np.sum(resid ** 2))
    n_points = resid.size
    rng_ = float(np.ptp(pred))
    lo, hi = sensitive_d_window(b_vec)
    peaks = peaks_2d(spec, T2, D, peak_min_fraction)
    inside = (D >= lo) & (D <= hi)
    in_window_total = float(spec[inside].sum())
    labels, n_regions = ndimage.label(spec >= peak_min_fraction * spec.max()) if spec.max() > 0 else (None, 0)
    # regions come back sorted by size, so re-find each one's mask from its position of maximum
    for pk in peaks:
        pk["constrained"] = bool(lo <= pk["D_mean"] <= hi)           # False: the data carry no information at this D
        # the part of this region that lies inside the window, as a share of everything inside the window (None if there is nothing inside it)
        i, j = int(np.argmin(np.abs(D - pk["D_peak"]))), int(np.argmin(np.abs(T2 - pk["T2_peak"])))
        mask = labels == labels[i, j]
        pk["fraction_in_window"] = float(spec[mask & inside[:, None]].sum() / in_window_total) if in_window_total > 0 else None
    return {
        "spectrum": spec, "T2": T2, "D": D, "pred": pred, "resid": resid,
        "echo_s": np.asarray(echo_vec, float), "b_vec": np.asarray(b_vec, float),
        "alpha": float(alpha), "n_T2": len(T2), "n_D": len(D), "t2_min": t2_min, "t2_max": t2_max, "d_min": d_min, "d_max": d_max,
        "rss": rss, "mse": rss / n_points, "rmse": float(rmse), "signal_range": rng_, "rmse_over_amp": float(rmse) / rng_ if rng_ > 0 else None,
        "noise_sigma": sigma, "chi2_per_point": rss / (sigma ** 2 * n_points) if sigma else None,
        "projection_T2": spec.sum(axis=0), "projection_D": spec.sum(axis=1),
        "peaks": peaks,
        "d_sensitive": [lo, hi],
        "fraction_in_sensitive_window": float(spec[inside].sum() / spec.sum()) if spec.sum() > 0 else None,
    }


def peaks_2d(spec, T2, D, min_fraction=PEAK_MIN_FRACTION):
    """Connected regions of the spectrum above min_fraction of its maximum.

    Each region: its share of the total signal, the amplitude-weighted geometric-mean T2 and D, and the position of its
    maximum. Regions that touch are one region (so two overlapping peaks can be reported as one).
    """
    spec = np.asarray(spec, float)
    total = spec.sum()
    if total <= 0 or spec.max() <= 0:
        return []
    labels, n = ndimage.label(spec >= min_fraction * spec.max())
    lnT, lnD = np.log(T2)[None, :], np.log(D)[:, None]
    out = []
    for k in range(1, n + 1):
        m = labels == k
        w = spec * m
        s = w.sum()
        i, j = np.unravel_index(np.argmax(np.where(m, spec, -np.inf)), spec.shape)
        out.append({"fraction": float(s / total), "T2_mean": float(np.exp((w * lnT).sum() / s)), "D_mean": float(np.exp((w * lnD).sum() / s)),
                    "T2_peak": float(T2[j]), "D_peak": float(D[i]), "height": float(spec[i, j])})
    return sorted(out, key=lambda p: -p["fraction"])
