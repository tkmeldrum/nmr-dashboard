"""Loading Kea T2-D (diffusion-T2, sequence SSE-T2) data.

Adapted from MeldrumLabCode/pythonic/kea_t2d_io.py. Files in the data folder:
    acqu.par       nrEchoes, nrPntsTau, echoTime (us), DELTA (ms)
    data2D_Cp.2d   complex echo trains, (nrEchoes x nrPntsTau); data2D_Re.2d / data2D_Im.2d hold the two parts separately
    data2D.dat     plain-text alternative (real), (nrPntsTau x nrEchoes)
    data.dat       tau (ms), b-value (x1e-9 s/m^2), per-tau amplitude
    T2Axis.dat     echo times (ms), DiffAxis.dat  b-values (x1e-9 s/m^2): written by Kea; used only to cross-check the calculated axes
The imaginary channel after phasing holds only noise, which gives a measured noise sigma (as for CPMG data).
"""
from pathlib import Path

import numpy as np

from . import constants, echoes, physics
from .kea_io import read_kea_binary

AXIS_AGREEMENT = 0.01        # the calculated axes and the spectrometer's files may differ by this fraction before the page warns
DATA_FILES = ("data2D_Cp.2d", "data2D.dat", "data.dat", "T2Axis.dat", "DiffAxis.dat")


def _axis(path):
    return np.loadtxt(path).ravel()


def load(d, params):
    """Return dict: data (n_echo, n_tau) phased real part, noise (same shape or None), echo_s, tau_s, b_vec, complex (bool)."""
    d = Path(d)
    n_echo, n_tau = int(params["nrEchoes"]), int(params["nrPntsTau"])
    cp, txt = d / "data2D_Cp.2d", d / "data2D.dat"
    if cp.is_file():
        _hdr, spec = read_kea_binary(cp)
        if spec.size < n_echo * n_tau:
            raise ValueError(f"data2D_Cp.2d holds {spec.size} values but acqu.par says {n_echo} echoes x {n_tau} delays")
        raw = spec[: n_echo * n_tau].reshape((n_echo, n_tau), order="F")
    elif txt.is_file():
        arr = np.loadtxt(txt)
        if arr.shape != (n_tau, n_echo):
            raise ValueError(f"data2D.dat has shape {arr.shape}, expected ({n_tau}, {n_echo}) from acqu.par")
        raw = arr.T
    else:
        raise ValueError("neither data2D_Cp.2d nor data2D.dat was found next to acqu.par")

    if np.iscomplexobj(raw):
        theta = -np.angle(raw[:, 0].sum())                 # one phase from the first delay's echo train, applied to all (as in pythonic)
        rot = raw * np.exp(1j * theta)
        data, noise = rot.real, rot.imag
    else:
        data, noise = np.asarray(raw, float), None

    # The axes are calculated from the acquisition parameters (physics.py), not taken from the spectrometer's files; those files, when
    # present, are only compared against the calculation (check_axes). The gradient is a setting, so b is calculated when processing.
    need = ("DELTA", "tauMin", "tauMax", "nrPntsTau", "echoTime")
    missing = [k for k in need if k not in params]
    if missing:
        raise ValueError("acqu.par has no " + ", ".join(missing) + ", which the diffusion delays and b-values are calculated from (DELTA, tauMin, tauMax, nrPntsTau) "
                         "and the echo times from (echoTime)")
    nucleus, _why = constants.guess_nucleus(params.get("b1Freq"))
    tau_s = physics.geometric_grid(float(params["tauMin"]) * 1e-3, float(params["tauMax"]) * 1e-3, n_tau)
    echo_s = 1e-6 * float(params["echoTime"]) * np.arange(1, n_echo + 1)
    files = {"tau": [], "b": [], "echo": []}                                                                    # (file name, the spectrometer's values)
    if (d / "DiffAxis.dat").is_file():
        files["b"].append(("DiffAxis.dat", _axis(d / "DiffAxis.dat")))
    if (d / "data.dat").is_file():
        a = np.loadtxt(d / "data.dat", ndmin=2)
        if a.shape[0] == n_tau and a.shape[1] >= 2:
            files["tau"].append(("data.dat", a[:, 0] * 1e-3))
            files["b"].append(("data.dat", a[:, 1] * 1e9))
    if (d / "T2Axis.dat").is_file():
        files["echo"].append(("T2Axis.dat", _axis(d / "T2Axis.dat") * 1e-3))
    kea_g = physics.gradient_implied_by_kea(params["gradient"]) if "gradient" in params else None
    info = {"nucleus": nucleus, "delta_ms": float(params["DELTA"]), "tau_min_ms": float(params["tauMin"]), "tau_max_ms": float(params["tauMax"]), "n_tau": n_tau,
            "echo_time_us": float(params["echoTime"]), "kea_gradient_khz_per_mm": float(params["gradient"]) if kea_g is not None else None, "kea_gradient_t_per_m": kea_g}
    if np.any(np.diff(echo_s) <= 0) or np.any(echo_s <= 0):
        raise ValueError("the echo times must be positive and increasing")
    L = {"data": data, "noise": noise, "echo_s": echo_s, "tau_s": tau_s, "delta_s": float(params["DELTA"]) * 1e-3, "complex": noise is not None, "b_info": info, "files": files}
    L["b_vec"] = b_values(L, physics.DEFAULT_GRADIENT_T_PER_M)                 # at the default gradient: what the Raw data tab shows; processing recalculates it
    L["b_check"] = check_axes(L, physics.DEFAULT_GRADIENT_T_PER_M)
    return L


def b_values(L, gradient_t_per_m):
    """The diffusion weighting of every delay, in the solver's units (b in 1e9 s/m^2), for a gradient in T/m."""
    return physics.b_value_stimulated_echo(L["tau_s"], L["delta_s"], gradient_t_per_m, L["b_info"]["nucleus"]) * 1e-9


def check_axes(L, gradient_t_per_m):
    """Compare the calculated delays, echo times and b-values with the spectrometer's own files (when they are there).

    Delays and echo times should match to a small fraction. The b-values depend on the gradient, which is a setting here, so they are
    compared for *shape* (the same tau and Delta dependence: the ratio file/calculated must be constant) and the scale is reported as the
    gradient the spectrometer's values correspond to.
    """
    b = b_values(L, gradient_t_per_m) * 1e9
    out = []
    for what, key, ours in (("delays tau", "tau", L["tau_s"]), ("echo times", "echo", L["echo_s"])):
        for fname, theirs in L["files"][key]:
            ok = len(theirs) == len(ours)
            out.append({"what": what, "file": fname, "kind": "match", "value": float(np.max(np.abs(ours / np.where(theirs == 0, np.nan, theirs) - 1))) if ok else None,
                        "agrees": bool(ok and np.max(np.abs(ours / np.where(theirs == 0, np.nan, theirs) - 1)) <= AXIS_AGREEMENT)})
    for fname, theirs in L["files"]["b"]:
        if len(theirs) != len(b) or np.any(theirs <= 0):
            out.append({"what": "b-values", "file": fname, "kind": "shape", "value": None, "agrees": False})
            continue
        r = theirs / b
        shape = float(np.max(np.abs(r / r.mean() - 1)))
        out.append({"what": "b-values", "file": fname, "kind": "shape", "value": shape, "agrees": shape <= AXIS_AGREEMENT,
                    "implied_gradient_t_per_m": float(gradient_t_per_m * np.sqrt(r.mean())), "b_ratio": float(r.mean())})
    return {"compared": out, "agrees": all(c["agrees"] for c in out), "tolerance": AXIS_AGREEMENT, "gradient_t_per_m": float(gradient_t_per_m)}


def noise_sigma(noise, skip=0):
    """Noise SD from the imaginary channel of the included echoes (all delays together); None for real-only data."""
    if noise is None:
        return None
    return echoes.noise_sigma(np.asarray(noise)[skip:].ravel())
