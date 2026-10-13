"""Loading Kea T2-D (diffusion-T2, sequence SSE-T2) data.

Adapted from MeldrumLabCode/pythonic/kea_t2d_io.py. Files in the data folder:
    acqu.par       nrEchoes, nrPntsTau, echoTime (us), DELTA (ms)
    data2D_Cp.2d   complex echo trains, (nrEchoes x nrPntsTau); data2D_Re.2d / data2D_Im.2d hold the two parts separately
    data2D.dat     plain-text alternative (real), (nrPntsTau x nrEchoes)
    data.dat       tau (ms), b-value (x1e-9 s/m^2), per-tau amplitude
    T2Axis.dat     echo times (ms), DiffAxis.dat  b-values (x1e-9 s/m^2): written by Kea, preferred over recomputing them
The imaginary channel after phasing holds only noise, which gives a measured noise sigma (as for CPMG data).
"""
from pathlib import Path

import numpy as np

from . import echoes
from .kea_io import read_kea_binary

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

    tau_ms = b_dat = None
    if (d / "data.dat").is_file():
        a = np.loadtxt(d / "data.dat", ndmin=2)
        tau_ms, b_dat = a[:, 0], a[:, 1]
    if (d / "DiffAxis.dat").is_file():
        b_vec = _axis(d / "DiffAxis.dat")
    elif b_dat is not None:
        b_vec = b_dat
    else:
        raise ValueError("neither DiffAxis.dat nor data.dat (b-values) was found")
    if len(b_vec) != n_tau:
        raise ValueError(f"{len(b_vec)} b-values but acqu.par says {n_tau} delays")
    echo_s = _axis(d / "T2Axis.dat") * 1e-3 if (d / "T2Axis.dat").is_file() else 1e-6 * float(params["echoTime"]) * np.arange(1, n_echo + 1)
    if len(echo_s) != n_echo:
        raise ValueError(f"{len(echo_s)} echo times but acqu.par says {n_echo} echoes")
    if np.any(np.diff(echo_s) <= 0) or np.any(echo_s <= 0):
        raise ValueError("the echo times must be positive and increasing")
    if np.any(b_vec < 0):
        raise ValueError("the b-values must not be negative")
    return {"data": data, "noise": noise, "echo_s": echo_s, "tau_s": None if tau_ms is None else tau_ms * 1e-3, "b_vec": b_vec, "complex": noise is not None}


def noise_sigma(noise, skip=0):
    """Noise SD from the imaginary channel of the included echoes (all delays together); None for real-only data."""
    if noise is None:
        return None
    return echoes.noise_sigma(np.asarray(noise)[skip:].ravel())
