"""Thin entry points called by the browser worker (and usable from notebooks)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from .display_params import build_display
from .kea_io import _autophase, read_kea_binary, read_kea_params


def hello():
    return {
        "nmrdash": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sum": float(np.arange(10).sum()),
    }


def _clean(a):
    return np.nan_to_num(np.asarray(a, dtype=float)).tolist()


def check_cpmg(params, header):
    """Return (is_cpmg, reason). Only complex 2D echo-train data is supported for now."""
    if header is None:
        return False, "no data.2d found (only data.1d or none)"
    if "echoTime" not in params:
        return False, "acqu.par has no echoTime"
    exp = str(params.get("experiment", ""))
    if exp and exp.upper() != "CPMG":
        return False, f'experiment is "{exp}", not CPMG'
    n_echo = int(params.get("nrEchoes", header["yDim"]))
    if header["yDim"] != n_echo:
        return False, f"data.2d has {header['yDim']} echoes but acqu.par says {n_echo}"
    return True, ""


def load(dirpath, timestamp_ms=None, fallback_to_mtime=True):
    """Read acqu.par + data.2d from dirpath; return a JSON-ready dict for the UI.

    timestamp_ms : file modification time in ms since epoch (from the browser's
        File.lastModified). If None: with fallback_to_mtime the data file's mtime
        on disk is used (desktop use); otherwise no timestamp is shown (the browser's
        in-memory files have no meaningful mtime, e.g. for bundled sample data).
    """
    d = Path(dirpath)
    par_path = d / "acqu.par"
    if not par_path.is_file():
        raise FileNotFoundError("acqu.par not found in the selected folder")
    params = read_kea_params(par_path)

    header = spec = None
    data_path = d / "data.2d"
    if data_path.is_file():
        header, spec = read_kea_binary(data_path)

    if timestamp_ms is not None:
        ts = datetime.fromtimestamp(timestamp_ms / 1000)
    elif fallback_to_mtime and data_path.is_file():
        ts = datetime.fromtimestamp(data_path.stat().st_mtime)
    else:
        ts = None

    is_cpmg, reason = check_cpmg(params, header)
    out = {
        "version": __version__,
        "display": build_display(params, ts),
        "params_all": {k: (v if isinstance(v, (str, int, float, list)) else str(v)) for k, v in params.items()},
        "is_cpmg": is_cpmg,
        "reason": reason,
        "raw": None,
    }
    if is_cpmg:
        n_pts, n_echo = header["xDim"], header["yDim"]
        td = spec[: n_pts * n_echo].reshape((n_pts, n_echo), order="F")
        echo_s = 1e-6 * float(params["echoTime"]) * np.arange(1, n_echo + 1)
        if np.iscomplexobj(td):
            summed = _autophase(td).real.sum(axis=0)
            re, im = td.real, td.imag
        else:
            summed, re, im = td.sum(axis=0), td, np.zeros_like(td)
        out["raw"] = {
            "n_points": n_pts,
            "n_echoes": n_echo,
            "echo_time_s": _clean(echo_s),
            "real": _clean(re),   # [point][echo]
            "imag": _clean(im),
            "summed": _clean(summed),
        }
    return out
