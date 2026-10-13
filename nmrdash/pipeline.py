"""Thin entry points called by the browser worker (and usable from notebooks)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from . import echoes, ilt_tools, multiexp
from .display_params import build_display
from .kea_io import read_kea_binary, read_kea_params

# The dataset currently loaded in this interpreter (one per browser tab / notebook session).
_STATE = {}


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
    _STATE.clear()
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
        summed, _phase = echoes.integrate(td)
        re, im = (td.real, td.imag) if np.iscomplexobj(td) else (td, np.zeros_like(td))
        _STATE.clear()
        _STATE.update(td=td, echo_s=echo_s, params=params)
        out["raw"] = {
            "n_points": n_pts,
            "n_echoes": n_echo,
            "echo_time_s": _clean(echo_s),
            "real": _clean(re),   # [point][echo]
            "imag": _clean(im),
            "summed": _clean(summed),
        }
    return out


def _jsonable(o):
    """numpy -> plain Python, NaN/inf -> None (strict JSON)."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def process(settings):
    """Run the requested processing on the loaded dataset; returns a JSON-ready dict.

    settings = {
      "echo": "integrate",
      "multiexp": {"enabled", "n", "offset", "ci_level", "compare"},
      "ilt": {"enabled", "t2_min", "t2_max", "n_steps" (max grid points), "alpha", "sweep",
              "peak_threshold_pct", "sweep_lo", "sweep_hi"},
    }
    """
    if "td" not in _STATE:
        raise RuntimeError("No CPMG dataset is loaded.")
    t = _STATE["echo_s"]
    if settings.get("echo", "integrate") != "integrate":
        raise ValueError(f"Unsupported echo processing: {settings.get('echo')!r}")
    y, phase = echoes.integrate(_STATE["td"])

    out = {"settings": settings, "version": __version__,
           "decay": {"t": t, "y": y, "phase_deg": phase}, "multiexp": None, "ilt": None}

    me = settings.get("multiexp") or {}
    if me.get("enabled"):
        compare = 4 if me.get("compare") else int(me["n"])
        out["multiexp"] = multiexp.fit_with_comparison(
            t, y, int(me["n"]), bool(me.get("offset", True)), float(me.get("ci_level", 0.95)), compare)

    il = settings.get("ilt") or {}
    if il.get("enabled"):
        out["ilt"] = ilt_tools.run(t, y, float(il["t2_min"]), float(il["t2_max"]),
                                   int(il["n_steps"]), float(il["alpha"]), bool(il.get("sweep")),
                                   peak_min_fraction=float(il.get("peak_threshold_pct", 5)) / 100,
                                   sweep_range=(float(il.get("sweep_lo", 4)), float(il.get("sweep_hi", 12))))
    return _jsonable(out)
