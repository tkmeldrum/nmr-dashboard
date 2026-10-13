"""Thin entry points called by the browser worker (and usable from notebooks)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from . import constants, echoes, ilt_tools, multiexp, spatial
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
        nucleus, why = constants.guess_nucleus(params.get("b1Freq"))
        out["defaults"] = {"nucleus": nucleus, "nucleus_reason": why,
                           "G": constants.DEFAULT_G_T_PER_M, "zero_fill": constants.DEFAULT_ZERO_FILL}
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


def _run_ft(cfg):
    td, params = _STATE["td"], _STATE["params"]
    nuc = cfg.get("nucleus", "1H")
    gamma = constants.GAMMA_MHZ_PER_T[nuc]
    g = float(cfg.get("G", constants.DEFAULT_G_T_PER_M))
    zf = int(cfg.get("zero_fill", constants.DEFAULT_ZERO_FILL))
    if g <= 0:
        raise ValueError("The gradient G must be positive.")
    r = echoes.ft(td, float(params.get("dwellTime", 1.0)), gamma, g, zf)
    z, S = r["z_um"], r["S"]
    prof = echoes.profile(S)
    z0, z1 = cfg.get("z_min"), cfg.get("z_max")
    auto = z0 is None or z1 is None
    if auto:
        z0, z1 = echoes.auto_window(z, prof)
    y = echoes.window_decay(S, z, float(z0), float(z1))
    dwell = float(params.get("dwellTime", 1.0))
    res = echoes.native_resolution_um(td.shape[0], dwell, gamma, g)
    spacing = cfg.get("spacing_um")
    spacing_auto = spacing is None
    spacing = float(float(f"{res:.2g}") if spacing_auto else spacing)
    idx = echoes.position_indices(z, float(z0), float(z1), spacing)
    Ypos = echoes.position_decays(S, idx)
    # thinned magnitude map for the heatmap (keeps the message small)
    zs, es = max(1, int(np.ceil(len(z) / 256))), max(1, int(np.ceil(S.shape[1] / 256)))
    ft_out = {
        "z_um": z, "profile": prof, "window": [float(z0), float(z1)], "window_auto": auto,
        "nucleus": nuc, "gamma_mhz_per_t": gamma, "G": g, "zero_fill": zf,
        "center_point": r["center"] + 1, "nfft": r["nfft"],
        "resolution_um": res, "spacing_um": spacing, "spacing_auto": spacing_auto, "n_positions": len(idx),
        "heat": {"z_um": z[::zs], "echo_idx": np.arange(S.shape[1])[::es], "mag": np.abs(S[::zs, ::es])},
    }
    return y, r["phase_deg"], ft_out, (z[idx], Ypos)


def process(settings, progress=None):
    """Run the requested processing on the loaded dataset; returns a JSON-ready dict.

    settings = {
      "echo": "integrate" | "ft",
      "ft": {"nucleus": "1H"|"19F", "G", "zero_fill", "z_min", "z_max" (None -> auto window),
             "spacing_um" (None -> native resolution)},  # FT mode also fits every z position
      "multiexp": {"enabled", "n", "offset", "ci_level", "compare"},
      "ilt": {"enabled", "t2_min", "t2_max", "n_steps" (max grid points), "alpha", "sweep",
              "peak_threshold_pct", "sweep_lo", "sweep_hi"},
    }
    """
    if "td" not in _STATE:
        raise RuntimeError("No CPMG dataset is loaded.")
    t = _STATE["echo_s"]
    mode = settings.get("echo", "integrate")
    ft_out, positions = None, None
    if mode == "integrate":
        y, phase = echoes.integrate(_STATE["td"])
    elif mode == "ft":
        y, phase, ft_out, positions = _run_ft(settings.get("ft") or {})
    else:
        raise ValueError(f"Unsupported echo processing: {mode!r}")

    out = {"settings": settings, "version": __version__, "mode": mode, "ft": ft_out, "spatial": None,
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
    if positions is not None and (me.get("enabled") or il.get("enabled")):
        out["spatial"] = spatial.fit_positions(t, positions[1], positions[0], me, il, progress)
    return _jsonable(out)
