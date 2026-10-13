"""Thin entry points called by the browser worker (and usable from notebooks)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from . import constants, echoes, ilt_tools, multiexp, spatial, uncertainty
from .display_params import build_display
from .kea_io import read_kea_binary, read_kea_params

ILT_MAX_POSITIONS = 1500   # the ILT is cheap, so its map may cover every zero-filled point
DEFAULT_MIN_SIGNAL_PCT = 5   # fit only positions whose profile is >= 5% of its maximum
HEAT_MAX_CELLS = 600_000   # cap on z x echo cells sent for the echo map

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
                           "G": constants.DEFAULT_G_T_PER_M, "zero_fill": constants.DEFAULT_ZERO_FILL,
                           "z_window_um": list(constants.DEFAULT_Z_WINDOW_UM)}
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
    idx = echoes.position_indices(z, float(z0), float(z1), spacing)                     # exponential fits
    ilt_sp = cfg.get("ilt_spacing_um")
    idx_ilt = (echoes.grid_indices(z, float(z0), float(z1)) if ilt_sp is None               # ILT map: every zero-filled point
               else echoes.position_indices(z, float(z0), float(z1), float(ilt_sp), max_positions=ILT_MAX_POSITIONS))
    if len(idx_ilt) > ILT_MAX_POSITIONS:
        raise ValueError(f"{len(idx_ilt)} ILT positions (maximum {ILT_MAX_POSITIONS}); set an ILT spacing or narrow the window.")
    # echo map on the full zero-filled grid (same zf as the processing); only echoes are thinned, to bound the message size
    es = max(1, int(np.ceil(S.shape[1] * len(z) / HEAT_MAX_CELLS)))
    ft_out = {
        "z_um": z, "profile": prof, "window": [float(z0), float(z1)], "window_auto": auto,
        "nucleus": nuc, "gamma_mhz_per_t": gamma, "G": g, "zero_fill": zf,
        "center_point": r["center"] + 1, "nfft": r["nfft"], "zf_step_um": float(z[1] - z[0]),
        "resolution_um": res, "spacing_um": spacing, "spacing_auto": spacing_auto, "n_positions": len(idx),
        "n_ilt_positions": len(idx_ilt),
        "heat": {"z_um": z, "echo_idx": np.arange(S.shape[1])[::es], "mag": np.abs(S[:, ::es])},
    }
    # Skip positions with no signal: fitting noise gives meaningless T2s and wastes time.
    min_sig = float(cfg.get("min_signal_pct", DEFAULT_MIN_SIGNAL_PCT)) / 100
    sig = prof / prof.max() if prof.max() > 0 else np.zeros_like(prof)
    keep, keep_ilt = sig[idx] >= min_sig, sig[idx_ilt] >= min_sig
    if not keep.any():
        raise ValueError(f"No position has signal above {min_sig:.0%} of the profile maximum; lower the threshold or widen the window.")
    ft_out.update({"min_signal_pct": min_sig * 100, "n_positions_kept": int(keep.sum()), "n_ilt_positions_kept": int(keep_ilt.sum())})
    positions = {"me": (z[idx], echoes.position_decays(S, idx), keep, echoes.position_noise(S, idx), echoes.position_noise_series(S, idx)),
                 "ilt": (z[idx_ilt], echoes.position_decays(S, idx_ilt), keep_ilt, echoes.position_noise(S, idx_ilt))}
    return y, r["phase_deg"], ft_out, positions, echoes.window_noise_series(S, z, float(z0), float(z1))


def process(settings, progress=None):
    """Run the requested processing on the loaded dataset; returns a JSON-ready dict.

    settings = {
      "echo": "integrate" | "ft",
      "ft": {"nucleus": "1H"|"19F", "G", "zero_fill", "z_min", "z_max" (None -> auto window),
             "spacing_um" (exp fits; None -> native resolution), "ilt_spacing_um" (None -> every zero-filled point),
             "min_signal_pct" (skip positions below this % of the profile max)},  # FT mode also fits every z position
      "multiexp": {"enabled", "n", "offset", "ci_level", "compare"},
      "ilt": {"enabled", "t2_min", "t2_max", "n_steps" (max grid points), "alpha", "sweep",
              "peak_threshold_pct", "sweep_lo", "sweep_hi"},
    }
    """
    if "td" not in _STATE:
        raise RuntimeError("No CPMG dataset is loaded.")
    t = _STATE["echo_s"]
    mode = settings.get("echo", "integrate")
    ft_out, positions, series = None, None, None
    if mode == "integrate":
        y, phase = echoes.integrate(_STATE["td"])
        series = echoes.integrate_noise_series(_STATE["td"])
    elif mode == "ft":
        y, phase, ft_out, positions, series = _run_ft(settings.get("ft") or {})
    else:
        raise ValueError(f"Unsupported echo processing: {mode!r}")

    sigma = echoes.noise_sigma(series) if series is not None else None
    out = {"settings": settings, "version": __version__, "mode": mode, "ft": ft_out, "spatial": None,
           "decay": {"t": t, "y": y, "phase_deg": phase, "noise_sigma": sigma}, "multiexp": None, "ilt": None}

    me = settings.get("multiexp") or {}
    if me.get("enabled"):
        compare = 4 if me.get("compare") else int(me["n"])
        out["multiexp"] = multiexp.fit_with_comparison(
            t, y, int(me["n"]), bool(me.get("offset", True)), float(me.get("ci_level", 0.95)), compare, sigma=sigma)

    il = settings.get("ilt") or {}
    if il.get("enabled"):
        out["ilt"] = ilt_tools.run(t, y, float(il["t2_min"]), float(il["t2_max"]),
                                   int(il["n_steps"]), float(il["alpha"]), bool(il.get("sweep")),
                                   peak_min_fraction=float(il.get("peak_threshold_pct", 5)) / 100,
                                   sweep_range=(float(il.get("sweep_lo", 4)), float(il.get("sweep_hi", 12))), sigma=sigma)
    _STATE["processed"] = {"settings": settings, "t": t, "y": y, "sigma": sigma, "noise": series, "positions": positions,
                           "me_fit": out["multiexp"]["selected"] if out["multiexp"] else None}
    if positions is not None and (me.get("enabled") or il.get("enabled")):
        out["spatial"] = spatial.fit_positions(
            t, positions["me"], positions["ilt"], me, il, progress,
            peak_min_fraction=float(il.get("peak_threshold_pct", 5)) / 100)
    return _jsonable(out)


# ── MCMC and bootstrap: chunked so the page can show progress, stop early and keep partial results ──

def _target(cfg):
    """The decay to analyse: the bulk decay, or (FT mode) the fitted position nearest cfg['z_um']."""
    pr = _STATE.get("processed")
    if not pr:
        raise RuntimeError("Run the processing first.")
    t, settings = pr["t"], pr["settings"]
    if cfg.get("target", "bulk") == "bulk":
        return {"t": t, "y": pr["y"], "sigma": pr["sigma"], "noise": pr["noise"], "me_fit": pr["me_fit"],
                "label": "window-summed decay" if settings.get("echo") == "ft" else "integrated decay", "z_um": None}
    if pr["positions"] is None:
        raise ValueError("A position can only be chosen after an FT run.")
    z, Y, keep, sig, noise = pr["positions"]["me"]
    ok = np.flatnonzero(keep)
    if len(ok) == 0:
        raise ValueError("No position has signal to analyse.")
    zq = cfg.get("z_um")
    i = int(ok[np.argmax(Y[ok].sum(axis=1))]) if zq is None else int(ok[np.argmin(np.abs(z[ok] - float(zq)))])
    s = float(sig[i]) if np.isfinite(sig[i]) else None
    return {"t": t, "y": Y[i], "sigma": s, "noise": None if noise is None else noise[i], "me_fit": None,
            "label": f"position z = {z[i]:.1f} µm", "z_um": float(z[i])}


def _me_for(tg, cfg_settings):
    me = cfg_settings.get("multiexp") or {}
    if not me.get("enabled"):
        raise ValueError("Enable the multi-exponential fit on the Process tab first (MCMC and the exponential bootstrap use that model).")
    fit = tg["me_fit"] or multiexp.fit(tg["t"], tg["y"], int(me["n"]), bool(me.get("offset", True)),
                                       float(me.get("ci_level", 0.95)), sigma=tg["sigma"])
    return me, fit


def mcmc_start(cfg):
    """Set up (and time) an MCMC run on the chosen decay. cfg: target, z_um, nwalkers, steps, burn, ci_level, sample_sigma, seed."""
    tg = _target(cfg)
    me, fit = _me_for(tg, _STATE["processed"]["settings"])
    sigma = None if (cfg.get("sample_sigma") or tg["sigma"] is None) else tg["sigma"]
    run = uncertainty.Mcmc(tg["t"], tg["y"], int(me["n"]), bool(me.get("offset", True)), fit, sigma=sigma,
                           ci_level=float(cfg.get("ci_level", 0.95)), nwalkers=int(cfg.get("nwalkers", 32)),
                           steps=int(cfg.get("steps", 6000)), burn=int(cfg.get("burn", 2000)), seed=int(cfg.get("seed", 0)))
    _STATE["mcmc"] = run
    return _jsonable({"ndim": run.ndim, "nwalkers": run.nwalkers, "steps": run.steps, "est_seconds": run.est_seconds,
                      "target": tg["label"], "sigma_sampled": run.sample_sigma,
                      "note": None if sigma is not None else "No imaginary channel (or sampling requested): sigma is sampled as a parameter."})


def mcmc_step(k):
    return _STATE["mcmc"].step(int(k))


def mcmc_finish(theme=None):
    return _jsonable(_STATE["mcmc"].result(theme))


def boot_start(cfg):
    """Set up (and time) a bootstrap. cfg: target, z_um, replicates, method, do_multiexp, do_ilt, ci_level, seed."""
    tg = _target(cfg)
    settings = _STATE["processed"]["settings"]
    me = fit = ilt = None
    if cfg.get("do_multiexp", True):
        me, fit = _me_for(tg, settings)
    il = settings.get("ilt") or {}
    if cfg.get("do_ilt", True):
        if not il.get("enabled"):
            if me is None:
                raise ValueError("Enable the multi-exponential fit or the ILT on the Process tab first.")
        else:
            ilt = il
    if me is None and ilt is None:
        raise ValueError("Nothing to bootstrap: choose the multi-exponential fit and/or the ILT.")
    run = uncertainty.Bootstrap(tg["t"], tg["y"], tg["noise"], tg["sigma"], me=me, ls_fit=fit, ilt=ilt,
                                method=cfg.get("method"), replicates=int(cfg.get("replicates", 200)),
                                ci_level=float(cfg.get("ci_level", 0.95)), seed=int(cfg.get("seed", 0)),
                                peak_min_fraction=float((il or {}).get("peak_threshold_pct", 5)) / 100)
    _STATE["boot"] = run
    return _jsonable({"replicates": run.B, "done": run.done, "est_seconds": run.est_seconds, "method": run.method, "target": tg["label"]})


def boot_step(k):
    return _STATE["boot"].step(int(k))


def boot_finish():
    return _jsonable(_STATE["boot"].result())
