"""Thin entry points called by the browser worker (and usable from notebooks)."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from . import constants, echoes, export, ilt2d, ilt_tools, multiexp, provenance, sequences, spatial, t2d, t2d_model, uncertainty
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
    n_echo = int(params.get("nrEchoes", header["yDim"]))
    if header["yDim"] != n_echo:
        return False, f"data.2d has {header['yDim']} echoes but acqu.par says {n_echo}"
    return True, ""


def _load_t1(d, params, out, timestamp_ms):
    """Saturation-recovery data: Kea writes data.dat with two columns, the delay (s) and the amplitude."""
    path = d / "data.dat"
    if not path.is_file():
        raise ValueError("data.dat (delay, amplitude) was not found next to acqu.par")
    try:
        arr = np.loadtxt(path, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"data.dat could not be read as two numeric columns: {exc}") from None
    if arr.shape[1] < 2 or arr.shape[0] < 4:
        raise ValueError("data.dat needs at least 4 rows of (delay, amplitude)")
    t, y = arr[:, 0].astype(float), arr[:, 1].astype(float)
    if np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("the delays in data.dat must be positive and increasing")
    _STATE.update(t1={"t": t, "y": y}, params=params, source={"dir": str(d), "timestamp_ms": timestamp_ms})
    out["raw"] = {"n_points": len(t), "t": _clean(t), "y": _clean(y)}
    out["defaults"] = {"t1_components": 1, "t1_offset": constants.T1_DEFAULT_OFFSET}


def _load_t2d(d, params, out, timestamp_ms):
    """T2-D (SSE-T2): echo trains for every diffusion delay; see t2d.py for the files."""
    L = t2d.load(d, params)
    skip_noise = L["noise"]
    _STATE.update(t2d=L, params=params, source={"dir": str(d), "timestamp_ms": timestamp_ms})
    n_echo, n_tau = L["data"].shape
    out["raw"] = {"n_echoes": n_echo, "n_tau": n_tau, "echo_time_s": _clean(L["echo_s"]), "b_vec": _clean(L["b_vec"]),
                  "tau_s": None if L["tau_s"] is None else _clean(L["tau_s"]),
                  "data": _clean(L["data"].T),                       # rows = diffusion delay, columns = echo
                  "noise_sigma": t2d.noise_sigma(skip_noise)}
    out["defaults"] = dict(constants.T2D_DEFAULTS)


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
    elif fallback_to_mtime and (d / "data2D_Cp.2d").is_file():
        ts = datetime.fromtimestamp((d / "data2D_Cp.2d").stat().st_mtime)
    elif fallback_to_mtime and (d / "data.dat").is_file():
        ts = datetime.fromtimestamp((d / "data.dat").stat().st_mtime)
    else:
        ts = None

    seq = sequences.describe(params)
    if seq["kind"] == "cpmg":
        is_cpmg, reason = check_cpmg(params, header)
        seq["supported"] = is_cpmg
        seq["reason"] = reason
    else:
        is_cpmg, reason = False, seq["reason"]
    _STATE.clear()
    out = {
        "version": __version__,
        "display": build_display(params, ts),
        "params_all": {k: (v if isinstance(v, (str, int, float, list)) else str(v)) for k, v in params.items()},
        "is_cpmg": is_cpmg,
        "kind": seq["kind"], "supported": seq["supported"], "analysis": seq["analysis"],
        "reason": reason,
        "raw": None,
    }
    if seq["kind"] in ("t1", "t2d") and seq["supported"]:
        try:
            (_load_t1 if seq["kind"] == "t1" else _load_t2d)(d, params, out, timestamp_ms)
        except (ValueError, OSError, KeyError) as exc:   # a malformed or incomplete folder: say so instead of crashing
            out.update(supported=False, reason=f"{exc}" if not isinstance(exc, KeyError) else f"acqu.par has no {exc} entry")
    if is_cpmg:
        n_pts, n_echo = header["xDim"], header["yDim"]
        td = spec[: n_pts * n_echo].reshape((n_pts, n_echo), order="F")
        echo_s = 1e-6 * float(params["echoTime"]) * np.arange(1, n_echo + 1)
        summed, _phase = echoes.integrate(td)
        re, im = (td.real, td.imag) if np.iscomplexobj(td) else (td, np.zeros_like(td))
        _STATE.clear()
        _STATE.update(td=td, echo_s=echo_s, params=params, source={"dir": str(d), "timestamp_ms": timestamp_ms})
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
    if out.get("raw") is not None and out.get("supported"):
        _STATE["raw_out"] = out["raw"]                    # what the Raw data tab shows; the saved raw-data figure uses it
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


MIN_POINTS_AFTER_EXCLUDING = 8      # fewer points than this cannot support a meaningful fit


def _skip_count(settings, n_points):
    """How many leading points (echoes, or delays for T1) to leave out of every fit: settings["exclude_first"], default 0."""
    try:
        skip = int(settings.get("exclude_first", 0) or 0)
    except (TypeError, ValueError):
        raise ValueError("exclude_first must be a whole number") from None
    if skip < 0:
        raise ValueError("The number of excluded points cannot be negative.")
    if skip and n_points - skip < MIN_POINTS_AFTER_EXCLUDING:
        raise ValueError(f"Excluding the first {skip} of {n_points} points leaves {n_points - skip}; at least {MIN_POINTS_AFTER_EXCLUDING} are needed.")
    return skip


def _run_ft(cfg, skip=0):
    td, params = _STATE["td"], _STATE["params"]
    nuc = cfg.get("nucleus", "1H")
    gamma = constants.GAMMA_MHZ_PER_T[nuc]
    g = float(cfg.get("G", constants.DEFAULT_G_T_PER_M))
    zf = int(cfg.get("zero_fill", constants.DEFAULT_ZERO_FILL))
    if g <= 0:
        raise ValueError("The gradient G must be positive.")
    r = echoes.ft(td, float(params.get("dwellTime", 1.0)), gamma, g, zf)
    z, S = r["z_um"], r["S"]
    S_fit = S[:, skip:]                           # excluded echoes take no part in the profile, window, fits or noise estimate
    prof = echoes.profile(S_fit)
    z0, z1 = cfg.get("z_min"), cfg.get("z_max")
    auto = z0 is None or z1 is None
    if auto:
        z0, z1 = echoes.auto_window(z, prof)
    y = echoes.window_decay(S, z, float(z0), float(z1))             # every echo, so the page can show the excluded ones
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
    positions = {"me": (z[idx], echoes.position_decays(S_fit, idx), keep, echoes.position_noise(S_fit, idx), echoes.position_noise_series(S_fit, idx)),
                 "ilt": (z[idx_ilt], echoes.position_decays(S_fit, idx_ilt), keep_ilt, echoes.position_noise(S_fit, idx_ilt))}
    return y, r["phase_deg"], ft_out, positions, echoes.window_noise_series(S_fit, z, float(z0), float(z1))


def _t2d_cfg(settings):
    cfg = {**constants.T2D_DEFAULTS, **(settings.get("t2d") or {})}
    if cfg["d_min"] == "auto" or cfg["d_max"] == "auto":                     # the range these b-values can resolve
        cfg["d_min"], cfg["d_max"] = (float(f"{v:.2g}") for v in ilt2d.sensitive_d_window(_STATE["t2d"]["b_vec"]))
    for k in ("t2_min", "t2_max", "d_min", "d_max", "alpha"):
        if not cfg[k] > 0:
            raise ValueError(f"{k} must be positive")
    if cfg["t2_min"] >= cfg["t2_max"] or cfg["d_min"] >= cfg["d_max"]:
        raise ValueError("The minimum of each range must be below its maximum.")
    return cfg


def t2d_estimate(settings):
    """{n_T2, n_D, seconds}: the grid the solver will use and a time estimate from a short timed trial on this machine."""
    L = _STATE.get("t2d")
    if L is None:
        raise RuntimeError("No T2-D dataset is loaded.")
    cfg = _t2d_cfg(settings)
    skip = _skip_count(settings, len(L["echo_s"]))
    nT2, nD = ilt2d.grid_sizes(len(L["echo_s"]) - skip, len(L["b_vec"]), cfg["t2_steps"], cfg["d_steps"])
    if "t2d_speed" not in _STATE:
        _STATE["t2d_speed"] = ilt2d.trial_seconds(L["echo_s"], L["b_vec"])
    return {"n_T2": nT2, "n_D": nD, "unknowns": nT2 * nD, "seconds": ilt2d.estimate_seconds(nT2, nD, _STATE["t2d_speed"])}


def _process_t2d(settings):
    """T2-D: one 2D ILT of the (echo x diffusion-delay) matrix. The first `exclude_first` echoes are left out, as in every fit."""
    L = _STATE["t2d"]
    cfg = _t2d_cfg(settings)
    skip = _skip_count(settings, len(L["echo_s"]))
    sigma = t2d.noise_sigma(L["noise"], skip)
    res = ilt2d.run(L["data"][skip:], L["echo_s"][skip:], L["b_vec"], cfg["t2_min"], cfg["t2_max"], cfg["t2_steps"], cfg["d_min"], cfg["d_max"], cfg["d_steps"],
                    cfg["alpha"], sigma=sigma, peak_min_fraction=float(cfg["peak_threshold_pct"]) / 100)
    tm = settings.get("t2d_model") or {}
    keep = None
    if tm.get("enabled"):                                    # the few-population model: fitted for every K, the BIC-preferred one (or the user's k) chosen
        k = tm.get("k")
        if k is not None and not (isinstance(k, int) and 1 <= k <= t2d_model.MAX_K):
            raise ValueError(f"t2d_model.k must be a whole number from 1 to {t2d_model.MAX_K}, or left out for automatic")
        d_range = res["d_sensitive"] if tm.get("d_window", "resolvable") == "resolvable" else (cfg["d_min"], cfg["d_max"])
        t, b, Y = L["echo_s"][skip:], L["b_vec"], L["data"][skip:]
        res["model"] = t2d_model.fit(t, b, Y, res["peaks"], (cfg["t2_min"], cfg["t2_max"]), d_range, sigma, k=k, max_k=tm.get("max_k"), seed=int(tm.get("seed", 0)))
        keep = {"t": t, "b": b, "Y": Y, "sigma": sigma, "t2_range": (cfg["t2_min"], cfg["t2_max"]), "d_range": d_range, "regions": res["peaks"], "fit": res["model"],
                "noise": None if L.get("noise") is None else np.asarray(L["noise"])[skip:]}
    out = {"settings": settings, "version": __version__, "mode": "t2d", "ft": None, "spatial": None, "multiexp": None, "ilt": None, "t2d": res,
           "decay": {"t": L["echo_s"], "y": None, "skip": skip, "phase_deg": None, "noise_sigma": sigma}}
    _STATE["processed"] = {"settings": settings, "t2d_model": keep}
    return _jsonable(out)


def _process_t1(settings):
    """Saturation recovery: fit 1 (or 2) components; the sparse data have no ILT and no noise channel."""
    t_all, y_all = _STATE["t1"]["t"], _STATE["t1"]["y"]
    skip = _skip_count(settings, len(t_all))
    t, y = t_all[skip:], y_all[skip:]
    me = settings.get("multiexp") or {}
    n = int(me.get("n", 1))
    if n not in (1, 2):
        raise ValueError("T1 recovery supports one or two components")
    res = multiexp.fit_with_comparison(t, y, n, bool(me.get("offset", constants.T1_DEFAULT_OFFSET)), float(me.get("ci_level", 0.95)),
                                       compare_up_to=2 if me.get("compare", True) else n, sigma=None, kind="saturation")
    out = {"settings": settings, "version": __version__, "mode": "t1", "ft": None, "spatial": None,
           "decay": {"t": t_all, "y": y_all, "skip": skip, "phase_deg": None, "noise_sigma": None}, "multiexp": res, "ilt": None}
    _STATE["processed"] = {"settings": settings, "t": t, "y": y, "sigma": None, "noise": None, "positions": None, "me_fit": res["selected"]}
    return _jsonable(out)


def process(settings, progress=None):
    """Run the analysis; the result is also kept so the page can save everything that is on screen (see save_bundle)."""
    out = _process(settings, progress)
    _STATE["last_result"] = out
    _STATE.pop("mcmc", None), _STATE.pop("boot", None)             # uncertainty runs belong to the processing they followed
    return out


def _process(settings, progress=None):
    """Run the requested processing on the loaded dataset; returns a JSON-ready dict.

    settings = {
      "exclude_first": 0,            # leading echoes (T1: delays) left out of every fit
      "echo": "integrate" | "ft",
      "ft": {"nucleus": "1H"|"19F", "G", "zero_fill", "z_min", "z_max" (None -> auto window),
             "spacing_um" (exp fits; None -> native resolution), "ilt_spacing_um" (None -> every zero-filled point),
             "min_signal_pct" (skip positions below this % of the profile max)},  # FT mode also fits every z position
      "multiexp": {"enabled", "n", "offset", "ci_level", "compare"},
      "ilt": {"enabled", "t2_min", "t2_max", "n_steps" (max grid points), "alpha", "sweep",
              "peak_threshold_pct", "sweep_lo", "sweep_hi"},
    }
    """
    if "t1" in _STATE:
        return _process_t1(settings)
    if "t2d" in _STATE:
        return _process_t2d(settings)
    if "td" not in _STATE:
        raise RuntimeError("No supported dataset is loaded.")
    t_all = _STATE["echo_s"]
    skip = _skip_count(settings, len(t_all))
    mode = settings.get("echo", "integrate")
    ft_out, positions, series = None, None, None
    if mode == "integrate":
        y_all, phase = echoes.integrate(_STATE["td"])
        series = echoes.integrate_noise_series(_STATE["td"])
        series = None if series is None else series[skip:]
    elif mode == "ft":
        y_all, phase, ft_out, positions, series = _run_ft(settings.get("ft") or {}, skip)
    else:
        raise ValueError(f"Unsupported echo processing: {mode!r}")
    t, y = t_all[skip:], y_all[skip:]                      # everything below (fits, ILT, MCMC, bootstrap) sees only these

    sigma = echoes.noise_sigma(series) if series is not None else None
    out = {"settings": settings, "version": __version__, "mode": mode, "ft": ft_out, "spatial": None,
           "decay": {"t": t_all, "y": y_all, "skip": skip, "phase_deg": phase, "noise_sigma": sigma}, "multiexp": None, "ilt": None}

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
    if out["multiexp"] and out["ilt"]:
        out["ilt"]["exp_markers"] = ilt_tools.exp_markers(out["multiexp"]["selected"], out["ilt"])
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
    if "t2d" in _STATE:
        raise ValueError("MCMC and the bootstrap are not available for T2-D data yet.")
    t, settings = pr["t"], pr["settings"]
    if cfg.get("target", "bulk") == "bulk":
        return {"t": t, "y": pr["y"], "sigma": pr["sigma"], "noise": pr["noise"], "me_fit": pr["me_fit"],
                "label": "recovery curve" if settings.get("kind") == "t1" else "window-summed decay" if settings.get("echo") == "ft" else "integrated decay", "z_um": None}
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


def _t2d_model_data():
    """What MCMC and the bootstrap of a T2-D dataset sample: the few-population model fitted by the last run (the regularised map has no intervals)."""
    pr = _STATE.get("processed") or {}
    keep = pr.get("t2d_model")
    if not keep:
        raise ValueError("For T2-D data, MCMC and the bootstrap sample the few-population model: switch it on and run the processing first "
                         "(the regularised map itself has no intervals).")
    return keep


def mcmc_start(cfg):
    """Set up (and time) an MCMC run on the chosen decay. cfg: target, z_um, nwalkers, steps, burn, ci_level, sample_sigma, seed."""
    if "t2d" in _STATE:
        from . import t2d_uncertainty
        k = _t2d_model_data()
        run = t2d_uncertainty.Mcmc2D(k["t"], k["b"], k["Y"], k["fit"], k["t2_range"], k["d_range"], sigma=None if cfg.get("sample_sigma") else k["sigma"],
                                     ci_level=float(cfg.get("ci_level", 0.95)), nwalkers=int(cfg.get("nwalkers", 32)), steps=int(cfg.get("steps", 6000)),
                                     burn=int(cfg.get("burn", 2000)), seed=int(cfg.get("seed", 0)))
        _STATE["mcmc"], _STATE["mcmc_target"] = run, "bulk"
        return _jsonable({"ndim": run.ndim, "nwalkers": run.nwalkers, "steps": run.steps, "est_seconds": run.est_seconds,
                          "target": f"{k['fit']['K']}-population T2-D model", "sigma_sampled": run.sample_sigma, "note": None})
    tg = _target(cfg)
    me, fit = _me_for(tg, _STATE["processed"]["settings"])
    sigma = None if (cfg.get("sample_sigma") or tg["sigma"] is None) else tg["sigma"]
    run = uncertainty.Mcmc(tg["t"], tg["y"], int(me["n"]), bool(me.get("offset", True)), fit, sigma=sigma,
                           ci_level=float(cfg.get("ci_level", 0.95)), nwalkers=int(cfg.get("nwalkers", 32)),
                           steps=int(cfg.get("steps", 6000)), burn=int(cfg.get("burn", 2000)), seed=int(cfg.get("seed", 0)))
    _STATE["mcmc"] = run
    _STATE["mcmc_target"] = cfg.get("target", "bulk")
    return _jsonable({"ndim": run.ndim, "nwalkers": run.nwalkers, "steps": run.steps, "est_seconds": run.est_seconds,
                      "target": tg["label"], "sigma_sampled": run.sample_sigma,
                      "note": None if sigma is not None else "No imaginary channel (or sampling requested): sigma is sampled as a parameter."})


def mcmc_step(k):
    return _STATE["mcmc"].step(int(k))


def mcmc_finish(theme=None, max_samples=0):
    return _jsonable(_STATE["mcmc"].result(theme, max_samples))


def boot_start(cfg):
    """Set up (and time) a bootstrap. cfg: target, z_um, replicates, method, do_multiexp, do_ilt, ci_level, seed."""
    if "t2d" in _STATE:
        from . import t2d_uncertainty
        k = _t2d_model_data()
        run = t2d_uncertainty.Bootstrap2D(k["t"], k["b"], k["Y"], k["fit"], k["t2_range"], k["d_range"], noise=k["noise"], sigma=k["sigma"], method=cfg.get("method"),
                                          replicates=int(cfg.get("replicates", 200)), ci_level=float(cfg.get("ci_level", 0.95)), seed=int(cfg.get("seed", 0)))
        _STATE["boot"] = run
        return _jsonable({"replicates": run.B, "done": run.done, "est_seconds": run.est_seconds, "method": run.method, "target": f"{k['fit']['K']}-population T2-D model"})
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


# ── provenance and export, for the web page ──────────────────────────────────

def _provenance_record(page=None, options=None, outputs=None):
    pr, src = _STATE.get("processed"), _STATE.get("source")
    if not pr or not src:
        raise RuntimeError("Run the processing first.")
    page = page or {}
    from . import params as P
    text = P.dump_toml(P.from_settings(pr["settings"], options))
    notes = ["Computed in the browser with Pyodide; the data never left this computer."] + list(page.get("notes") or [])
    extra = dict(page.get("browser") or {})
    if page.get("uncertainty"):
        extra["uncertainty_runs"] = page["uncertainty"]
    return provenance.build(src["dir"], pr["settings"], options or {}, params_text=text, command=None, dataset=page.get("dataset"),
                            acquisition=_STATE["params"], notes=notes, extra=extra or None, outputs=outputs, modified_ms=src["timestamp_ms"]), text


def get_provenance(page=None, options=None):
    """Provenance of the last processing run on the loaded dataset: {record, lines}.

    page: {dataset, browser: {...}, uncertainty: {...}} from the page (label, user agent, which MCMC / bootstrap ran).
    options: the uncertainty options in the same shape the script's params use (see params.to_settings).
    """
    rec, _ = _provenance_record(page, options)
    return _jsonable({"record": rec, "lines": provenance.describe(rec)})


def export_params(settings, options=None, dataset=None, style=None):
    return export.params_text(settings, options, dataset, style)


def export_script():
    return export.SCRIPT


def save_bundle(opts):
    """Everything the page shows, as files: tables, figures, params_used.toml, provenance.json and summary.md.

    opts: {figure_format, dpi, style, csv, figures, samples, mcmc, boot, zip, options, page}.
    Returns {"folder": "processed_<stamp>", "files": {name: base64}} or, with zip, {"zip_name", "zip_b64", "files": [names]}.
    """
    import base64, io, tempfile, zipfile
    from pathlib import Path
    from . import outputs
    from . import params as P
    res = _STATE.get("last_result")
    pr = _STATE.get("processed")
    if not res or not pr:
        raise RuntimeError("Run the processing first.")
    options, page = opts.get("options") or {}, opts.get("page") or {}
    settings = pr["settings"]
    samples = bool(opts.get("samples", True))
    mcmc = boot = None
    notes = []
    if opts.get("mcmc") and _STATE.get("mcmc") is not None:
        mcmc = _jsonable(_STATE["mcmc"].result({"fg": "#1a1a1a", "grid": "#bbbbbb"}, 5000 if samples else 0))
    if opts.get("boot") and _STATE.get("boot") is not None:
        boot = _jsonable(_STATE["boot"].result())
    band = mcmc if (mcmc and _STATE.get("mcmc_target", "bulk") == "bulk") else None
    notes += list((res.get("multiexp") or {}).get("selected", {}).get("warnings", []) or [])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f"processed_{stamp}"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        files = []
        if opts.get("csv", True):
            files += outputs.write_tables(d, res, mcmc, boot, samples=samples)
        if opts.get("figures", True):
            files += outputs.write_figures(d, res, mcmc, boot, opts.get("figure_format", "png"), int(opts.get("dpi", 150)), True, band, style=opts.get("style") or {},
                                           raw=_STATE.get("raw_out"), kind=("t1" if "t1" in _STATE else "t2d" if "t2d" in _STATE else "cpmg"))
        label = page.get("dataset") or "dataset"
        (d / "params_used.toml").write_text(P.dump_toml(P.with_style(P.from_settings(settings, options), opts.get("style")), header=f"Used for: {label}"))
        files.append("params_used.toml")
        names = files + ["provenance.json", "summary.md"]
        rec, _ = _provenance_record({**page, "notes": list(page.get("notes") or []) + notes}, options, outputs=names)
        (d / "provenance.json").write_text(json.dumps(rec, indent=2, default=str))
        (d / "summary.md").write_text(outputs.summary_markdown(res, rec, mcmc, boot, names, notes))
        blobs = {n: (d / n).read_bytes() for n in names}
    if opts.get("zip"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in blobs.items():
                z.writestr(f"{folder}/{n}", b)
        return {"zip_name": f"{folder}.zip", "zip_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "files": names}
    return {"folder": folder, "files": {n: base64.b64encode(b).decode("ascii") for n, b in blobs.items()}}


def params_for_page(text, name="params.toml"):
    """Read a params file (TOML or JSON text) and return what the page needs to fill its controls for the loaded dataset.

    {kind, settings, options, style, notes}. Values left "auto" in the file come back as None (ft.nucleus, z-window, spacing) or
    resolved from the loaded data (the ILT T2 range); the page keeps its own value where the file has none.
    """
    import tomllib
    from . import params as P
    try:
        raw = json.loads(text) if str(name).lower().endswith(".json") else tomllib.loads(text)
    except Exception as exc:
        raise ValueError(f"could not read {name}: {exc}") from None
    p = P.validate(raw)
    if "t1" in _STATE:
        kind, (settings, options) = "t1", P.to_settings_t1(p)
    elif "t2d" in _STATE:
        kind, (settings, options) = "t2d", P.to_settings_t2d(p)
    elif "echo_s" in _STATE:
        kind, (settings, options) = "cpmg", P.to_settings(p, np.asarray(_STATE["echo_s"], float), default_nucleus=None)
    else:
        raise RuntimeError("Load a dataset first.")
    notes = []
    if p["output"] != P.DEFAULTS["output"] or p["batch"] != P.DEFAULTS["batch"]:
        notes.append("The [output] and [batch] settings only matter to the command-line script, so they were not used here.")
    return _jsonable({"kind": kind, "settings": settings, "options": options, "style": P.style_dict(p), "notes": notes})


def batch_start(args):
    from . import webbatch
    return webbatch.start(args)


def batch_run_one(args):
    from . import webbatch
    return webbatch.run_one(args)


def batch_finish(args=None):
    from . import webbatch
    return webbatch.finish(args)
