"""Thin entry points called by the browser worker (and usable from notebooks)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from . import constants, echoes, export, ilt_tools, multiexp, provenance, sequences, spatial, uncertainty
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
    if seq["kind"] == "t1" and seq["supported"]:
        try:
            _load_t1(d, params, out, timestamp_ms)
        except ValueError as exc:                       # a malformed data.dat: say so instead of crashing
            out.update(supported=False, reason=str(exc))
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


def mcmc_finish(theme=None, max_samples=0):
    return _jsonable(_STATE["mcmc"].result(theme, max_samples))


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


# ── provenance and export, for the web page ──────────────────────────────────

def get_provenance(page=None, options=None):
    """Provenance of the last processing run on the loaded dataset: {record, lines}.

    page: {dataset, browser: {...}, uncertainty: {...}} from the page (label, user agent, which MCMC / bootstrap ran).
    options: the uncertainty options in the same shape the script's params use (see params.to_settings).
    """
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
    rec = provenance.build(src["dir"], pr["settings"], options or {}, params_text=text, command=None, dataset=page.get("dataset"),
                           acquisition=_STATE["params"], notes=notes, extra=extra or None, modified_ms=src["timestamp_ms"])
    return _jsonable({"record": rec, "lines": provenance.describe(rec)})


def export_params(settings, options=None, dataset=None):
    return export.params_text(settings, options, dataset)


def export_script():
    return export.SCRIPT
