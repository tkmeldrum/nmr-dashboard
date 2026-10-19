"""The params file: every setting a run needs, with defaults, checks and a commented TOML writer.

One schema serves three things: reading the file a user edits, turning it into the `settings` dict that
`pipeline.process` takes (the same dict the web page builds), and writing a fully commented file back out
(the page's "Export" button and `nmrdash params`).
"""
import math

from . import constants

try:
    import tomllib                      # Python 3.11+
except ImportError:                     # pragma: no cover
    import tomli as tomllib

AUTO = "auto"


# ── checks ──────────────────────────────────────────────────────────────────

def _one_of(*options):
    def check(v):
        if v not in options:
            raise ValueError("must be one of " + ", ".join(repr(o) for o in options))
    return check


def _number(lo=None, hi=None, lo_open=False):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("must be a number")
        if lo is not None and (v < lo or (lo_open and v == lo)):
            raise ValueError(f"must be {'>' if lo_open else '>='} {lo}")
        if hi is not None and v > hi:
            raise ValueError(f"must be <= {hi}")
    return check


def _integer(lo=None, hi=None):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("must be a whole number")
        _number(lo, hi)(v)
    return check


def _boolean(v):
    if not isinstance(v, bool):
        raise ValueError("must be true or false")


def _auto_or(inner):
    def check(v):
        if v != AUTO:
            inner(v)
    return check


def _text(v):
    if not isinstance(v, str):
        raise ValueError("must be text in quotes")


def _t0(v):
    if v not in ("first", "none") and not isinstance(v, str):
        raise ValueError("must be \"first\", \"none\" or an ISO date-time such as \"2026-10-13T09:00:00\"")


# (key, default, comment, check) per section, in file order
SPEC = {
    "processing": [
        ("exclude_first", 0, "Leave the first n echoes (for T1: delays) out of every fit, ILT, MCMC and bootstrap. They are still shown, as open markers.", _integer(0)),
        ("echo", "integrate", "How each echo becomes a decay: \"integrate\" (autophase, sum the points) or \"ft\" (Fourier transform:\n# signal vs position z; the analyses below then run at every z position).", _one_of("integrate", "ft")),
    ],
    "ft": [
        ("nucleus", AUTO, "\"1H\", \"19F\" or \"auto\" (chosen from b1Freq in acqu.par).", _one_of(AUTO, "1H", "19F")),
        ("G", constants.DEFAULT_G_T_PER_M, "Static field gradient, T/m (the same quantity as t2d.gradient).", _number(0, lo_open=True)),
        ("zero_fill", 4, "FFT length is the next power of two above points x 2^zero_fill.", _integer(0, 6)),
        ("z_min", -200.0, "Sample window, micrometres (+ z = + frequency = closer to the magnet). \"auto\" for both = where the\n# profile exceeds 10% of its maximum.", _auto_or(_number())),
        ("z_max", 200.0, "", _auto_or(_number())),
        ("spacing_um", AUTO, "Spacing of the exponential fits across the window; \"auto\" = the natural resolution 1/(points x dwell x gamma G).", _auto_or(_number(0, lo_open=True))),
        ("ilt_spacing_um", AUTO, "Spacing of the ILT map; \"auto\" = every zero-filled FFT point.", _auto_or(_number(0, lo_open=True))),
        ("min_signal_pct", 5.0, "Positions whose profile is below this % of its maximum are not fitted (they only contain noise).", _number(0, 100)),
    ],
    "multiexp": [
        ("enabled", True, "Fit y(t) = sum_i A_i exp(-t/T2_i) [+ c].", _boolean),
        ("n", 2, "Number of components (1-4).", _integer(1, 4)),
        ("offset", True, "Include the constant offset c.", _boolean),
        ("compare", True, "Also fit 1-4 components and compare them (Delta AIC / Delta BIC). Applies to the whole decay.", _boolean),
    ],
    "ilt": [
        ("enabled", True, "Inverse Laplace transform (regularised non-negative least squares).", _boolean),
        ("t2_min", AUTO, "T2 grid limits, seconds; \"auto\" = the decade at or below the first echo / at or above 10x the last echo time.", _auto_or(_number(0, lo_open=True))),
        ("t2_max", AUTO, "", _auto_or(_number(0, lo_open=True))),
        ("max_grid_points", 100, "Most grid points (never more than the number of echoes).", _integer(10, 1000)),
        ("alpha", 1e8, "Regularisation strength: larger = less smoothing. Typical range 1e6 - 1e10.", _number(0, lo_open=True)),
        ("peak_threshold_pct", 5.0, "Peaks below this % of the tallest are ignored in the peak table and the T2 tracks.", _number(0, 100)),
        ("alpha_sweep", False, "Also record fit RMSE and spectrum roughness over a range of alpha (L-curve).", _boolean),
        ("sweep_lo", 4, "The sweep runs alpha = 10^sweep_lo ... 10^sweep_hi.", _integer()),
        ("sweep_hi", 12, "", _integer()),
    ],
    "uncertainty": [
        ("ci_level", 0.95, "Level of the fit confidence intervals and of the MCMC / bootstrap intervals (0.68, 0.95 or 0.99 are usual).", _number(0, 1, lo_open=True)),
        ("target", "bulk", "What MCMC and the bootstrap analyse: \"bulk\" (the whole decay, or the FT window sum) or \"position\" (one z, FT only).", _one_of("bulk", "position")),
        ("z_um", "strongest", "For target = \"position\": the z (um) to use, or \"strongest\" for the position with the most signal.", lambda v: None if v == "strongest" else _number()(v)),
    ],
    "t1": [
        ("components", 1, "Saturation recovery (sequence T1Sat): number of exponential components, 1 or 2 (two is rare and needs good data).", _integer(1, 2)),
        ("offset", True, "Include a constant offset c: M(t) = c + sum_i A_i (1 - exp(-t/T1_i)). Kea recovery curves usually do not start at zero.", _boolean),
        ("compare", True, "Also fit 1 and 2 components and compare them (Delta AIC / Delta BIC).", _boolean),
    ],
    "t2d": [
        ("gradient", constants.DEFAULT_G_T_PER_M, "T2-D (sequence SSE-T2): the static field gradient G of the magnet, T/m. The b-values are calculated from it with the delays in acqu.par:\n# b = (gamma G)^2 tau^2 (Delta + 2 tau / 3). It is not read from the spectrometer's files, which only serve as a cross-check.", _number(0, lo_open=True)),
        ("t2_min", 1e-5, "T2 grid, seconds.", _number(0, lo_open=True)),
        ("t2_max", 1.0, "", _number(0, lo_open=True)),
        ("t2_steps", 50, "T2 grid points (never more than echoes). The solve is a dense problem of T2 x D unknowns: 50 x 25 takes seconds, 100 x 25 about a minute, in the browser several times longer.", _integer(10, 400)),
        ("d_min", 1e-12, "Diffusion coefficient grid, m^2/s (processed internally as D x 1e9, reported in m^2/s). The b-values can only tell apart D from about\n# 1/b_max to 5/b_min; outside that the solution is regularisation, not data. \"auto\" for both = exactly that range for this dataset.", _auto_or(_number(0, lo_open=True))),
        ("d_max", 1e-8, "", _auto_or(_number(0, lo_open=True))),
        ("d_steps", 25, "D grid points (never more than the number of diffusion delays).", _integer(5, 200)),
        ("alpha", 1e9, "Regularisation strength: larger = less smoothing.", _number(0, lo_open=True)),
        ("peak_threshold_pct", 10.0, "Regions of the spectrum above this % of its maximum are reported as peaks.", _number(0, 100)),
    ],
    "t2d_model": [
        ("enabled", True, "T2-D: also fit a few-population model (T2, D and amplitude per population, each with an interval). MCMC and the bootstrap sample this\n# model; the regularised map itself has no intervals.", _boolean),
        ("populations", "auto", "Number of populations: \"auto\" (the number BIC prefers, from 1 to 4) or a whole number from 1 to 4.", _auto_or(_integer(1, 4))),
        ("d_window", "resolvable", "Diffusion range the populations may occupy: \"resolvable\" (what the b-values can tell apart) or \"map\" (the map's d_min to d_max).", _one_of("resolvable", "map")),
        ("seed", 0, "", _integer(0)),
    ],
    "mcmc": [
        ("enabled", False, "MCMC (emcee) on the multi-exponential model (T2-D: on the few-population model): needs the emcee package.", _boolean),
        ("walkers", 32, "", _integer(8)),
        ("steps", 6000, "", _integer(100)),
        ("burn", 2000, "Steps discarded as burn-in (must be below steps).", _integer(0)),
        ("sample_sigma", False, "Sample the noise sigma too (otherwise it is fixed from the imaginary channel).", _boolean),
        ("seed", 0, "", _integer(0)),
    ],
    "bootstrap": [
        ("enabled", False, "Refit surrogate decays (fitted curve + resampled noise).", _boolean),
        ("replicates", 200, "", _integer(20)),
        ("method", AUTO, "Noise source: \"auto\" (the measured imaginary-channel noise, else residuals), \"noise\", \"residual\" or \"parametric\".", _one_of(AUTO, "noise", "residual", "parametric")),
        ("multiexp", True, "Bootstrap the multi-exponential fit.", _boolean),
        ("ilt", True, "Bootstrap the ILT spectrum.", _boolean),
        ("seed", 0, "", _integer(0)),
    ],
    "output": [
        ("directory", "", "Where results go. Empty = a processed_<timestamp> folder inside each data folder (raw data is never touched);\n# otherwise a folder you name, mirrored by dataset.", _text),
        ("csv", True, "Numeric results as CSV files.", _boolean),
        ("figures", True, "Figures.", _boolean),
        ("report", True, "A 2-4 page PDF report per dataset (US Letter): settings, key numbers with intervals, warnings, the main figures and the provenance; a batch also gets a batch report.", _boolean),
        ("figure_format", "png", "", _one_of("png", "pdf", "svg")),
        ("figure_dpi", 150, "", _integer(50, 1200)),
        ("titles", True, "Put titles on figures (switch off for publication figures).", _boolean),
        ("samples", True, "Write MCMC / bootstrap samples (large) as CSV.", _boolean),
    ],
    "style": [
        ("font_size", 10.0, "Plot settings for the saved figures (the web page has the same ones in its Plot settings panel).\n# Font size in points; every text scales with it.", _number(4, 40)),
        ("line_scale", 1.0, "Multiplies the line widths.", _number(0.1, 10)),
        ("marker_scale", 1.0, "Multiplies the marker sizes.", _number(0.1, 10)),
        ("cmap", "viridis", "Colour map of the heat maps (any matplotlib name, e.g. viridis, plasma, magma, cividis, gray).", _text),
        ("width", AUTO, "Figure width in inches, or \"auto\" to keep each figure's own size.", _auto_or(_number(1, 40))),
        ("height", AUTO, "Figure height in inches, or \"auto\".", _auto_or(_number(1, 40))),
        ("legend", True, "Show legends.", _boolean),
        ("title", "auto", "Titles: \"auto\" (each figure's own), \"none\" or \"custom\" (title_text on every figure).", _one_of("auto", "none", "custom")),
        ("title_text", "", "", _text),
        ("time_unit", "s", "Time unit of tick and axis labels: \"s\" or \"ms\" (tables stay in seconds).", _one_of("s", "ms")),
        ("decay_logy", False, "Log y-axis on the decay figure.", _boolean),
    ],
    "batch": [
        ("t0", "first", "Zero of the elapsed-time column: \"first\" (the earliest dataset), \"none\" (no elapsed time) or an ISO date-time.", _t0),
        ("jobs", 1, "Datasets processed in parallel.", _integer(1, 256)),
        ("timestamps_file", "", "Optional CSV with columns folder,timestamp (ISO) that overrides the data files' modification times, e.g. after files were copied.", _text),
    ],
}

# one-figure overrides: [figures.decay_fit] and so on. Limits and scales apply to the single-plot figures; the title to any figure.
FIGURE_NAMES = ("raw_data", "ft_profile_map", "decay_fit", "model_comparison", "ilt_spectrum", "alpha_sweep", "spatial", "spatial_extra", "mcmc_trace",
                "bootstrap_ilt", "bootstrap_distributions", "t2d_spectrum", "t2d_populations", "t2d_fit")
FIGURE_KEYS = {
    "title": _text,
    "xmin": _auto_or(_number()), "xmax": _auto_or(_number()), "ymin": _auto_or(_number()), "ymax": _auto_or(_number()),
    "xscale": _one_of("auto", "linear", "log"), "yscale": _one_of("auto", "linear", "log"),
}
FIGURE_DEFAULTS = {"title": "", "xmin": AUTO, "xmax": AUTO, "ymin": AUTO, "ymax": AUTO, "xscale": AUTO, "yscale": AUTO}

DEFAULTS = {sec: {k: d for k, d, _doc, _chk in rows} for sec, rows in SPEC.items()}


def defaults():
    return {**{sec: dict(vals) for sec, vals in DEFAULTS.items()}, "figures": {}}


# ── reading ─────────────────────────────────────────────────────────────────

def _validate_figures(vals):
    if not isinstance(vals, dict):
        raise ValueError("[figures] must hold one table per figure, e.g. [figures.decay_fit]")
    out = {}
    for name, o in vals.items():
        if name not in FIGURE_NAMES:
            raise ValueError(f"unknown figure [figures.{name}]; the figures are: " + ", ".join(FIGURE_NAMES))
        if not isinstance(o, dict):
            raise ValueError(f"[figures.{name}] must be a table")
        f = dict(FIGURE_DEFAULTS)
        for k, v in o.items():
            if k not in FIGURE_KEYS:
                raise ValueError(f"unknown setting figures.{name}.{k}; the settings are: " + ", ".join(FIGURE_KEYS))
            try:
                FIGURE_KEYS[k](v)
            except ValueError as exc:
                raise ValueError(f"figures.{name}.{k} = {v!r}: {exc}") from None
            f[k] = v
        out[name] = f
    return out


def validate(params):
    """Fill defaults and check every value; raises ValueError naming the section.key at fault."""
    out = defaults()
    for sec, vals in (params or {}).items():
        if sec == "figures":
            out["figures"] = _validate_figures(vals)
            continue
        if sec not in SPEC:
            raise ValueError(f"unknown section [{sec}]; the sections are: " + ", ".join(SPEC))
        if not isinstance(vals, dict):
            raise ValueError(f"[{sec}] must be a table")
        known = {k for k, *_ in SPEC[sec]}
        for k, v in vals.items():
            if k not in known:
                raise ValueError(f"unknown setting {sec}.{k}; the settings in [{sec}] are: " + ", ".join(sorted(known)))
            out[sec][k] = v
    for sec, rows in SPEC.items():
        for k, _d, _doc, check in rows:
            try:
                check(out[sec][k])
            except ValueError as exc:
                raise ValueError(f"{sec}.{k} = {out[sec][k]!r}: {exc}") from None
    if out["mcmc"]["enabled"] and out["mcmc"]["burn"] >= out["mcmc"]["steps"]:
        raise ValueError("mcmc.burn must be smaller than mcmc.steps")
    for a, b in (("z_min", "z_max"),):
        lo, hi = out["ft"][a], out["ft"][b]
        if (lo == AUTO) != (hi == AUTO):
            raise ValueError("ft.z_min and ft.z_max must both be numbers or both \"auto\"")
        if lo != AUTO and lo >= hi:
            raise ValueError("ft.z_min must be below ft.z_max")
    if out["ilt"]["sweep_lo"] >= out["ilt"]["sweep_hi"]:
        raise ValueError("ilt.sweep_lo must be below ilt.sweep_hi")
    td = out["t2d"]
    if (td["d_min"] == AUTO) != (td["d_max"] == AUTO):
        raise ValueError("t2d.d_min and t2d.d_max must both be numbers or both \"auto\"")
    if td["t2_min"] >= td["t2_max"] or (td["d_min"] != AUTO and td["d_min"] >= td["d_max"]):
        raise ValueError("t2d: the minimum of each range must be below its maximum")
    t2lo, t2hi = out["ilt"]["t2_min"], out["ilt"]["t2_max"]
    if t2lo != AUTO and t2hi != AUTO and t2lo >= t2hi:
        raise ValueError("ilt.t2_min must be below ilt.t2_max")
    return out


def load(path):
    """Read and validate a params file (.toml, or .json with the same structure)."""
    path = str(path)
    if path.lower().endswith(".json"):
        import json
        with open(path) as f:
            raw = json.load(f)
    else:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    return validate(raw)


# ── to the settings the pipeline (and the web page) use ─────────────────────

def _decade_range(t):
    """Same rule as the page: the decade at or below the first echo, at or above 10x the last."""
    return 10.0 ** math.floor(math.log10(t[0])), 10.0 ** math.ceil(math.log10(10 * t[-1]))


def to_settings(params, echo_times_s, default_nucleus="1H"):
    """Resolve 'auto' values for one dataset. Returns (settings, uncertainty_options).

    settings has exactly the structure pipeline.process takes; uncertainty_options has what the chunked
    MCMC / bootstrap calls take.
    """
    p = validate(params)
    t2lo, t2hi = _decade_range(echo_times_s)
    ft, il, me, un = p["ft"], p["ilt"], p["multiexp"], p["uncertainty"]
    nz = lambda v: None if v == AUTO else v
    settings = {
        "exclude_first": p["processing"]["exclude_first"],
        "echo": p["processing"]["echo"],
        "ft": {"nucleus": default_nucleus if ft["nucleus"] == AUTO else ft["nucleus"], "G": ft["G"], "zero_fill": ft["zero_fill"],
               "z_min": nz(ft["z_min"]), "z_max": nz(ft["z_max"]), "spacing_um": nz(ft["spacing_um"]),
               "ilt_spacing_um": nz(ft["ilt_spacing_um"]), "min_signal_pct": ft["min_signal_pct"]},
        "multiexp": {"enabled": me["enabled"], "n": me["n"], "offset": me["offset"], "ci_level": un["ci_level"], "compare": me["compare"]},
        "ilt": {"enabled": il["enabled"], "t2_min": t2lo if il["t2_min"] == AUTO else il["t2_min"],
                "t2_max": t2hi if il["t2_max"] == AUTO else il["t2_max"], "n_steps": il["max_grid_points"], "alpha": il["alpha"],
                "peak_threshold_pct": il["peak_threshold_pct"], "sweep": il["alpha_sweep"], "sweep_lo": il["sweep_lo"], "sweep_hi": il["sweep_hi"]},
    }
    if settings["echo"] != "ft":
        settings.pop("ft")
    mc, bs = p["mcmc"], p["bootstrap"]
    target = {"target": un["target"], "z_um": None if un["z_um"] == "strongest" else un["z_um"], "ci_level": un["ci_level"]}
    options = {
        "mcmc": {"enabled": mc["enabled"], **target, "nwalkers": mc["walkers"], "steps": mc["steps"], "burn": mc["burn"],
                 "sample_sigma": mc["sample_sigma"], "seed": mc["seed"]},
        "bootstrap": {"enabled": bs["enabled"], **target, "replicates": bs["replicates"], "method": None if bs["method"] == AUTO else bs["method"],
                      "do_multiexp": bs["multiexp"], "do_ilt": bs["ilt"], "seed": bs["seed"]},
    }
    return settings, options


def to_settings_t1(params):
    """Settings for a saturation-recovery dataset (no echo processing, no ILT, no FT)."""
    p = validate(params)
    t1, un, mc, bs = p["t1"], p["uncertainty"], p["mcmc"], p["bootstrap"]
    settings = {"kind": "t1", "exclude_first": p["processing"]["exclude_first"], "multiexp": {"enabled": True, "n": t1["components"], "offset": t1["offset"], "ci_level": un["ci_level"], "compare": t1["compare"]}}
    target = {"target": "bulk", "z_um": None, "ci_level": un["ci_level"]}
    options = {
        "mcmc": {"enabled": mc["enabled"], **target, "nwalkers": mc["walkers"], "steps": mc["steps"], "burn": mc["burn"], "sample_sigma": mc["sample_sigma"], "seed": mc["seed"]},
        "bootstrap": {"enabled": bs["enabled"], **target, "replicates": bs["replicates"], "method": None if bs["method"] == AUTO else bs["method"],
                      "do_multiexp": True, "do_ilt": False, "seed": bs["seed"]},
    }
    return settings, options


def to_settings_t2d(params):
    """Settings for a T2-D (SSE-T2) dataset, and the uncertainty options (MCMC / bootstrap sample the few-population model)."""
    p = validate(params)
    tm, un, mc, bs = p["t2d_model"], p["uncertainty"], p["mcmc"], p["bootstrap"]
    settings = {"kind": "t2d", "exclude_first": p["processing"]["exclude_first"], "t2d": dict(p["t2d"]),
                "t2d_model": {"enabled": tm["enabled"], "k": None if tm["populations"] == AUTO else tm["populations"], "d_window": tm["d_window"], "seed": tm["seed"]}}
    target = {"target": "bulk", "z_um": None, "ci_level": un["ci_level"]}
    options = {
        "mcmc": {"enabled": mc["enabled"], **target, "nwalkers": mc["walkers"], "steps": mc["steps"], "burn": mc["burn"], "sample_sigma": mc["sample_sigma"], "seed": mc["seed"]},
        "bootstrap": {"enabled": bs["enabled"], **target, "replicates": bs["replicates"], "method": None if bs["method"] == AUTO else bs["method"],
                      "do_multiexp": True, "do_ilt": False, "seed": bs["seed"]},
    }
    return settings, options


def from_settings(settings, options=None, base=None):
    """The inverse: the params that reproduce a page run (the page's settings + uncertainty options)."""
    p = validate(base) if base else defaults()
    p["processing"]["exclude_first"] = int(settings.get("exclude_first", 0) or 0)
    if settings.get("kind") == "t2d":
        p["t2d"].update(settings.get("t2d") or {})
        tm = settings.get("t2d_model") or {}
        p["t2d_model"].update({"enabled": bool(tm.get("enabled", True)), "populations": AUTO if tm.get("k") in (None, "auto") else tm["k"],
                               "d_window": tm.get("d_window", "resolvable"), "seed": tm.get("seed", 0)})
        p["uncertainty"]["ci_level"] = (options or {}).get("mcmc", {}).get("ci_level", 0.95)
        for name in ("mcmc", "bootstrap"):
            o = (options or {}).get(name) or {}
            if o:
                p[name]["enabled"] = bool(o.get("enabled", False))
                if name == "mcmc":
                    p["mcmc"].update({"walkers": o.get("nwalkers", 32), "steps": o.get("steps", 6000), "burn": o.get("burn", 2000),
                                      "sample_sigma": bool(o.get("sample_sigma", False)), "seed": o.get("seed", 0)})
                else:
                    p["bootstrap"].update({"replicates": o.get("replicates", 200), "method": o.get("method") or AUTO, "seed": o.get("seed", 0)})
        return validate(p)
    if settings.get("kind") == "t1":
        me = settings.get("multiexp") or {}
        p["t1"].update({"components": me.get("n", 1), "offset": bool(me.get("offset", True)), "compare": bool(me.get("compare", True))})
        p["uncertainty"]["ci_level"] = me.get("ci_level", 0.95)
        for name in ("mcmc", "bootstrap"):
            o = (options or {}).get(name) or {}
            if o:
                p[name]["enabled"] = bool(o.get("enabled", False))
                if name == "mcmc":
                    p["mcmc"].update({"walkers": o.get("nwalkers", 32), "steps": o.get("steps", 6000), "burn": o.get("burn", 2000),
                                      "sample_sigma": bool(o.get("sample_sigma", False)), "seed": o.get("seed", 0)})
                else:
                    p["bootstrap"].update({"replicates": o.get("replicates", 200), "method": o.get("method") or AUTO, "seed": o.get("seed", 0)})
        return validate(p)
    p["processing"]["echo"] = settings.get("echo", "integrate")
    ft = settings.get("ft") or {}
    if ft:
        p["ft"].update({"nucleus": ft.get("nucleus", AUTO), "G": ft.get("G", constants.DEFAULT_G_T_PER_M), "zero_fill": ft.get("zero_fill", 4),
                        "z_min": AUTO if ft.get("z_min") is None else ft["z_min"], "z_max": AUTO if ft.get("z_max") is None else ft["z_max"],
                        "spacing_um": AUTO if ft.get("spacing_um") is None else ft["spacing_um"],
                        "ilt_spacing_um": AUTO if ft.get("ilt_spacing_um") is None else ft["ilt_spacing_um"],
                        "min_signal_pct": ft.get("min_signal_pct", 5.0)})
    me, il = settings.get("multiexp") or {}, settings.get("ilt") or {}
    p["multiexp"].update({"enabled": bool(me.get("enabled", False)), "n": me.get("n", 2), "offset": me.get("offset", True), "compare": bool(me.get("compare", False))})
    p["ilt"].update({"enabled": bool(il.get("enabled", False)), "t2_min": il.get("t2_min", AUTO), "t2_max": il.get("t2_max", AUTO),
                     "max_grid_points": il.get("n_steps", 100), "alpha": il.get("alpha", 1e8), "peak_threshold_pct": il.get("peak_threshold_pct", 5.0),
                     "alpha_sweep": bool(il.get("sweep", False)), "sweep_lo": il.get("sweep_lo", 4), "sweep_hi": il.get("sweep_hi", 12)})
    p["uncertainty"]["ci_level"] = me.get("ci_level", 0.95)
    for name, key in (("mcmc", "mcmc"), ("bootstrap", "bootstrap")):
        o = (options or {}).get(key) or {}
        if not o:
            continue
        p[name]["enabled"] = bool(o.get("enabled", False))
        p["uncertainty"]["target"] = o.get("target", "bulk")
        p["uncertainty"]["z_um"] = "strongest" if o.get("z_um") is None else o["z_um"]
        if name == "mcmc":
            p["mcmc"].update({"walkers": o.get("nwalkers", 32), "steps": o.get("steps", 6000), "burn": o.get("burn", 2000),
                              "sample_sigma": bool(o.get("sample_sigma", False)), "seed": o.get("seed", 0)})
        else:
            p["bootstrap"].update({"replicates": o.get("replicates", 200), "method": o.get("method") or AUTO,
                                   "multiexp": bool(o.get("do_multiexp", True)), "ilt": bool(o.get("do_ilt", True)), "seed": o.get("seed", 0)})
    return validate(p)


def style_dict(params, titles=True):
    """The plot settings in the form figures.style_from takes. `titles` is the older output.titles switch (False = no titles)."""
    st = dict(params["style"])
    for k in ("width", "height"):
        st[k] = None if st[k] == AUTO else st[k]
    if st["title"] == "auto" and not titles:
        st["title"] = "none"
    figs = {}
    for name, o in (params.get("figures") or {}).items():
        figs[name] = {"title": o["title"], "xlim": [None if o["xmin"] == AUTO else o["xmin"], None if o["xmax"] == AUTO else o["xmax"]],
                      "ylim": [None if o["ymin"] == AUTO else o["ymin"], None if o["ymax"] == AUTO else o["ymax"]],
                      "xscale": None if o["xscale"] == AUTO else o["xscale"], "yscale": None if o["yscale"] == AUTO else o["yscale"]}
    st["figures"] = figs
    return st


def with_style(params, style):
    """params with the page's plot settings (the dict the page sends to figures.style_from) written into [style] and [figures.*]."""
    p = validate(params)
    s = style or {}
    for k in ("font_size", "line_scale", "marker_scale", "cmap", "legend", "title", "title_text", "time_unit", "decay_logy"):
        if s.get(k) is not None:
            p["style"][k] = s[k]
    for k in ("width", "height"):
        p["style"][k] = AUTO if s.get(k) in (None, "") else s[k]
    figs = {}
    for name, o in (s.get("figures") or {}).items():
        xl, yl = o.get("xlim") or [None, None], o.get("ylim") or [None, None]
        figs[name] = {"title": o.get("title") or "", "xmin": AUTO if xl[0] is None else xl[0], "xmax": AUTO if xl[1] is None else xl[1],
                      "ymin": AUTO if yl[0] is None else yl[0], "ymax": AUTO if yl[1] is None else yl[1],
                      "xscale": o.get("xscale") or AUTO, "yscale": o.get("yscale") or AUTO}
    p["figures"] = figs
    return validate(p)


# ── writing a commented TOML file ───────────────────────────────────────────

def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):
        return repr(v) if not v.is_integer() or abs(v) >= 1e15 else f"{v:.1f}"
    return str(v)


def dump_toml(params=None, header=None):
    """A commented TOML text for params (defaults if None); reading it back gives the same values."""
    p = validate(params) if params is not None else defaults()
    lines = ["# nmrdash params file. Edit the values, then run:", "#   nmrdash run params.toml <data folder>", "#   nmrdash batch params.toml <folder holding many datasets>"]
    if header:
        lines += ["#"] + [f"# {h}" for h in header.splitlines()]
    for sec, rows in SPEC.items():
        lines += ["", f"[{sec}]"]
        for k, _d, doc, _chk in rows:
            if doc:
                lines += [f"# {d}" if not d.startswith("#") else d for d in doc.split("\n")]
            lines.append(f"{k} = {_toml_value(p[sec][k])}")
    figs = p.get("figures") or {}
    if not figs:
        lines += ["", "[figures]", "# One-figure overrides go in tables named after the figure (see FIGURE_NAMES in params.py; e.g. decay_fit, ilt_spectrum, alpha_sweep,",
                  "# raw_data, spatial_extra, mcmc_trace); limits and scales apply to the single-plot figures, in plotted units (s):",
                  "# [figures.decay_fit]", "# title = \"My sample\"", "# xmin = 0.002", "# xmax = 0.01", "# yscale = \"log\"   # \"auto\", \"linear\" or \"log\""]
    for name, o in figs.items():
        lines += ["", f"[figures.{name}]"] + [f"{k} = {_toml_value(o[k])}" for k in FIGURE_DEFAULTS if o[k] != FIGURE_DEFAULTS[k]]
    return "\n".join(lines) + "\n"
