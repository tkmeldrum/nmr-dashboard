"""matplotlib figures (rendered to PNG so the page can show them and, later, save them)."""
import base64
import io

import numpy as np


def corner_png(samples, labels, truths=None, fg="#1a1a1a", grid="#cccccc", accent="#0072B2", mark="#D55E00", bins=40):
    """Corner plot of posterior samples (n_samples, k): marginal histograms on the diagonal, 2-D densities below.

    truths : optional length-k values drawn as lines (e.g. the least-squares fit). Returns base64 PNG with a
    transparent background; fg/grid/accent are CSS colours so the figure matches the page theme.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    samples = np.asarray(samples, float)
    k = samples.shape[1]
    size = max(5.0, 1.7 * k + 1)
    fig, axes = plt.subplots(k, k, figsize=(size, size), squeeze=False)
    fig.patch.set_alpha(0)
    for i in range(k):
        for j in range(k):
            ax = axes[i][j]
            ax.patch.set_alpha(0)
            for side in ax.spines.values():
                side.set_color(grid)
            ax.tick_params(colors=fg, labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, color=accent, alpha=0.85, histtype="stepfilled")
                ax.set_yticks([])
                if truths is not None and np.isfinite(truths[i]):
                    ax.axvline(truths[i], color=mark, lw=1.2)
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap="viridis", cmin=1)
                if truths is not None and np.isfinite(truths[j]) and np.isfinite(truths[i]):
                    ax.plot(truths[j], truths[i], "s", color=mark, ms=4)
                    ax.axvline(truths[j], color=mark, lw=0.8, alpha=0.7)
                    ax.axhline(truths[i], color=mark, lw=0.8, alpha=0.7)
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[j], color=fg, fontsize=9)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(30)
            if j > 0:
                ax.set_yticklabels([])
            elif i > 0:
                ax.set_ylabel(labels[i], color=fg, fontsize=9)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── result figures (seconds, decade ticks, sci notation; the same conventions as the page) ───────────────

_BLUE, _ORANGE, _GREEN, _PURPLE, _GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7a828c"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _decade(x, _pos=None):
    if x <= 0:
        return "0"
    e = int(round(np.log10(x)))
    return f"1e{e}" if abs(x - 10.0 ** e) < 1e-9 * abs(x) else ""


def _log_t2_axis(ax, which="x"):
    from matplotlib.ticker import FuncFormatter, LogLocator
    axis = ax.xaxis if which == "x" else ax.yaxis
    getattr(ax, f"set_{which}scale")("log")
    axis.set_major_locator(LogLocator(base=10, numticks=30))
    axis.set_major_formatter(FuncFormatter(_decade))


def _sci_axis(ax, which="y"):
    from matplotlib.ticker import ScalarFormatter
    f = ScalarFormatter(useMathText=True)
    f.set_powerlimits((-2, 3))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)


def decay_figure(res, mcmc=None, titles=True):
    """Decay with the multi-exponential fit, the ILT fit and (if present) the MCMC band; residuals below."""
    plt = _plt()
    t = np.asarray(res["decay"]["t"], float)
    y = np.asarray(res["decay"]["y"], float)
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7.2, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    skip = res["decay"].get("skip", 0)
    ax.plot(t[skip:], y[skip:], ".", color=_GREY, ms=3, label="data")
    if skip:
        ax.plot(t[:skip], y[:skip], "o", mfc="none", mec=_GREY, ms=4, label=f"excluded ({skip})")
    t1 = res.get("mode") == "t1"
    me = (res.get("multiexp") or {}).get("selected")
    if me:
        ax.plot(me["t_dense"], me["fit_dense"], color=_ORANGE, lw=1.8, label=f"{me['n']}-exponential " + ("recovery fit" if t1 else "fit"))
        rax.plot(t[skip:], me["resid"], color=_ORANGE, lw=0.8)
    if mcmc and mcmc.get("predictive"):
        pr = mcmc["predictive"]
        ax.fill_between(pr["t"], pr["lo"], pr["hi"], color=_ORANGE, alpha=0.25, lw=0, label=f"MCMC {round(100 * mcmc['level'])}% band")
    il = res.get("ilt")
    if il:
        ax.plot(t[skip:], il["pred"], "--", color=_GREEN, lw=1.6, label="ILT fit")
        rax.plot(t[skip:], il["resid"], color=_GREEN, lw=0.8)
    rax.axhline(0, color="#222222", lw=1.6, zorder=1)             # bold zero line: where the residuals fall relative to 0
    rax.set_xlabel("Delay (s)" if t1 else "Echo time (s)")
    ax.set_ylabel("Signal (arb.)")
    rax.set_ylabel("Residual")
    for a in (ax, rax):
        _sci_axis(a)
    ax.legend(frameon=False)
    if titles:
        ax.set_title("Saturation recovery" if t1 else "Echo decay" + (" (FT, summed over the z-window)" if res.get("mode") == "ft" else " (phased, integrated)"))
    fig.tight_layout()
    return fig


def ilt_figure(res, titles=True):
    plt = _plt()
    il = res["ilt"]
    tau = np.asarray(il["tau"], float)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(tau, il["spectrum"], color=_GREEN, lw=1.8)
    for p in il["peaks"]:
        ax.annotate(f"{p['T2_peak']:.2e}".replace("e-0", "e-") + f"\n{100 * p['fraction']:.0f}%", (p["T2_peak"], np.interp(p["T2_peak"], tau, il["spectrum"])),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    me = (res.get("multiexp") or {}).get("selected")
    marks = [m for m in (il.get("exp_markers") or []) if m["height"] is not None]
    if me and il.get("exp_markers"):
        for m in marks:                                     # drawn at the height an ILT peak holding that amplitude would have
            ax.vlines(m["T2"], 0, m["height"], color=_ORANGE, lw=1.3, ls=":")
        ax.plot([m["T2"] for m in marks], [m["height"] for m in marks], "D", color=_ORANGE, ms=7, label=f"{me['n']}-exp components (height = amount)")
        for m in marks:
            ax.annotate(f"{100 * m['fraction']:.0f}%", (m["T2"], m["height"]), textcoords="offset points", xytext=(8, -3), fontsize=8, color=_ORANGE)
        ax.legend(frameon=False)
    _log_t2_axis(ax)
    ax.set_xlim(il["t2_min"], il["t2_max"])
    ax.set_xlabel("T$_2$ (s)")
    ax.set_ylabel("Amplitude")
    if titles:
        ax.set_title(f"T$_2$ spectrum (ILT, alpha = {il['alpha']:.0e})".replace("e+0", "e"))
    fig.tight_layout()
    return fig


def spatial_figure(res, titles=True):
    """FT mode: profile, T2(z) from the exponential fits, amplitude(z) and the ILT T2 map with its peak tracks."""
    plt = _plt()
    ft, sp = res["ft"], res.get("spatial") or {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0][0]
    ax.plot(ft["z_um"], ft["profile"], color=_BLUE)
    ax.axvspan(*ft["window"], color=_ORANGE, alpha=0.15, lw=0)
    ax.set_xlabel("z (µm)")
    ax.set_ylabel("|Σ echoes| (arb.)")
    _sci_axis(ax)
    ax.set_xlim(*ft["window"])
    if titles:
        ax.set_title("Spatial profile")
    me = sp.get("multiexp")
    if me:
        z = np.asarray(me["z_um"], float)
        ax2, ax3 = axes[0][1], axes[1][0]
        for k, (T, se, A, ase) in enumerate(zip(me["T2"], me["T2_se"], me["A"], me["A_se"])):
            col = (_BLUE, _ORANGE, _GREEN, _PURPLE)[k % 4]
            T = np.array([np.nan if v is None else v for v in T], float)
            se = np.array([np.nan if v is None else v for v in se], float)
            A = np.array([np.nan if v is None else v for v in A], float)
            ase = np.array([np.nan if v is None else v for v in ase], float)
            good = np.isfinite(se) & (se <= 10 * np.abs(T))
            ax2.errorbar(z, T, yerr=np.where(good, se, 0), fmt="o", ms=4, color=col, lw=1, label=f"T$_2$,{k + 1}" if me["n"] > 1 else "T$_2$")
            ax3.errorbar(z, A, yerr=np.where(np.isfinite(ase), ase, 0), fmt="o", ms=4, color=col, lw=1)
        _log_t2_axis(ax2, "y")
        ax2.set_xlabel("z (µm)")
        ax2.set_ylabel("T$_2$ (s)")
        ax2.legend(frameon=False)
        ax2.set_xlim(*ft["window"])
        ax3.set_xlim(*ft["window"])
        ax3.set_xlabel("z (µm)")
        ax3.set_ylabel("Amplitude (arb.)")
        _sci_axis(ax3)
        if titles:
            ax2.set_title("T$_2$ vs position (multi-exponential)")
            ax3.set_title("Amplitude vs position")
    il = sp.get("ilt")
    ax4 = axes[1][1]
    if il:
        z, tau = np.asarray(il["z_um"], float), np.asarray(il["tau"], float)
        S = np.array([[np.nan if v is None else v for v in row] for row in il["spectra"]], float)
        m = ax4.pcolormesh(z, tau, S.T, shading="auto", cmap="viridis")
        fig.colorbar(m, ax=ax4, label="f")
        for k, b in enumerate(il["bands"]):
            ax4.plot(z, [np.nan if v is None else v for v in il["band_mean_T2"][k]], lw=2, color=("white", _ORANGE, _PURPLE, "#56B4E9")[k % 4],
                     label=f"peak {b['t2_peak']:.1e}".replace("e-0", "e-"))
        _log_t2_axis(ax4, "y")
        ax4.set_xlim(*ft["window"])
        ax4.set_xlabel("z (µm)")
        ax4.set_ylabel("T$_2$ (s)")
        ax4.legend(frameon=False, fontsize=7, loc="upper right")
        if titles:
            ax4.set_title("ILT T$_2$ distribution vs position")
    else:
        ax4.axis("off")
    fig.tight_layout()
    return fig


def bootstrap_ilt_figure(boot, titles=True):
    plt = _plt()
    il = boot["ilt"]
    tau = np.asarray(il["tau"], float)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.fill_between(tau, il["lo"], il["hi"], color=_GREEN, alpha=0.25, lw=0, label=f"{round(100 * boot['level'])}% band")
    ax.plot(tau, il["median"], color=_GREEN, lw=1.8, label="bootstrap median")
    ax.plot(tau, il["spectrum"], "--", color=_BLUE, lw=1.4, label="ILT of the data")
    _log_t2_axis(ax)
    ax.set_xlabel("T$_2$ (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False)
    if titles:
        ax.set_title(f"ILT spectrum with bootstrap band ({boot['replicates']} replicates)")
    fig.tight_layout()
    return fig


def bootstrap_hist_figure(boot, titles=True):
    plt = _plt()
    me = boot["multiexp"]
    names, vals = me["samples"]["names"], np.asarray(me["samples"]["values"], float)
    k = len(names)
    cols = min(k, 3)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= k:
            ax.axis("off")
            continue
        p = me["params"][j]
        ax.hist(vals[:, j], bins=25, color=_BLUE, alpha=0.85)
        ax.axvline(p["ls"], color=_ORANGE, lw=1.3)
        for x in (p["lo"], p["hi"]):
            ax.axvline(x, color=_GREY, lw=1, ls=":")
        ax.set_xlabel(names[j])
        ax.set_yticks([])
    if titles:
        fig.suptitle(f"Bootstrap distributions ({boot['replicates']} replicates; orange = least squares, dotted = {round(100 * boot['level'])}% interval)", fontsize=9)
    fig.tight_layout()
    return fig


def save_figure(fig, path, fmt="png", dpi=150):
    import matplotlib.pyplot as plt
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── plot settings (the "Plot settings" panel on the page and [output.style] ideas for scripts) ─────────────

STYLE_DEFAULTS = {
    "font_size": 10.0,        # points; every text scales with it
    "line_scale": 1.0,        # multiplies the line widths
    "marker_scale": 1.0,      # multiplies the marker sizes
    "cmap": "viridis",        # colour map of the heat maps (the diverging residual map keeps its own)
    "width": None, "height": None,   # inches; None keeps each figure's own size
    "legend": True,
    "title": "auto",          # "auto" = each figure's own title, "none", or "custom" (with title_text)
    "title_text": "",
    "time_unit": "s",         # "s" or "ms": tick labels and axis labels only; the numbers saved to the CSV stay in seconds
    "decay_logy": False,      # log y-axis on the decay figure
    "figures": {},            # per figure: {"title": str, "xlim": [lo, hi], "ylim": [lo, hi], "xscale": "linear"|"log", "yscale": ...}
}

FIGURES = ("raw_data", "ft_profile_map", "decay_fit", "model_comparison", "ilt_spectrum", "alpha_sweep", "spatial", "spatial_extra", "mcmc_trace",
           "bootstrap_ilt", "bootstrap_distributions", "t2d_spectrum", "t2d_populations", "t2d_fit")
LIMIT_FIGURES = ("decay_fit", "ilt_spectrum", "alpha_sweep", "model_comparison", "bootstrap_ilt", "t2d_spectrum")      # one main plot to put limits on


def _pair(v):
    try:
        lo, hi = (None if x in (None, "") else float(x) for x in v)
    except (TypeError, ValueError):
        return [None, None]
    return [lo, hi]


def style_from(d):
    """A complete, validated style dict from a partial one (unknown keys are ignored)."""
    s = dict(STYLE_DEFAULTS)
    for k, v in (d or {}).items():
        if k in s and v is not None and v != "":
            s[k] = v
    s["font_size"] = min(max(float(s["font_size"]), 4.0), 40.0)
    s["line_scale"] = min(max(float(s["line_scale"]), 0.1), 10.0)
    s["marker_scale"] = min(max(float(s["marker_scale"]), 0.1), 10.0)
    for k in ("width", "height"):
        s[k] = None if s[k] is None else min(max(float(s[k]), 1.0), 40.0)
    if s["title"] not in ("auto", "none", "custom"):
        s["title"] = "auto"
    if s["time_unit"] not in ("s", "ms"):
        s["time_unit"] = "s"
    s["legend"], s["decay_logy"] = bool(s["legend"]), bool(s["decay_logy"])
    figs = {}
    for name, o in (s["figures"] or {}).items():
        if name not in FIGURES or not isinstance(o, dict):
            continue
        f = {"title": str(o.get("title") or ""), "xlim": _pair(o.get("xlim") or []), "ylim": _pair(o.get("ylim") or [])}
        for ax in ("xscale", "yscale"):
            f[ax] = o.get(ax) if o.get(ax) in ("linear", "log") else None
        figs[name] = f
    s["figures"] = figs
    import matplotlib
    if s["cmap"] not in matplotlib.colormaps:
        s["cmap"] = "viridis"
    return s


def _scale(ax, which, want):
    """Switch an axis between linear and log (None keeps what the figure has); log axes get the decade tick labels."""
    from matplotlib.ticker import ScalarFormatter
    if want is None or getattr(ax, f"get_{which}scale")() == want:
        return
    if want == "log":
        _log_t2_axis(ax, which)
    else:
        getattr(ax, f"set_{which}scale")("linear")
        (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(ScalarFormatter())


def styled(builder, style, *args, name="", retight=True):
    """Build a figure with builder(*args, titles=...) and then apply the plot settings to it."""
    plt = _plt()
    s = style_from(style)
    with plt.rc_context({"font.size": s["font_size"]}):
        fig = builder(*args, titles=s["title"] == "auto")
        _apply(fig, s, name, retight)
    return fig


def _apply(fig, s, name, retight):
    import warnings
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.text import Annotation
    from matplotlib.ticker import FuncFormatter
    for a in fig.axes:
        for ln in a.findobj(Line2D):
            ln.set_linewidth(ln.get_linewidth() * s["line_scale"])
            ln.set_markersize(ln.get_markersize() * s["marker_scale"])
        for lc in a.findobj(LineCollection):
            lc.set_linewidth([w * s["line_scale"] for w in np.atleast_1d(lc.get_linewidth())])
        for an in a.findobj(Annotation):
            an.set_fontsize(an.get_fontsize() * s["font_size"] / 10.0)      # annotations carry their own small size
    for m in fig.findobj(lambda o: isinstance(o, ScalarMappable)):
        cm = getattr(m, "cmap", None)
        if cm is not None and cm.name == "viridis" and s["cmap"] != "viridis":
            m.set_cmap(s["cmap"])
    if not s["legend"]:
        for a in fig.axes:
            if a.get_legend():
                a.get_legend().remove()
    over = s["figures"].get(name) or {}
    if s["decay_logy"] and name == "decay_fit" and fig.axes:
        fig.axes[0].set_yscale("log")
    main = fig.axes[0] if (name in LIMIT_FIGURES and fig.axes) else None
    if main is not None:
        _scale(main, "x", over.get("xscale"))
        _scale(main, "y", over.get("yscale"))
        for which, lim in (("x", over.get("xlim")), ("y", over.get("ylim"))):
            if lim and (lim[0] is not None or lim[1] is not None):
                cur = getattr(main, f"get_{which}lim")()
                getattr(main, f"set_{which}lim")(cur[0] if lim[0] is None else lim[0], cur[1] if lim[1] is None else lim[1])
    title = over.get("title") or (s["title_text"] if s["title"] == "custom" else "")
    if title:
        if len(fig.axes) > 2:
            fig.suptitle(title)
        else:
            fig.axes[0].set_title(title)
    if s["width"] or s["height"]:
        w0, h0 = fig.get_size_inches()
        fig.set_size_inches(s["width"] or w0, s["height"] or h0)
        if retight:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fig.tight_layout()
                except Exception:
                    pass
    if s["time_unit"] == "ms":
        for a in fig.axes:
            for which in ("x", "y"):
                label = getattr(a, f"get_{which}label")()
                if "(s)" not in label:
                    continue
                getattr(a, f"set_{which}label")(label.replace("(s)", "(ms)"))
                axis = a.xaxis if which == "x" else a.yaxis
                if getattr(a, f"get_{which}scale")() == "log":
                    axis.set_major_formatter(FuncFormatter(lambda v, _p: _decade(v * 1e3)))
                else:
                    axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * 1e3:.6g}"))


# ── the rest of what the page shows: raw data, FT profile and map, model comparison, alpha sweep, spatial fits, MCMC traces ──

def raw_figure(raw, kind="cpmg", titles=True):
    """The Raw data tab: echo magnitudes, one echo, the summed decay (CPMG); the recovery (T1); the echo x b-value map (T2-D)."""
    plt = _plt()
    if kind == "t1":
        fig, ax = plt.subplots(figsize=(7.2, 4))
        ax.plot(raw["t"], raw["y"], "o-", color=_BLUE, ms=4, lw=1)
        ax.set_xlabel("Delay (s)")
        ax.set_ylabel("Signal (arb.)")
        _sci_axis(ax)
        if titles:
            ax.set_title("Saturation recovery, as measured")
        fig.tight_layout()
        return fig
    if kind == "t2d":
        e, b, Z = np.asarray(raw["echo_time_s"], float), np.asarray(raw["b_vec"], float), np.asarray(raw["data"], float)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        m = ax.pcolormesh(e, b, Z, shading="auto", cmap="viridis")
        fig.colorbar(m, ax=ax, label="signal")
        ax.set_xlabel("Echo time (s)")
        ax.set_ylabel("b-value (×10$^{-9}$ s/m$^2$)")
        if titles:
            ax.set_title("T$_2$–D data, as measured")
        fig.tight_layout()
        return fig
    t = np.asarray(raw["echo_time_s"], float)
    re, im = np.asarray(raw["real"], float), np.asarray(raw["imag"], float)
    mag = np.hypot(re, im)
    fig = plt.figure(figsize=(10.5, 7))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    m = ax.pcolormesh(t, np.arange(1, mag.shape[0] + 1), mag, shading="auto", cmap="viridis")
    fig.colorbar(m, ax=ax, label="|S|")
    ax.set_xlabel("Echo time (s)")
    ax.set_ylabel("Point within echo")
    ref = min(1, mag.shape[1] - 1)                              # the second echo: the first can be distorted
    pts = np.arange(1, mag.shape[0] + 1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(pts, re[:, ref], color=_BLUE, lw=1.2, label="real")
    ax2.plot(pts, im[:, ref], color=_ORANGE, lw=1.2, label="imaginary")
    ax2.plot(pts, mag[:, ref], color=_GREEN, lw=1.2, label="magnitude")
    ax2.set_xlabel("Point within echo")
    ax2.set_ylabel("Signal (arb.)")
    ax2.legend(frameon=False)
    _sci_axis(ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, raw["summed"], ".-", color=_BLUE, ms=3, lw=0.8)
    ax3.set_xlabel("Echo time (s)")
    ax3.set_ylabel("Signal (arb.)")
    _sci_axis(ax3)
    if titles:
        ax.set_title("Echo magnitude")
        ax2.set_title(f"Echo {ref + 1}")
        ax3.set_title("Summed echo decay (phased real part)")
    fig.tight_layout()
    return fig


def ft_profile_map_figure(res, titles=True):
    """FT mode: the spatial profile with the analysed window, and the echo magnitude after the transform."""
    plt = _plt()
    ft = res["ft"]
    t = np.asarray(res["decay"]["t"], float)
    h = ft["heat"]
    w0, w1 = ft["window"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax.plot(ft["z_um"], ft["profile"], color=_BLUE)
    ax.axvspan(w0, w1, color=_ORANGE, alpha=0.15, lw=0)
    ax.set_xlim(w0, w1)
    ax.set_xlabel("z (µm)")
    ax.set_ylabel("|Σ echoes| (arb.)")
    _sci_axis(ax)
    m = ax2.pcolormesh(t[np.asarray(h["echo_idx"], int)], np.asarray(h["z_um"], float), np.asarray(h["mag"], float), shading="auto", cmap="viridis")
    fig.colorbar(m, ax=ax2, label="|S|")
    ax2.set_ylim(w1, w0)                                         # + z is down, toward the magnet
    ax2.set_xlabel("Echo time (s)")
    ax2.set_ylabel("z (µm)")
    if titles:
        ax.set_title("Spatial profile")
        ax2.set_title("Echo magnitude after FT")
    fig.tight_layout()
    return fig


def model_comparison_figure(res, titles=True):
    """How many exponentials: the AIC and BIC of each model relative to the best (0 = best)."""
    plt = _plt()
    t2d = res.get("mode") == "t2d"
    rows = res["t2d"]["model"]["comparison"] if t2d else res["multiexp"]["comparison"]
    n = [r["k"] if t2d else r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n, [r["d_aic"] for r in rows], "o-", color=_BLUE, label="ΔAIC")
    ax.plot(n, [r["d_bic"] for r in rows], "s-", color=_ORANGE, label="ΔBIC")
    ax.axhline(0, color="#222222", lw=1.2, zorder=1)
    ax.set_xticks(n)
    ax.set_xlabel("Number of populations" if t2d else "Number of components")
    ax.set_ylabel("Δ relative to the best model")
    ax.legend(frameon=False)
    if titles:
        ax.set_title("Model comparison (lower is better)")
    fig.tight_layout()
    return fig


def alpha_sweep_figure(res, titles=True):
    """The L-curve: spectrum roughness against fit RMSE for each alpha."""
    plt = _plt()
    sw = res["ilt"]["sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot([s["rmse"] for s in sw], [s["roughness"] for s in sw], "o-", color=_PURPLE)
    for s in sw:
        ax.annotate(f"α={s['alpha']:.0e}".replace("e+0", "e").replace("e+", "e"), (s["rmse"], s["roughness"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fit RMSE")
    ax.set_ylabel("Spectrum roughness ‖D$_2$x‖/‖x‖")
    if titles:
        ax.set_title("α sweep (L-curve)")
    fig.tight_layout()
    return fig


def spatial_extra_figure(res, titles=True):
    """FT mode: what spatial_figure leaves out: fit quality and the ILT peak tracks and shares along z."""
    plt = _plt()
    sp = res.get("spatial") or {}
    me, il = sp.get("multiexp"), sp.get("ilt")
    panels = (1 if me else 0) + (2 if il else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 3.8), squeeze=False)
    axes = list(axes[0])
    nan = lambda a: np.array([np.nan if v is None else v for v in a], float)
    if me:
        ax = axes.pop(0)
        ax.plot(me["z_um"], nan(me["chi2_red"]), "o-", color=_BLUE, ms=4, lw=1)
        ax.axhline(1, color=_GREY, ls=":")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("z (µm)")
        ax.set_ylabel("Reduced χ$^2$")
        if titles:
            ax.set_title("Fit quality vs position")
    if il:
        z = np.asarray(il["z_um"], float)
        colours = (_BLUE, _ORANGE, _GREEN, _PURPLE, "#56B4E9")
        a1, a2 = axes
        for k, b in enumerate(il["bands"]):
            lab = f"peak {b['t2_peak']:.1e}".replace("e-0", "e-")
            a1.plot(z, nan(il["band_mean_T2"][k]), "o-", ms=3, lw=1, color=colours[k % 5], label=lab)
            a2.plot(z, nan(il["band_fraction"][k]), color=colours[k % 5], lw=1.6, label=lab)
        _log_t2_axis(a1, "y")
        a1.set_xlabel("z (µm)")
        a1.set_ylabel("T$_2$ (s)")
        a1.legend(frameon=False, fontsize=7)
        a2.set_ylim(0, 1)
        a2.set_xlabel("z (µm)")
        a2.set_ylabel("Fraction")
        if titles:
            a1.set_title("ILT peak T$_2$ vs position")
            a2.set_title("Share of the signal in each peak")
    fig.tight_layout()
    return fig


def mcmc_trace_figure(mcmc, titles=True):
    """A few walkers per parameter, with the burn-in shaded: a healthy chain looks like stationary noise."""
    plt = _plt()
    tr = mcmc["trace"]
    names, vals, step, burn = tr["names"], np.asarray(tr["values"], float), np.asarray(tr["step"], float), tr["burn"]
    k = len(names)
    fig, axes = plt.subplots(k, 1, figsize=(7.2, 1.6 * k + 0.6), sharex=True, squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        ax.plot(step, vals[:, :, j], lw=0.5, color=_BLUE, alpha=0.6)         # steps x walkers x parameters
        ax.axvspan(0, burn, color=_GREY, alpha=0.2, lw=0)
        ax.set_ylabel(names[j], fontsize=8)
    axes[-1, 0].set_xlabel("Step")
    if titles:
        axes[0, 0].set_title("MCMC traces (grey = burn-in)")
    fig.tight_layout()
    return fig


def timeseries_figure(rows, titles=True, symbol="T2", x=None, xlabel=None, xticklabels=None):
    """Batch: each multi-exponential time constant (T2 or T1) and the total amplitude against elapsed time (or dataset number)."""
    plt = _plt()
    use_t = x is None and all(r.get("elapsed_min") is not None for r in rows)
    x = np.array(x if x is not None else [r["elapsed_min"] if use_t else i for i, r in enumerate(rows)], float)
    n = max(r["result"]["multiexp"]["n"] for r in rows)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6), sharex=True)
    for k in range(n):
        T = np.array([r["result"]["multiexp"]["components"][k]["T2"] if k < len(r["result"]["multiexp"]["components"]) else np.nan for r in rows], float)
        se = np.array([(r["result"]["multiexp"]["components"][k]["T2_se"] or np.nan) if k < len(r["result"]["multiexp"]["components"]) else np.nan for r in rows], float)
        good = np.isfinite(se) & (se <= 10 * np.abs(T))
        d = symbol[1:]
        a1.errorbar(x, T, yerr=np.where(good, se, 0), fmt="o-", ms=4, lw=1, color=(_BLUE, _ORANGE, _GREEN, _PURPLE)[k % 4], label=f"T$_{d}$,{k + 1}" if n > 1 else f"T$_{d}$")
    _log_t2_axis(a1, "y")
    a1.set_ylabel(f"T$_{symbol[1:]}$ (s)")
    a1.legend(frameon=False)
    a2.plot(x, [r["result"]["multiexp"]["total_amplitude"] for r in rows], "o-", ms=4, lw=1, color=_BLUE)
    a2.set_ylabel("Total amplitude (arb.)")
    _sci_axis(a2)
    a2.set_xlabel(xlabel or ("Elapsed time (min)" if use_t else "Dataset number"))
    if xticklabels is not None:
        a2.set_xticks(x)
        a2.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=8)
    if titles:
        a1.set_title("Multi-exponential fit vs " + ("array" if xticklabels is not None else "time"))
    fig.tight_layout()
    return fig


def timeseries_t2d_figure(rows, titles=True):
    """Batch of T2-D datasets: each population's T2, D and share against elapsed time (populations are matched by their T2 order)."""
    plt = _plt()
    use_t = all(r.get("elapsed_min") is not None for r in rows)
    x = np.array([r["elapsed_min"] if use_t else i for i, r in enumerate(rows)], float)
    pops = [r["result"]["t2d"]["model"]["populations"] for r in rows]
    n = max(len(p) for p in pops)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    colours = (_BLUE, _ORANGE, _GREEN, _PURPLE)
    nan = lambda v: np.nan if v is None else v
    for k in range(n):
        get = lambda key: np.array([nan(p[k][key]) if k < len(p) else np.nan for p in pops], float)
        T, Tse, D, Dse, fr, fse = get("T2"), get("T2_se"), get("D"), get("D_se"), get("fraction"), get("fraction_se")
        c = colours[k % 4]
        a1.errorbar(x, T, yerr=np.where(np.isfinite(Tse), Tse, 0), fmt="o-", ms=4, lw=1, color=c, label=f"population {k + 1}")
        bound = np.array([bool(k < len(p) and p[k]["D_at_limit"]) for p in pops])
        a2.errorbar(x[~bound], D[~bound], yerr=np.where(np.isfinite(Dse), Dse, 0)[~bound], fmt="o-", ms=4, lw=1, color=c)
        if bound.any():
            a2.plot(x[bound], D[bound], "v", ms=7, mfc="none", color=c)                    # only a bound (open triangle)
        a3.errorbar(x, fr, yerr=np.where(np.isfinite(fse), fse, 0), fmt="o-", ms=4, lw=1, color=c)
    _log_t2_axis(a1, "y")
    _log_t2_axis(a2, "y")
    a1.set_ylabel("T$_2$ (s)")
    a2.set_ylabel("D (m$^2$/s)")
    a3.set_ylabel("Fraction")
    a3.set_ylim(0, 1)
    a3.set_xlabel("Elapsed time (min)" if use_t else "Dataset number")
    a1.legend(frameon=False)
    if titles:
        a1.set_title("T$_2$–D populations vs time (open triangles: D is only a bound)")
    fig.tight_layout()
    return fig


def t2d_spectrum_figure(res, titles=True):
    """The T2-D map (T2 across, D up, both log) with its two projections."""
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    r = res["t2d"]
    T2, D, S = np.asarray(r["T2"], float), np.asarray(r["D"], float), np.asarray(r["spectrum"], float)
    fig = plt.figure(figsize=(7.6, 6.4))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    ax, top, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1])
    m = ax.pcolormesh(T2, D, S, shading="auto", cmap="viridis")
    for p in r["peaks"][:6]:
        ax.plot(p["T2_mean"], p["D_mean"], "o", mfc="none", mec="white", ms=6)
        ax.annotate(f"{100 * p['fraction']:.0f}%", (p["T2_mean"], p["D_mean"]), textcoords="offset points", xytext=(6, 4), color="white", fontsize=8)
    _log_t2_axis(ax, "x")
    _log_t2_axis(ax, "y")
    lo, hi = r["d_sensitive"]                                  # where the b-values can place a diffusion coefficient
    for a, b in ((D[0] / 10, lo), (hi, D[-1] * 10)):
        if b > a:
            ax.axhspan(a, b, color="white", alpha=0.45, lw=0, hatch="///", zorder=3)
    ax.set_ylim(D[0], D[-1])
    ax.set_xlabel("T$_2$ (s)")
    ax.set_ylabel("D (m$^2$/s)")
    top.plot(T2, r["projection_T2"], color=_BLUE)
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    top.set_xticklabels([])
    top.set_ylabel("Σ over D")
    right.plot(r["projection_D"], D, color=_BLUE)
    right.set_yscale("log")
    right.set_ylim(ax.get_ylim())
    right.set_yticklabels([])
    right.set_xlabel("Σ over T$_2$")
    from matplotlib.ticker import MaxNLocator
    right.xaxis.set_major_locator(MaxNLocator(2, prune="lower"))
    fig.colorbar(m, ax=[top, right], location="right", fraction=0.04, pad=0.02).set_label("amplitude")
    if titles:
        fig.suptitle(f"T$_2$–D spectrum (α = {r['alpha']:.0e})".replace("e+0", "e"))
    return fig


def t2d_populations_figure(res, mcmc=None, boot=None, titles=True):
    """The few-population model on the regularised map: each population with its interval, and the bootstrap / MCMC samples around it."""
    plt = _plt()
    r = res["t2d"]
    m = r["model"]
    T2, D, S = np.asarray(r["T2"], float), np.asarray(r["D"], float), np.asarray(r["spectrum"], float)
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.pcolormesh(T2, D, S, shading="auto", cmap="Greys", alpha=0.45)
    colours = (_BLUE, _ORANGE, _GREEN, _PURPLE)
    from matplotlib.lines import Line2D
    handles = []
    for result, marker, label, ms, alpha in ((boot, "x", "bootstrap", 4.5, 0.7), (mcmc, ".", "MCMC", 2.5, 0.25)):     # a different marker per method
        sc = (result or {}).get("scatter")
        if not sc:
            continue
        for k, (t_, d_) in enumerate(zip(sc["T2"], sc["D"])):
            ax.plot(t_, d_, marker, ms=ms, alpha=alpha, color=colours[k % 4], mec=None if marker == "x" else "none", mew=0.8)
        handles.append(Line2D([], [], ls="", marker=marker if marker == "x" else "o", ms=7 if marker == "x" else 6, color="#222222", label=label))
    lo, hi = m["d_window"]
    for a_, b_ in ((D[0] / 10, lo), (hi, D[-1] * 10)):
        if b_ > a_:
            ax.axhspan(a_, b_, color="white", alpha=0.45, lw=0, hatch="///", zorder=3)
    for i, p in enumerate(m["populations"]):
        c = colours[i % 4]
        xe = None if p["T2_ci"][0] is None else [[p["T2"] - p["T2_ci"][0]], [p["T2_ci"][1] - p["T2"]]]
        ye = None if p["D_ci"][0] is None else [[p["D"] - p["D_ci"][0]], [p["D_ci"][1] - p["D"]]]
        bound = p["D_at_limit"] or p["T2_at_limit"]
        ax.errorbar([p["T2"]], [p["D"]], xerr=xe, yerr=ye, fmt="v" if p["D_at_limit"] == "low" else "^" if p["D_at_limit"] == "high" else "o", ms=8, color=c, mec="black", mfc=c if p["supported"] and not bound else "none", capsize=3, lw=1.4, zorder=5)
        ax.annotate(f"{i + 1}: {100 * p['fraction']:.0f}%", (p["T2"], p["D"]), textcoords="offset points", xytext=(8, 6), fontsize=8, zorder=6)
    _log_t2_axis(ax, "x")
    _log_t2_axis(ax, "y")
    xs = [p["T2"] for p in m["populations"]] + [v for res_ in (boot, mcmc) if res_ and res_.get("scatter") for arr in res_["scatter"]["T2"] for v in (np.min(arr), np.max(arr))]
    ax.set_xlim(max(T2[0], min(xs) / 4), min(T2[-1], max(xs) * 4))             # around the populations: most of the grid is empty
    ax.set_ylim(D[0], D[-1])
    ax.set_xlabel("T$_2$ (s)")
    ax.set_ylabel("D (m$^2$/s)")
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper left")
    if titles:
        ax.set_title(f"T$_2$–D populations ({m['K']}-population model on the regularised map)")
    fig.tight_layout()
    return fig


def t2d_fit_figure(res, titles=True):
    """Data, fit and residual as maps over echo time and b-value (the residual should look like noise)."""
    plt = _plt()
    r = res["t2d"]
    e, b = np.asarray(r["echo_s"], float), np.asarray(r["b_vec"], float)
    pred, resid = np.asarray(r["pred"], float), np.asarray(r["resid"], float)
    data = pred + resid
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    vmax = np.nanmax(np.abs(data))
    for ax, Z, name, cmap, lim in ((axes[0], data, "data", "viridis", (data.min(), vmax)), (axes[1], pred, "fit", "viridis", (data.min(), vmax)),
                                   (axes[2], resid, "residual", "RdBu_r", (-np.nanmax(np.abs(resid)), np.nanmax(np.abs(resid))))):
        m = ax.pcolormesh(e, b, Z, shading="auto", cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_xlabel("Echo time (s)")
        fig.colorbar(m, ax=ax)
        if titles:
            ax.set_title(name)
    axes[0].set_ylabel("b-value (×10$^{-9}$ s/m$^2$)")
    fig.tight_layout()
    return fig
