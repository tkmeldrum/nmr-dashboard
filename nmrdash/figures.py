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

FIGURES = ("decay_fit", "ilt_spectrum", "spatial", "bootstrap_ilt", "bootstrap_distributions", "t2d_spectrum", "t2d_fit")
LIMIT_FIGURES = ("decay_fit", "ilt_spectrum", "bootstrap_ilt", "t2d_spectrum")      # one main plot to put limits on


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


def timeseries_figure(rows, titles=True, symbol="T2"):
    """Batch: each multi-exponential time constant (T2 or T1) and the total amplitude against elapsed time (or dataset number)."""
    plt = _plt()
    use_t = all(r.get("elapsed_min") is not None for r in rows)
    x = np.array([r["elapsed_min"] if use_t else i for i, r in enumerate(rows)], float)
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
    a2.set_xlabel("Elapsed time (min)" if use_t else "Dataset number")
    if titles:
        a1.set_title("Multi-exponential fit vs time")
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
