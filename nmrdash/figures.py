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
    ax.plot(t, y, ".", color=_GREY, ms=3, label="data")
    me = (res.get("multiexp") or {}).get("selected")
    if me:
        ax.plot(me["t_dense"], me["fit_dense"], color=_ORANGE, lw=1.8, label=f"{me['n']}-exponential fit")
        rax.plot(t, me["resid"], color=_ORANGE, lw=0.8)
    if mcmc and mcmc.get("predictive"):
        pr = mcmc["predictive"]
        ax.fill_between(pr["t"], pr["lo"], pr["hi"], color=_ORANGE, alpha=0.25, lw=0, label=f"MCMC {round(100 * mcmc['level'])}% band")
    il = res.get("ilt")
    if il:
        ax.plot(t, il["pred"], "--", color=_GREEN, lw=1.6, label="ILT fit")
        rax.plot(t, il["resid"], color=_GREEN, lw=0.8)
    rax.axhline(0, color=_GREY, lw=0.6)
    rax.set_xlabel("Echo time (s)")
    ax.set_ylabel("Signal (arb.)")
    rax.set_ylabel("Residual")
    for a in (ax, rax):
        _sci_axis(a)
    ax.legend(frameon=False)
    if titles:
        ax.set_title("Echo decay" + (" (FT, summed over the z-window)" if res.get("mode") == "ft" else " (phased, integrated)"))
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


def timeseries_figure(rows, titles=True):
    """Batch: each multi-exponential T2 and the total amplitude against elapsed time (or dataset number)."""
    plt = _plt()
    use_t = all(r.get("elapsed_min") is not None for r in rows)
    x = np.array([r["elapsed_min"] if use_t else i for i, r in enumerate(rows)], float)
    n = max(r["result"]["multiexp"]["n"] for r in rows)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6), sharex=True)
    for k in range(n):
        T = np.array([r["result"]["multiexp"]["components"][k]["T2"] if k < len(r["result"]["multiexp"]["components"]) else np.nan for r in rows], float)
        se = np.array([(r["result"]["multiexp"]["components"][k]["T2_se"] or np.nan) if k < len(r["result"]["multiexp"]["components"]) else np.nan for r in rows], float)
        good = np.isfinite(se) & (se <= 10 * np.abs(T))
        a1.errorbar(x, T, yerr=np.where(good, se, 0), fmt="o-", ms=4, lw=1, color=(_BLUE, _ORANGE, _GREEN, _PURPLE)[k % 4], label=f"T$_2$,{k + 1}" if n > 1 else "T$_2$")
    _log_t2_axis(a1, "y")
    a1.set_ylabel("T$_2$ (s)")
    a1.legend(frameon=False)
    a2.plot(x, [r["result"]["multiexp"]["total_amplitude"] for r in rows], "o-", ms=4, lw=1, color=_BLUE)
    a2.set_ylabel("Total amplitude (arb.)")
    _sci_axis(a2)
    a2.set_xlabel("Elapsed time (min)" if use_t else "Dataset number")
    if titles:
        a1.set_title("Multi-exponential fit vs time")
    fig.tight_layout()
    return fig
