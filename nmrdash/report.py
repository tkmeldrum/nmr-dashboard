"""A per-analysis PDF report: US Letter, 2-4 pages, built with matplotlib's multi-page PDF backend (vector text, embedded figures).

Page 1: what was analysed, the settings used, the key numbers with their intervals, and every warning. Then the main figures. The last page:
how the numbers were obtained (the physics and estimation relations that apply) and the provenance. The content is composed first as a list of
blocks (compose), flowed onto pages (paginate), and only then drawn (render), so the text can be checked without drawing anything.
"""
import base64
import io
import textwrap

import numpy as np

from . import figures, formatting as F, inference, physics

PAGE_W, PAGE_H = 8.5, 11.0                   # US Letter, inches
MARGIN = 0.65
CONTENT_W = PAGE_W - 2 * MARGIN
FOOTER_H = 0.4
CONTENT_H = PAGE_H - 2 * MARGIN - FOOTER_H
MAX_PAGES = 4
FIG_DPI = 170
SIZES = {"title": 17, "h2": 11.5, "para": 8.6, "note": 7.8, "table": 8.0, "footer": 7.2}
INK, MUTED, RULE, SHADE, WARN = "#1a1a1a", "#5a6270", "#c9ced6", "#f1f3f6", "#b45309"


# ── measuring ────────────────────────────────────────────────────────────────

def _lh(size):
    """Line height in inches for a font size in points."""
    return size * 1.32 / 72.0


def _chars(width_in, size):
    """How many characters of DejaVu Sans fit in a width (a conservative average)."""
    return max(8, int(width_in * 72.0 / (size * 0.54)))


def _wrap(text, width_in, size):
    out = []
    for para in str(text).split("\n"):
        out += textwrap.wrap(para, _chars(width_in, size), break_long_words=True) or [""]
    return out


# ── blocks ───────────────────────────────────────────────────────────────────

def title(text, sub=None):
    return {"type": "title", "text": text, "sub": sub}


def h2(text):
    return {"type": "h2", "text": text}


def para(text, size="para", color=INK, bold=False):
    return {"type": "para", "text": text, "size": size, "color": color, "bold": bold}


def bullets(items, size="para", color=INK):
    return {"type": "bullets", "items": list(items), "size": size, "color": color}


def table(header, rows, widths, size="table"):
    return {"type": "table", "header": header, "rows": [[str(c) for c in r] for r in rows], "widths": widths, "size": size}


def image(arr, width=CONTENT_W, optional=False, caption=None):
    return {"type": "image", "array": arr, "width": width, "optional": optional, "caption": caption}


def spacer(h=0.12):
    return {"type": "spacer", "h": h}


def height(b, scale=1.0):
    t = b["type"]
    if t == "title":
        return _lh(SIZES["title"]) + (_lh(SIZES["h2"]) + 0.05 if b["sub"] else 0) + 0.12
    if t == "h2":
        return _lh(SIZES["h2"]) + 0.16
    if t == "para":
        s = SIZES[b["size"]]
        return len(_wrap(b["text"], CONTENT_W, s)) * _lh(s) + 0.07
    if t == "bullets":
        s = SIZES[b["size"]]
        return sum(len(_wrap(i, CONTENT_W - 0.18, s)) for i in b["items"]) * _lh(s) + 0.04 * len(b["items"]) + 0.05
    if t == "table":
        s = SIZES[b["size"]]
        rows = [b["header"]] + b["rows"] if b["header"] else b["rows"]
        return sum(max(len(_wrap(c, w - 0.1, s)) for c, w in zip(r, b["widths"])) * _lh(s) + 0.07 for r in rows) + 0.08
    if t == "image":
        a = b["array"]
        w = b["width"] * scale * b.get("fit", 1.0)
        return w * a.shape[0] / a.shape[1] + (_lh(SIZES["note"]) if b["caption"] else 0) + 0.1
    return b["h"]


MIN_FIT = 0.55           # a figure may shrink to this fraction of its size to fill the rest of a page instead of leaving it empty


def paginate(blocks, scale=1.0):
    """Flow the blocks onto pages. A heading is never left at the bottom of a page; tables and images are not split (a figure may shrink to fit the space left)."""
    pages, cur, used = [], [], 0.0
    for i, b in enumerate(blocks):
        h = height(b, scale)
        nxt = blocks[i + 1] if b["type"] == "h2" and i + 1 < len(blocks) else None
        keep = (height(nxt, scale) * (MIN_FIT if nxt["type"] == "image" else 1.0)) if nxt else 0.0
        extra = (_lh(SIZES["note"]) if b.get("caption") else 0) + 0.1 if b["type"] == "image" else 0.0       # caption and padding do not shrink with the figure
        if b["type"] == "image" and cur and used + h > CONTENT_H and CONTENT_H - used - extra >= MIN_FIT * (h - extra):
            b = dict(b, fit=(CONTENT_H - used - extra - 0.02) / (h - extra))                                 # shrink into the space that is left
            h = height(b, scale)
        elif cur and used + h + keep > CONTENT_H:
            pages.append(cur)
            cur, used = [], 0.0
        if b["type"] == "spacer" and not cur:
            continue
        cur.append(b)
        used += h
    if cur:
        pages.append(cur)
    return pages


def fit(blocks, max_pages=MAX_PAGES):
    """Pages within the limit: shrink the figures a little, then drop the optional ones, last first."""
    blocks = list(blocks)
    for scale in (1.0, 0.9, 0.8, 0.72):
        pages = paginate(blocks, scale)
        if len(pages) <= max_pages:
            return pages, scale
    while len(pages) > max_pages:
        opt = [i for i, b in enumerate(blocks) if b["type"] == "image" and b["optional"]]
        if not opt:
            break
        blocks.pop(opt[-1])
        pages = paginate(blocks, 0.72)
    return pages, 0.72


# ── drawing ──────────────────────────────────────────────────────────────────

def render(pages, scale, footer_left, footer_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    buf = io.BytesIO()
    with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        with PdfPages(buf, metadata={"Title": footer_id, "Creator": "nmrdash"}) as pdf:
            for n, blocks in enumerate(pages, 1):
                fig = plt.figure(figsize=(PAGE_W, PAGE_H))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.set_xlim(0, PAGE_W)
                ax.set_ylim(PAGE_H, 0)
                ax.axis("off")
                y = MARGIN
                T = lambda x, yy, s, size, color=INK, weight="normal", ha="left": ax.text(x, yy, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="top")
                for b in blocks:
                    t = b["type"]
                    if t == "title":
                        T(MARGIN, y, b["text"], SIZES["title"], weight="bold")
                        y += _lh(SIZES["title"])
                        if b["sub"]:
                            T(MARGIN, y + 0.03, b["sub"], SIZES["h2"], MUTED)
                            y += _lh(SIZES["h2"]) + 0.05
                        ax.plot([MARGIN, PAGE_W - MARGIN], [y + 0.05, y + 0.05], color=RULE, lw=0.8)
                        y += 0.12
                    elif t == "h2":
                        y += 0.06
                        T(MARGIN, y, b["text"], SIZES["h2"], weight="bold")
                        y += _lh(SIZES["h2"]) + 0.1
                    elif t == "para":
                        s = SIZES[b["size"]]
                        for line in _wrap(b["text"], CONTENT_W, s):
                            T(MARGIN, y, line, s, b["color"], "bold" if b["bold"] else "normal")
                            y += _lh(s)
                        y += 0.07
                    elif t == "bullets":
                        s = SIZES[b["size"]]
                        for item in b["items"]:
                            lines = _wrap(item, CONTENT_W - 0.18, s)
                            T(MARGIN + 0.04, y, "•", s, b["color"])
                            for line in lines:
                                T(MARGIN + 0.18, y, line, s, b["color"])
                                y += _lh(s)
                            y += 0.04
                        y += 0.05
                    elif t == "table":
                        s = SIZES[b["size"]]
                        x0 = [MARGIN + sum(b["widths"][:i]) for i in range(len(b["widths"]))]
                        rows = ([("head", b["header"])] if b["header"] else []) + [("row", r) for r in b["rows"]]
                        for k, (kind, r) in enumerate(rows):
                            cells = [_wrap(c, w - 0.1, s) for c, w in zip(r, b["widths"])]
                            rh = max(len(c) for c in cells) * _lh(s) + 0.07
                            if kind == "head" or k % 2 == 1:
                                ax.add_patch(Rectangle((MARGIN, y - 0.02), sum(b["widths"]), rh, color=SHADE if kind == "row" else "#e3e7ee", lw=0))
                            for x, lines in zip(x0, cells):
                                for j, line in enumerate(lines):
                                    T(x + 0.05, y + 0.02 + j * _lh(s), line, s, INK, "bold" if kind == "head" else "normal")
                            y += rh
                        y += 0.08
                    elif t == "image":
                        a = b["array"]
                        w = b["width"] * scale * b.get("fit", 1.0)
                        h = w * a.shape[0] / a.shape[1]
                        x = MARGIN + (CONTENT_W - w) / 2
                        ax.imshow(a, extent=(x, x + w, y + h, y), interpolation="antialiased", aspect="auto", zorder=1)
                        y += h
                        if b["caption"]:
                            T(PAGE_W / 2, y, b["caption"], SIZES["note"], MUTED, ha="center")
                            y += _lh(SIZES["note"])
                        y += 0.1
                    else:
                        y += b["h"]
                ax.set_xlim(0, PAGE_W)
                ax.set_ylim(PAGE_H, 0)
                ax.plot([MARGIN, PAGE_W - MARGIN], [PAGE_H - MARGIN - 0.22] * 2, color=RULE, lw=0.6)
                T(MARGIN, PAGE_H - MARGIN - 0.17, footer_left, SIZES["footer"], MUTED)
                T(PAGE_W - MARGIN, PAGE_H - MARGIN - 0.17, f"page {n} of {len(pages)}", SIZES["footer"], MUTED, ha="right")
                pdf.savefig(fig)
                plt.close(fig)
    return buf.getvalue()


# ── content ──────────────────────────────────────────────────────────────────

def _png_array(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _figure(builder, style, *args, name=""):
    return _png_array(figures.styled(builder, {**(style or {}), "title": "none"}, *args, name=name))


def _png_b64(b64):
    import matplotlib.pyplot as plt
    return plt.imread(io.BytesIO(base64.b64decode(b64)))


def _sci(name):
    return "T2" in name or "T1" in name or name.startswith("D") or name.startswith("mean D")


def collect_warnings(res, mcmc=None, boot=None, extra=()):
    w = list(extra)
    me = (res.get("multiexp") or {}).get("selected")
    if me:
        w += list(me.get("warnings", []))
        rec = res["multiexp"].get("recommended_n")
        if rec is not None and rec != me["n"] and len(res["multiexp"].get("comparison", [])) > 1:
            w.append(f"BIC prefers {rec} component{'s' if rec > 1 else ''} over the {me['n']} chosen (see the model comparison).")
    if res.get("mode") == "t1":
        w.append("No imaginary channel in these data, so there is no measured noise level: reduced χ² is not available.")
    t = res.get("t2d")
    if t:
        m = t.get("model")
        if m:
            w += list(m.get("warnings", []))
        bc = t.get("b_check") or {}
        if bc and not bc.get("agrees", True):
            w.append("The calculated delays, echo times or b-values do not match the spectrometer's files: " + "; ".join(
                f"{c['what']} vs {c['file']}" for c in bc["compared"] if not c["agrees"]) + ". The calculated values were used.")
        frac = t.get("fraction_in_sensitive_window")
        if frac is not None and frac < 0.9:
            w.append(f"Only {F.pct(frac, 0)} of the T₂–D map lies where the b-values can constrain it; signal outside that range is extrapolation, not measurement.")
    for r in (mcmc, boot):
        if r:
            w += list(r.get("warnings", []))
    seen, out, edge = set(), [], {}
    for x in w:
        if x.startswith("The least-squares fit puts"):                       # MCMC and the bootstrap each say it: say it once
            key = x.split(" at the ")[0]
            if key in edge:
                out[edge[key]] += " The same holds for the bootstrap and MCMC intervals."
                continue
            edge[key] = len(out)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _settings_rows(res, prov):
    s = res.get("settings") or {}
    mode = res.get("mode")
    rows = []
    skip = (res.get("decay") or {}).get("skip", 0)
    if mode == "t2d":
        t, tm = s.get("t2d") or {}, s.get("t2d_model") or {}
        rows += [("T₂–D inversion", f"{F.fmt_sci(res['t2d']['t2_min'])}–{F.fmt_sci(res['t2d']['t2_max'])} s × {F.fmt_sci(res['t2d']['d_min'])}–{F.fmt_sci(res['t2d']['d_max'])} m²/s, "
                                    f"{res['t2d']['n_T2']} × {res['t2d']['n_D']} points, α = {F.fmt_sci(res['t2d']['alpha'])}"),
                 ("Gradient G", f"{F.fmt(res['t2d']['b_info']['gradient_t_per_m'], 4)} T/m (b-values calculated from it, Δ and τ)")]
        if tm.get("enabled"):
            rows.append(("Few-population model", f"{'automatic (BIC)' if tm.get('k') is None else str(tm['k'])} population(s); D allowed in {'the resolvable range' if tm.get('d_window', 'resolvable') == 'resolvable' else 'the map range'}"))
    else:
        if mode == "t1":
            rows.append(("Analysis", "saturation recovery"))
        else:
            rows.append(("Echo processing", "FT of each echo (signal vs position)" if mode == "ft" else "integrate each echo (autophased)"))
        me = s.get("multiexp") or {}
        if me.get("enabled"):
            rows.append(("Exponential fit", f"{me.get('n')} component(s), offset {'on' if me.get('offset') else 'off'}, comparison {'on' if me.get('compare') else 'off'}"))
        il = s.get("ilt") or {}
        if il.get("enabled"):
            rows.append(("ILT", f"T₂ {F.fmt_sci(il.get('t2_min'))}–{F.fmt_sci(il.get('t2_max'))} s, α = {F.fmt_sci(il.get('alpha'))}, up to {il.get('n_steps')} points" + (", α sweep" if il.get("sweep") else "")))
        ft = s.get("ft")
        if mode == "ft" and ft:
            rows.append(("FT", f"{ft.get('nucleus')}, G = {F.fmt(ft.get('G'), 4)} T/m, zero-fill ×2^{ft.get('zero_fill')}"))
    if skip:
        rows.append(("Excluded", f"the first {skip} {'delays' if mode == 't1' else 'echoes'} (shown as open markers)"))
    opts = prov.get("uncertainty_options") or {}
    mc, bs = opts.get("mcmc") or {}, opts.get("bootstrap") or {}
    if mc.get("enabled"):
        rows.append(("MCMC", f"{mc.get('nwalkers')} walkers × {mc.get('steps')} steps, burn-in {mc.get('burn')}"))
    if bs.get("enabled"):
        rows.append(("Bootstrap", f"{bs.get('replicates')} replicates, noise: {bs.get('method') or 'measured'}"))
    return rows


def _compare_rows(mcmc, boot):
    """Parameter, least squares, bootstrap, MCMC: the methods side by side (as on the page)."""
    bm = None if not boot else (boot.get("multiexp") or (boot if boot.get("kind") == "bootstrap_t2d" else None))
    names = []
    for r in ([mcmc] if mcmc else []) + ([bm] if bm else []):
        for p in r["params"] + r["derived"]:
            if p["name"] not in names and p["name"] != "sigma":
                names.append(p["name"])
    find = lambda r, n: next((p for p in (r["params"] + r["derived"]) if p["name"] == n), None) if r else None
    rows = []
    for n in names:
        m, b = find(mcmc, n), find(bm, n)
        ref = m or b
        ls = ref.get("ls")
        sci = _sci(n)
        lsc = "–" if ls is None else (F.pair(ls, ref.get("ls_se"), sci) if F.ok_unc(ls, ref.get("ls_se")) else F.fmt_sci(ls) if sci else F.fmt(ls))
        cell = lambda p: "–" if p is None else F.interval(p["median"], p["lo"], p["hi"], p.get("sd"), sci)
        rows.append([n, lsc, cell(b), cell(m)])
    return rows


def _header(res, prov, kind):
    acq = prov.get("acquisition", {})
    experiment = acq.get("experiment", "?")
    analysis = {"t1": "saturation recovery", "t2d": "T₂–D correlation (2D inversion and few-population model)", "ft": "FT of each echo (signal vs position)",
                "integrate": "each echo integrated"}.get(res.get("mode") if res.get("mode") in ("t1", "t2d", "ft") else "integrate")
    when = next((f["modified_utc"] for f in prov["inputs"] if f["name"] in ("data.2d", "data2D_Cp.2d", "data2D.dat", "data.dat")), None)
    nm = prov.get("nmrdash", {})
    blocks = [title("nmrdash analysis report", prov.get("dataset")),
              table(None, [["Sequence", f"{experiment}", "Data written", when or "unknown"], ["Analysis", analysis, "Processed", prov.get("created_utc", "?")],
                           ["Tool", f"nmrdash {nm.get('version', '?')} ({(nm.get('commit') or 'unknown')[:10]})", "Dataset folder", prov.get("dataset") or "?"]], [1.0, 2.6, 1.0, 2.6], size="note")]
    return blocks


def _component_table(res, sym):
    me = (res.get("multiexp") or {}).get("selected")
    if not me:
        return []
    lvl = round(100 * me["ci_level"])
    rows = []
    for i, c in enumerate(me["components"], 1):
        rows.append([i, F.pair(c["T2"], c["T2_se"], sci=True), F.interval(c["T2"], *c["T2_ci"], se=c["T2_se"], sci=True) if F.ok_unc(c["T2"], c["T2_se"]) else "undetermined",
                     F.fmt(c["A"]), F.pct(c["fraction"]) + (f" ± {F.pct(c['fraction_se'])}" if F.ok_unc(c["fraction"], c["fraction_se"]) else "")])
    shape = ("S(t) = " + " + ".join((f"A{'' if me['n'] == 1 else i}·(1 − exp(−t/{sym}{'' if me['n'] == 1 else ',' + str(i)}))" if sym == "T1" else
                                   f"A{'' if me['n'] == 1 else i}·exp(−t/{sym}{'' if me['n'] == 1 else ',' + str(i)})") for i in range(1, me["n"] + 1)) + (" + c" if me["offset"] else ""))
    stats = f"RSS {F.fmt_sci(me['rss'])};  residual SD / signal range {F.pct(me['resid_sd_over_amp'], 2)};  noise σ {F.fmt(me['noise_sigma'])};  reduced χ² {F.fmt(me['chi2_red'])};  R² {F.fmt(me['r2'], 5)};  dof {me['dof']}"
    return [h2("Exponential fit" if sym == "T2" else "Saturation-recovery fit"), para(shape, color=MUTED, size="note"),
            table(["#", f"{sym} (s) ± SE", f"{lvl}% interval", "amplitude", "fraction"], rows, [0.4, 2.0, 2.2, 1.2, 1.4]), para(stats, size="note", color=MUTED)]


def _compose_decay(res, prov, mcmc, boot, raw, style, kind):
    sym = "T1" if res.get("mode") == "t1" else "T2"
    B = _header(res, prov, kind)
    B += [h2("Settings"), table(None, [[a, b] for a, b in _settings_rows(res, prov)], [1.5, 5.7])]
    B += _component_table(res, sym)
    il = res.get("ilt")
    if il:
        B += [h2("ILT"), para(f"α = {F.fmt_sci(il['alpha'])}, {il['n_steps']} grid points, T₂ {F.fmt_sci(il['t2_min'])}–{F.fmt_sci(il['t2_max'])} s; RMSE / signal range {F.pct(il['rmse_over_amp'], 2)}; χ² per point {F.fmt(il['chi2_per_point'])}", size="note", color=MUTED),
              table(["peak T₂ (s)", "mean T₂ of peak (s)", "fraction"], [[F.fmt_sci(p["T2_peak"]), F.fmt_sci(p["T2_mean"]), F.pct(p["fraction"])] for p in il["peaks"]] or [["none", "", ""]], [2.0, 2.4, 1.6])]
    ft = res.get("ft")
    if ft:
        B += [h2("Spatial analysis (FT)"), para(f"{ft['nucleus']}, G = {F.fmt(ft['G'], 4)} T/m, zero-fill ×2^{ft['zero_fill']} → {ft['nfft']} points (Δz = {F.fmt(ft['zf_step_um'])} µm); natural resolution {F.fmt(ft['resolution_um'])} µm; "
                                                f"window {F.fmt(ft['window'][0])} to {F.fmt(ft['window'][1])} µm; + z is toward the magnet. Positions below {F.fmt(ft['min_signal_pct'])}% of the profile maximum are not fitted.", size="note", color=MUTED)]
    if mcmc or boot:
        rows = _compare_rows(mcmc, boot)
        d = (mcmc or {}).get("diagnostics")
        B += [h2("Uncertainty: methods compared"),
              para(("MCMC: " + f"{mcmc['nwalkers']} walkers × {mcmc['steps_run']} steps, acceptance {F.fmt(d['acceptance'], 2)}, autocorrelation time {F.fmt(d['tau_max'])} steps. " if mcmc else "") +
                   (f"Bootstrap: {boot['replicates']} replicates, noise: {boot['method']}. " if boot else "") + "Intervals are at the chosen confidence level.", size="note", color=MUTED),
              table(["parameter", "least squares ± SE", "bootstrap", "MCMC"], rows, [1.2, 2.0, 2.0, 2.0], size="note")]
    ws = collect_warnings(res, mcmc, boot)
    B += [h2("Warnings and notes")] + ([bullets(ws, color=WARN)] if ws else [para("None.", color=MUTED)])
    # figures
    B.append(h2("Figures"))
    if raw:
        B.append(image(_figure(lambda r, titles: figures.raw_figure(r, "t1" if res.get("mode") == "t1" else "cpmg", titles), style, raw, name="raw_data"), width=5.6 if res.get("mode") != "t1" else 4.4, optional=True, caption="Raw data as measured"))
    B.append(image(_figure(lambda r, b, titles: figures.decay_figure(r, b, titles), style, res, None, name="decay_fit"), width=5.4, caption="Decay, fits and residuals"))
    if il:
        B.append(image(_figure(figures.ilt_figure, style, res, name="ilt_spectrum"), width=5.2, caption="T₂ spectrum (ILT)"))
        if il.get("sweep"):
            B.append(image(_figure(figures.alpha_sweep_figure, style, res, name="alpha_sweep"), width=3.8, optional=True, caption="α sweep (L-curve)"))
    if (res.get("multiexp") or {}).get("comparison") and len(res["multiexp"]["comparison"]) > 1:
        B.append(image(_figure(figures.model_comparison_figure, style, res, name="model_comparison"), width=3.4, optional=True, caption="Model comparison"))
    if ft:
        B.append(image(_figure(figures.ft_profile_map_figure, style, res, name="ft_profile_map"), width=CONTENT_W, caption="Spatial profile and echo map"))
        B.append(image(_figure(figures.spatial_figure, style, res, name="spatial"), width=CONTENT_W, caption="Position-resolved fits"))
        sp = res.get("spatial") or {}
        if sp.get("multiexp") or sp.get("ilt"):
            B.append(image(_figure(figures.spatial_extra_figure, style, res, name="spatial_extra"), width=CONTENT_W, optional=True, caption="Fit quality and ILT peaks along z"))
    if mcmc and mcmc.get("corner_png_b64"):
        B.append(image(_png_b64(mcmc["corner_png_b64"]), width=4.4, optional=True, caption="MCMC posterior (corner plot)"))
    return B


def _compose_t2d(res, prov, mcmc, boot, raw, style, kind):
    t = res["t2d"]
    B = _header(res, prov, kind)
    B += [h2("Settings"), table(None, [[a, b] for a, b in _settings_rows(res, prov)], [1.5, 5.7])]
    bi, bc = t["b_info"], t["b_check"]
    shape = next((c for c in bc["compared"] if c["kind"] == "shape" and c["value"] is not None), None)
    B += [para(f"b-values calculated as (γG)²·τ²·(Δ + 2τ/3) with G = {F.fmt(bi['gradient_t_per_m'], 4)} T/m, Δ = {F.fmt(bi['delta_ms'])} ms, τ = {F.fmt(bi['tau_min_ms'])}–{F.fmt(bi['tau_max_ms'])} ms ({bi['n_tau']}, log-spaced)."
               + (f" The spectrometer's own b-values have the same τ and Δ dependence and correspond to G = {F.fmt(shape['implied_gradient_t_per_m'], 4)} T/m." if shape else ""), size="note", color=MUTED),
          para(f"Fit: RMSE / signal range {F.pct(t['rmse_over_amp'], 2)}; noise σ {F.fmt(t['noise_sigma'])}; χ² per point {F.fmt(t['chi2_per_point'])}. Diffusion range the b-values can resolve: "
               f"{F.fmt_sci(t['d_sensitive'][0])}–{F.fmt_sci(t['d_sensitive'][1])} m²/s.", size="note", color=MUTED)]
    m = t.get("model")
    if m:
        rows = []
        for i, p in enumerate(m["populations"], 1):
            notes = [x for x in (f"D at the {p['D_at_limit']} limit: a bound only" if p["D_at_limit"] else "", f"T₂ at the {p['T2_at_limit']} limit" if p["T2_at_limit"] else "",
                                 "not supported by the data" if not p["supported"] and not p["D_at_limit"] and not p["T2_at_limit"] else "") if x]
            f2 = lambda v, ci, bound: F.fmt_sci(v) + " (bound)" if bound else (F.interval(v, ci[0], ci[1], None, sci=True) if ci[0] is not None else F.fmt_sci(v))
            rows.append([i, F.pct(p["fraction"]) + (f" ± {F.pct(p['fraction_se'])}" if F.ok_unc(p["fraction"], p["fraction_se"]) else ""), f2(p["T2"], p["T2_ci"], p["T2_at_limit"]), f2(p["D"], p["D_ci"], p["D_at_limit"]), "; ".join(notes) or "–"])
        B += [h2("Few-population model"), para(f"{m['K']} population{'s' if m['K'] > 1 else ''} (BIC prefers {m['recommended_k']}); S(t, b) = Σ Aₖ exp(−t/T₂ₖ) exp(−b Dₖ); χ² per point {F.fmt(m['chi2_per_point'])}; 95% least-squares intervals.", size="note", color=MUTED),
              table(["#", "fraction", "T₂ (s)", "D (m²/s)", "note"], rows, [0.35, 1.3, 2.05, 2.05, 1.45])]
    B += [h2("Regularised map: regions"), para(f"α = {F.fmt_sci(t['alpha'])}. Connected regions above {F.fmt(float((res.get('settings') or {}).get('t2d', {}).get('peak_threshold_pct', 10)))}% of the maximum; "
                                              "the map is a regularised view, not an estimate with intervals.", size="note", color=MUTED),
          table(["#", "fraction", "T₂ (s), mean", "D (m²/s), mean", "constrained by the data"], [[i + 1, F.pct(p["fraction"]), F.fmt_sci(p["T2_mean"]), F.fmt_sci(p["D_mean"]), "yes" if p["constrained"] else "no"] for i, p in enumerate(t["peaks"])] or [["–", "none", "", "", ""]],
                [0.4, 1.2, 1.8, 1.9, 1.9])]
    if mcmc or boot:
        B += [h2("Uncertainty: methods compared"), table(["parameter", "least squares ± SE", "bootstrap", "MCMC"], _compare_rows(mcmc, boot), [1.2, 2.0, 2.0, 2.0], size="note")]
    ws = collect_warnings(res, mcmc, boot)
    B += [h2("Warnings and notes")] + ([bullets(ws, color=WARN)] if ws else [para("None.", color=MUTED)])
    B.append(h2("Figures"))
    B.append(image(_figure(figures.t2d_spectrum_figure, style, res, name="t2d_spectrum"), width=5.2, caption="T₂–D map with projections"))
    if m:
        B.append(image(_figure(figures.t2d_populations_figure, style, res, mcmc, boot, name="t2d_populations"), width=5.4, caption="Populations on the map" + (" with MCMC / bootstrap samples" if mcmc or boot else "")))
        B.append(image(_figure(figures.model_comparison_figure, style, res, name="model_comparison"), width=3.4, optional=True, caption="Model comparison"))
    B.append(image(_figure(figures.t2d_fit_figure, style, res, name="t2d_fit"), width=CONTENT_W, optional=True, caption="Data, fit and residual"))
    if raw:
        B.append(image(_figure(lambda r, titles: figures.raw_figure(r, "t2d", titles), style, raw, name="raw_data"), width=4.4, optional=True, caption="Raw data as measured"))
    return B


# which registry entries apply to which analysis (ids from physics.MODELS and inference.METHODS)
_APPLIES = {"cpmg": ("exp_decay", "multi_exp", "kernels", "phasing", "noise", "fit_quality", "criteria", "interval", "regularisation"),
            "ft": ("exp_decay", "multi_exp", "kernels", "position", "phasing", "noise", "fit_quality", "criteria", "interval", "regularisation"),
            "t1": ("saturation", "fit_quality", "criteria", "interval"),
            "t2d": ("exp_decay", "diffusion", "b_value", "t2d", "kernels", "resolvable", "phasing", "noise", "fit_quality", "criteria", "interval", "regularisation")}


def _method_blocks(res, prov):
    mode = res.get("mode")
    kind = "t1" if mode == "t1" else "t2d" if mode == "t2d" else "ft" if mode == "ft" else "cpmg"
    entries = {m["id"]: m for m in physics.MODELS + inference.METHODS}
    rows = [[entries[i]["name"], entries[i]["equation"]] for i in _APPLIES[kind] if i in entries]
    B = [h2("How these numbers were obtained"),
         para("The relations the analysis assumes (the code for each is listed on the tool's physics page and lives in nmrdash/physics.py and nmrdash/inference.py):", size="note", color=MUTED),
         table(["relation", "equation"], rows, [2.4, 4.8], size="note"),
         para("Not modelled: restricted diffusion or a distribution of D within a population; non-exponential relaxation and exchange between populations; finite pulse lengths; "
              "diffusion during the CPMG echo train in the static gradient (the fitted T₂ is the apparent T₂).", size="note", color=MUTED)]
    return B


def _provenance_blocks(prov):
    from . import provenance
    inputs = [[f["name"], f"{f['bytes']:,} bytes", f["sha256"][:16] + "…", f.get("modified_utc", "")] for f in prov["inputs"]]
    return [h2("Provenance"), bullets([line for line in provenance.describe(prov) if not line.startswith("input ")] , size="note", color=MUTED),
            table(["input file", "size", "SHA-256 (start)", "modified"], inputs, [1.6, 1.3, 2.0, 2.3], size="note"),
            para("Provided as is, without warranty. Fits, inversions and uncertainty estimates depend on the model, the settings and the noise; check them independently before relying on a number. "
                 "The full record (every setting and the full hashes) is in provenance.json; the settings used are in params_used.toml.", size="note", color=MUTED)]


def compose(res, prov, mcmc=None, boot=None, raw=None, style=None):
    """The report as a list of blocks (no drawing)."""
    kind = res.get("mode")
    B = (_compose_t2d if kind == "t2d" else _compose_decay)(res, prov, mcmc, boot, raw, style, kind)
    return B + _method_blocks(res, prov) + _provenance_blocks(prov)


def build(res, prov, mcmc=None, boot=None, raw=None, style=None):
    """The PDF bytes for one analysis. res, prov, mcmc, boot are the JSON-like dicts the pipeline returns; raw is the Raw data tab's content."""
    blocks = compose(res, prov, mcmc, boot, raw, style)
    pages, scale = fit(blocks)
    nm = prov.get("nmrdash", {})
    return render(pages, scale, f"nmrdash {nm.get('version', '?')} ({(nm.get('commit') or 'unknown')[:10]})  ·  {prov.get('dataset')}", f"nmrdash report: {prov.get('dataset')}")


# ── a batch of datasets ──────────────────────────────────────────────────────

def _row_result(r):
    """One line for a dataset in the batch table: its main numbers."""
    res = r.get("result") or {}
    if r["status"] != "ok":
        return r.get("reason") or r["status"]
    me, t = res.get("multiexp"), res.get("t2d")
    if t and t.get("model"):
        m = t["model"]
        return f"{m['K']} population(s): " + "; ".join(f"T₂ {F.fmt_sci(p['T2'])} s, D {F.fmt_sci(p['D'])} m²/s, {F.pct(p['fraction'], 0)}" for p in m["populations"][:3]) + ("; …" if m["K"] > 3 else "")
    if t:
        return "regions: " + "; ".join(f"T₂ {F.fmt_sci(p['T2_mean'])} s, D {F.fmt_sci(p['D_mean'])} m²/s" for p in t["peaks"][:3])
    if me:
        sym = "T₁" if r.get("kind") == "t1" else "T₂"
        return f"{me['n']} component(s): " + "; ".join(f"{sym} {F.fmt_sci(c['T2'])} s ({F.pct(c['fraction'], 0)})" for c in me["components"][:3])
    return "–"


def build_batch(rows, params, root_name, t0=None, params_text=None):
    """A report for a whole batch: what was run, one line per dataset, the time series, and a note on how to read them."""
    ok = [r for r in rows if r["status"] == "ok"]
    counts = {k: sum(r["status"] == k for r in rows) for k in ("ok", "skipped", "failed")}
    times = [r["timestamp_utc"] for r in rows if r.get("timestamp_utc")]
    B = [title("nmrdash batch report", root_name),
         table(None, [["Datasets", f"{len(rows)} ({counts['ok']} processed, {counts['skipped']} skipped, {counts['failed']} failed)", "Measured", (min(times) + " to " + max(times)) if times else "unknown"],
                      ["Time zero", t0.isoformat() if t0 else "none", "Tool", f"nmrdash {__import__('nmrdash').__version__}"]], [0.9, 3.0, 0.9, 2.4], size="note"),
         h2("Datasets")]
    trows = [[r["dataset"], (r.get("timestamp_utc") or "–")[:19].replace("T", " "), "–" if r.get("elapsed_min") is None else f"{r['elapsed_min']:.1f}", r["status"], _row_result(r)] for r in rows]
    B.append(table(["dataset", "measured (UTC)", "min", "status", "main result"], trows, [1.5, 1.3, 0.5, 0.7, 3.2], size="note"))
    figs = []
    for sym, kind, name in (("T2", "cpmg", "decay"), ("T1", "t1", "recovery")):
        sel = [r for r in ok if r.get("kind") == kind and (r.get("result") or {}).get("multiexp")]
        if len(sel) >= 2:
            figs.append(image(_png_array(figures.timeseries_figure(sel, False, sym)), width=4.8, caption=f"{'T₂' if sym == 'T2' else 'T₁'} and amplitude against elapsed time ({name})"))
    sel = [r for r in ok if r.get("kind") == "t2d" and ((r.get("result") or {}).get("t2d") or {}).get("model")]
    if len(sel) >= 2:
        figs.append(image(_png_array(figures.timeseries_t2d_figure(sel, False)), width=4.4, caption="T₂–D populations against elapsed time (populations are matched by their T₂ order)"))
    if figs:
        B += [h2("Time series")] + figs
    warn = [f"{r['dataset']}: {r['status']}" + (f" ({r['reason']})" if r.get("reason") else "") for r in rows if r["status"] != "ok"]
    for r in ok:
        warn += [f"{r['dataset']}: {w}" for w in (r.get("warnings") or [])[:2]]
    B += [h2("Notes")] + ([bullets(warn[:40], color=WARN, size="note")] if warn else [para("No warnings.", color=MUTED)])
    B.append(para("Each dataset has its own report (report.pdf) in its processed_… folder, with the settings used, the full numbers, the figures and the provenance. Provided as is, without warranty; "
                  "check results independently before relying on them.", size="note", color=MUTED))
    pages, scale = fit(B, max_pages=60)
    return render(pages, scale, f"nmrdash batch report  ·  {root_name}", f"nmrdash batch report: {root_name}")
