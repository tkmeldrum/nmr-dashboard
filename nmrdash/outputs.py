"""Write results to disk: CSV tables, figures and a human-readable summary.md.

Everything here takes the JSON-like dicts that pipeline.process / mcmc_finish / boot_finish return.
"""
import base64
import csv
import math
from pathlib import Path

import numpy as np

from . import figures, formatting as F, provenance


def _num(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return repr(float(v)) if isinstance(v, float) else v


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_num(v) for v in r])


def _col(a):
    return [None if v is None else v for v in a]


# ── tables ───────────────────────────────────────────────────────────────────

def _t2d_tables(res, out):
    r = res["t2d"]
    T2, D = r["T2"], r["D"]
    out("t2d_spectrum.csv", ["D_m2s \\ T2_s"] + [repr(x) for x in T2], [[d] + list(row) for d, row in zip(D, r["spectrum"])])
    out("t2d_projection_T2.csv", ["T2_s", "amplitude"], zip(T2, r["projection_T2"]))
    out("t2d_projection_D.csv", ["D_m2s", "amplitude"], zip(D, r["projection_D"]))
    out("t2d_peaks.csv", ["fraction", "T2_mean_s", "D_mean_m2s", "T2_peak_s", "D_peak_m2s", "height", "constrained_by_data"],
        [[p["fraction"], p["T2_mean"], p["D_mean"], p["T2_peak"], p["D_peak"], p["height"], int(p["constrained"])] for p in r["peaks"]])
    out("t2d_statistics.csv", ["quantity", "value"], [("alpha", r["alpha"]), ("T2_points", r["n_T2"]), ("D_points", r["n_D"]), ("T2_min_s", r["t2_min"]), ("T2_max_s", r["t2_max"]),
        ("D_min_m2s", r["d_min"]), ("D_max_m2s", r["d_max"]), ("excluded_first_echoes", res["decay"].get("skip", 0)), ("D_sensitive_min_m2s", r["d_sensitive"][0]), ("D_sensitive_max_m2s", r["d_sensitive"][1]), ("fraction_of_spectrum_in_sensitive_window", r["fraction_in_sensitive_window"]),
        ("rss", r["rss"]), ("mse", r["mse"]), ("rmse", r["rmse"]),
        ("signal_range", r["signal_range"]), ("rmse_over_signal_range", r["rmse_over_amp"]), ("noise_sigma", r["noise_sigma"]), ("chi2_per_point", r["chi2_per_point"])])
    pred, resid = np.asarray(r["pred"]), np.asarray(r["resid"])                         # (n_b, n_echo)
    rows = [[e, b, pred[i, j] + resid[i, j], pred[i, j], resid[i, j]] for i, b in enumerate(r["b_vec"]) for j, e in enumerate(r["echo_s"])]
    out("t2d_data_fit_residual.csv", ["echo_time_s", "b_value_1e-9_s_per_m2", "data", "fit", "residual"], rows)


def write_tables(d, res, mcmc=None, boot=None, samples=True):
    """CSV files into directory d; returns the list of file names written."""
    d = Path(d)
    written = []

    def out(name, header, rows):
        write_csv(d / name, header, rows)
        written.append(name)

    if res.get("mode") == "t2d":
        _t2d_tables(res, out)
        return written

    t, y = res["decay"]["t"], res["decay"]["y"]
    me = (res.get("multiexp") or {}).get("selected")
    il = res.get("ilt")
    lab = "T1" if res.get("mode") == "t1" else "T2"                  # name of the time constant in the file headers
    skip = res["decay"].get("skip", 0)
    pad = [None] * skip                                            # excluded leading points have no fit or residual
    cols, head = [t, y, [int(i >= skip) for i in range(len(t))]], ["delay_s" if lab == "T1" else "echo_time_s", "signal", "included_in_fits"]
    if me:
        cols += [pad + list(me["fit"]), pad + list(me["resid"])]
        head += [f"exp{me['n']}_fit", f"exp{me['n']}_residual"]
    if il:
        cols += [pad + list(il["pred"]), pad + list(il["resid"])]
        head += ["ilt_fit", "ilt_residual"]
    out("decay.csv", head, zip(*cols))

    if me:
        lvl = round(100 * me["ci_level"])
        out("multiexp_components.csv",
            ["component", f"{lab}_s", f"{lab}_se", f"{lab}_ci{lvl}_lo", f"{lab}_ci{lvl}_hi", "amplitude", "amplitude_se", f"amplitude_ci{lvl}_lo", f"amplitude_ci{lvl}_hi", "fraction", "fraction_se"],
            [[i + 1, c["T2"], c["T2_se"], c["T2_ci"][0], c["T2_ci"][1], c["A"], c["A_se"], c["A_ci"][0], c["A_ci"][1], c["fraction"], c["fraction_se"]]
             for i, c in enumerate(me["components"])])
        q = [("n_components", me["n"]), ("offset_fitted", int(me["offset"])), ("offset", me["offset_value"]), ("offset_se", me["offset_se"]),
             ("total_amplitude", me["total_amplitude"]), ("total_amplitude_se", me["total_amplitude_se"]), (f"mean_{lab}_s", me["mean_T2"]), (f"mean_{lab}_se", me["mean_T2_se"]),
             ("rss", me["rss"]), ("mse", me["mse"]), ("rmse", me["rmse"]), ("residual_sd", me["resid_sd"]), ("signal_range", me["signal_range"]),
             ("residual_sd_over_signal_range", me["resid_sd_over_amp"]), ("noise_sigma", me["noise_sigma"]), ("reduced_chi2", me["chi2_red"]),
             ("r2", me["r2"]), ("aic", me["aic"]), ("bic", me["bic"]), ("degrees_of_freedom", me["dof"]), ("ci_level", me["ci_level"])]
        out("multiexp_fit_quality.csv", ["quantity", "value"], q)
        cmp_ = res["multiexp"]["comparison"]
        out("model_comparison.csv", ["components", "parameters", "rss", "mse", "residual_sd_over_signal_range", "reduced_chi2", "delta_aic", "delta_bic"],
            [[c["n"], c["n_params"], c["rss"], c["mse"], c["resid_sd_over_amp"], c["chi2_red"], c["d_aic"], c["d_bic"]] for c in cmp_])

    if il:
        out("ilt_spectrum.csv", ["T2_s", "amplitude"], zip(il["tau"], il["spectrum"]))
        out("ilt_peaks.csv", ["peak_T2_s", "mean_T2_s", "signal_fraction"], [[p["T2_peak"], p["T2_mean"], p["fraction"]] for p in il["peaks"]])
        out("ilt_statistics.csv", ["quantity", "value"], [("alpha", il["alpha"]), ("grid_points", il["n_steps"]), ("T2_min_s", il["t2_min"]), ("T2_max_s", il["t2_max"]),
            ("rss", il["rss"]), ("mse", il["mse"]), ("rmse", il["rmse"]), ("signal_range", il["signal_range"]), ("rmse_over_signal_range", il["rmse_over_amp"]),
            ("noise_sigma", il["noise_sigma"]), ("chi2_per_point", il["chi2_per_point"])])
        if il.get("sweep"):
            out("ilt_alpha_sweep.csv", ["alpha", "rmse", "roughness"], [[s["alpha"], s["rmse"], s["roughness"]] for s in il["sweep"]])

    ft, sp = res.get("ft"), res.get("spatial") or {}
    if ft:
        lo, hi = ft["window"]
        out("ft_profile.csv", ["z_um", "profile", "in_window"], [[z, p, int(lo <= z <= hi)] for z, p in zip(ft["z_um"], ft["profile"])])
        m = sp.get("multiexp")
        if m:
            n = m["n"]
            head = ["z_um", "fitted"] + [f"T2_{i + 1}_s" for i in range(n)] + [f"T2_{i + 1}_se" for i in range(n)] + [f"A_{i + 1}" for i in range(n)] + \
                   [f"A_{i + 1}_se" for i in range(n)] + [f"fraction_{i + 1}" for i in range(n)] + ["mean_T2_s", "mean_T2_se", "total_amplitude", "rmse", "reduced_chi2", "flagged"]
            rows = []
            for i, z in enumerate(m["z_um"]):
                rows.append([z, int(m["ok"][i])] + [m["T2"][k][i] for k in range(n)] + [m["T2_se"][k][i] for k in range(n)] + [m["A"][k][i] for k in range(n)] +
                            [m["A_se"][k][i] for k in range(n)] + [m["fraction"][k][i] for k in range(n)] +
                            [m["mean_T2"][i], m["mean_T2_se"][i], m["total_amplitude"][i], m["rmse"][i], m["chi2_red"][i], int(m["flag"][i])])
            out("spatial_multiexp.csv", head, rows)
        m = sp.get("ilt")
        if m:
            out("spatial_ilt_map.csv", ["z_um"] + [repr(t_) for t_ in m["tau"]], [[z] + list(row) for z, row in zip(m["z_um"], m["spectra"])])
            nb = len(m["bands"])
            out("spatial_ilt_peaks.csv", ["z_um", "mean_T2_s", "rmse"] + [f"peak{b['t2_peak']:.2e}_meanT2_s" for b in m["bands"]] + [f"peak{b['t2_peak']:.2e}_fraction" for b in m["bands"]],
                [[z, m["mean_T2"][i], m["rmse"][i]] + [m["band_mean_T2"][k][i] for k in range(nb)] + [m["band_fraction"][k][i] for k in range(nb)] for i, z in enumerate(m["z_um"])])

    if mcmc:
        out("mcmc_summary.csv", ["parameter", "least_squares", "least_squares_se", "posterior_median", "interval_lo", "interval_hi", "posterior_sd", "posterior_mean"],
            [[p["name"], p.get("ls"), p.get("ls_se"), p["median"], p["lo"], p["hi"], p["sd"], p["mean"]] for p in mcmc["params"] + mcmc["derived"]])
        dg = mcmc["diagnostics"]
        out("mcmc_diagnostics.csv", ["quantity", "value"], [("walkers", mcmc["nwalkers"]), ("steps_run", mcmc["steps_run"]), ("steps_planned", mcmc["steps_planned"]), ("burn_in", mcmc["burn"]),
            ("interval_level", mcmc["level"]), ("acceptance_fraction", dg["acceptance"]), ("autocorrelation_time_max", dg["tau_max"]), ("effective_samples", dg["n_eff"]),
            ("steps_per_autocorrelation_time", dg["steps_per_tau"]), ("noise_sigma_sampled", int(mcmc["sample_sigma"]))])
        if samples and mcmc.get("samples"):
            out("mcmc_samples.csv", mcmc["samples"]["names"], mcmc["samples"]["values"])

    if boot:
        bm = boot.get("multiexp")
        if bm:
            out("bootstrap_summary.csv", ["parameter", "least_squares", "bootstrap_median", "interval_lo", "interval_hi", "bootstrap_sd", "bootstrap_mean"],
                [[p["name"], p.get("ls"), p["median"], p["lo"], p["hi"], p["sd"], p["mean"]] for p in bm["params"] + bm["derived"]])
            if samples:
                out("bootstrap_samples.csv", bm["samples"]["names"], bm["samples"]["values"])
        bi = boot.get("ilt")
        if bi:
            out("bootstrap_ilt_band.csv", ["T2_s", "ilt_of_data", "bootstrap_lo", "bootstrap_median", "bootstrap_hi"], zip(bi["tau"], bi["spectrum"], bi["lo"], bi["median"], bi["hi"]))
            out("bootstrap_ilt_peaks.csv", ["peak_T2_s", "T2_of_data_s", "T2_lo", "T2_median", "T2_hi", "fraction_of_data", "fraction_lo", "fraction_median", "fraction_hi"],
                [[b["t2_peak"], b["base_mean_T2"], b["T2"]["lo"], b["T2"]["median"], b["T2"]["hi"], b["base_fraction"], b["fraction"]["lo"], b["fraction"]["median"], b["fraction"]["hi"]] for b in bi["bands"]])
    return written


# ── figures ──────────────────────────────────────────────────────────────────

def write_figures(d, res, mcmc=None, boot=None, fmt="png", dpi=150, titles=True, band=None):
    """band: the MCMC result to draw as a band on the decay figure (only when MCMC analysed that same decay)."""
    d = Path(d)
    written = []

    def save(name, fig):
        figures.save_figure(fig, d / f"{name}.{fmt}", fmt, dpi)
        written.append(f"{name}.{fmt}")

    if res.get("mode") == "t2d":
        save("t2d_spectrum", figures.t2d_spectrum_figure(res, titles))
        save("t2d_fit", figures.t2d_fit_figure(res, titles))
        return written
    save("decay_fit", figures.decay_figure(res, band, titles))
    if res.get("ilt"):
        save("ilt_spectrum", figures.ilt_figure(res, titles))
    if res.get("ft"):
        save("spatial", figures.spatial_figure(res, titles))
    if mcmc:
        (d / "mcmc_corner.png").write_bytes(base64.b64decode(mcmc["corner_png_b64"]))
        written.append("mcmc_corner.png")
    if boot and boot.get("ilt"):
        save("bootstrap_ilt", figures.bootstrap_ilt_figure(boot, titles))
    if boot and boot.get("multiexp"):
        save("bootstrap_distributions", figures.bootstrap_hist_figure(boot, titles))
    return written


# ── summary.md ───────────────────────────────────────────────────────────────

def _md_table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _t2d_summary(res, prov, files, warnings):
    r, acq = res["t2d"], prov.get("acquisition", {})
    when = next((f["modified_utc"] for f in prov["inputs"] if f["name"] in ("data2D_Cp.2d", "data2D.dat")), None)
    skip = res["decay"].get("skip", 0)
    L = [f"# nmrdash results: {prov['dataset']}", "",
         f"- Sequence: {acq.get('experiment', '?')} (T₂–D correlation), {acq.get('nrEchoes', '?')} echoes × {acq.get('nrPntsTau', '?')} diffusion delays, echo time {acq.get('echoTime', '?')} µs, Δ = {acq.get('DELTA', '?')} ms",
         f"- 2D data written: {when or 'unknown'}"]
    if skip:
        L.append(f"- The first {skip} echoes are excluded from the inversion.")
    L += [f"- Grid: {r['n_T2']} T₂ points {F.fmt_sci(r['t2_min'])} – {F.fmt_sci(r['t2_max'])} s × {r['n_D']} D points {F.fmt_sci(r['d_min'])} – {F.fmt_sci(r['d_max'])} m²/s; α = {F.fmt_sci(r['alpha'])}",
          f"- Fit: RMSE / signal range {F.pct(r['rmse_over_amp'], 2)}; noise σ (imaginary channel) {F.fmt(r['noise_sigma'])}; χ² per point {F.fmt(r['chi2_per_point'])}", ""]
    for w in warnings:
        L.append(f"> **Warning:** {w}")
    lo, hi = r["d_sensitive"]
    if r["fraction_in_sensitive_window"] is not None and r["fraction_in_sensitive_window"] < 0.9:
        L += ["", f"> **Only {F.pct(r['fraction_in_sensitive_window'], 0)} of the spectrum lies where the data can constrain it.** The b-values "
                  f"({F.fmt_sci(float(min(r['b_vec'])) * 1e9)} – {F.fmt_sci(float(max(r['b_vec'])) * 1e9)} s/m²) resolve diffusion coefficients from about {F.fmt_sci(lo)} to {F.fmt_sci(hi)} m²/s. "
                  "Signal the regularised solution places outside that window is extrapolation, not measurement; the regions flagged *no* below are in it."]
    edge = [p for p in r["peaks"] if p["D_peak"] <= r["D"][0] * 1.0001 or p["D_peak"] >= r["D"][-1] * 0.9999 or p["T2_peak"] <= r["T2"][0] * 1.0001 or p["T2_peak"] >= r["T2"][-1] * 0.9999]
    if edge:
        L += ["", f"> {len(edge)} region(s) peak at the edge of the grid; signal may extend beyond the range you chose, so widen it to check."]
    L += ["", "## T₂–D regions", "", "Connected regions of the spectrum above "
          f"{F.fmt(float(res['settings'].get('t2d', {}).get('peak_threshold_pct', 10)))}% of its maximum (overlapping peaks can appear as one region).", ""]
    L += _md_table(["#", "fraction", "T₂ (s), mean", "D (m²/s), mean", "T₂ at max (s)", "D at max (m²/s)", "constrained by the data"],
                   [[i + 1, F.pct(p["fraction"]), F.fmt_sci(p["T2_mean"]), F.fmt_sci(p["D_mean"]), F.fmt_sci(p["T2_peak"]), F.fmt_sci(p["D_peak"]), "yes" if p["constrained"] else "**no**"]
                    for i, p in enumerate(r["peaks"])] or [["–", "no regions", "", "", "", "", ""]]) + [""]
    L += ["## Provenance", "", *[f"- {line}" for line in provenance.describe(prov)], "- full detail: `provenance.json`; the settings used: `params_used.toml`", ""]
    if files:
        L += ["## Files", "", *[f"- `{f}`" for f in sorted(files)], ""]
    return "\n".join(L)


def summary_markdown(res, prov, mcmc=None, boot=None, files=(), warnings=()):
    if res.get("mode") == "t2d":
        return _t2d_summary(res, prov, files, warnings)
    L = [f"# nmrdash results: {prov['dataset']}", ""]
    acq = prov.get("acquisition", {})
    t1 = res.get("mode") == "t1"
    sym = "T1" if t1 else "T2"
    when = next((f["modified_utc"] for f in prov["inputs"] if f["name"] in ("data.2d", "data.dat")), None)
    if t1:
        L += [f"- Sequence: {acq.get('experiment', '?')} (saturation recovery), {len(res['decay']['t'])} delays", f"- data.dat written: {when or 'unknown'}",
              "- No imaginary channel in these data, so there is no measured noise σ (reduced χ² is not available; MCMC samples σ, the bootstrap resamples residuals).", ""]
    else:
        L += [f"- Sequence: {acq.get('experiment', '?')}, echo time {acq.get('echoTime', '?')} µs, {acq.get('nrEchoes', '?')} echoes, {acq.get('nrPnts', '?')} points per echo",
              f"- data.2d written: {when or 'unknown'}", f"- Echo processing: {'FT of each echo (signal vs z)' if res.get('mode') == 'ft' else 'integrate each echo'}; "
              f"autophase rotation {F.fmt(res['decay']['phase_deg'], 4)}°; noise σ from the imaginary channel: {F.fmt(res['decay'].get('noise_sigma'))}", ""]
    skip = res["decay"].get("skip", 0)
    if skip:
        L += [f"- The first {skip} {'delays' if t1 else 'echoes'} are excluded from every fit (they are still shown, as open markers).", ""]
    for w in warnings:
        L.append(f"> **Warning:** {w}")
    if warnings:
        L.append("")
    me = (res.get("multiexp") or {}).get("selected")
    if me:
        lvl = round(100 * me["ci_level"])
        shape = (lambda a, tt: f"{a} (1 − exp(−t/{tt}))") if t1 else (lambda a, tt: f"{a} exp(−t/{tt})")
        L += [("## Saturation-recovery fit" if t1 else "## Multi-exponential fit") + (" (window-summed decay)" if res.get("mode") == "ft" else ""), "",
              "S(t) = " + " + ".join(shape(f"A{'' if me['n'] == 1 else i}", f"{sym}{'' if me['n'] == 1 else ',' + str(i)}") for i in range(1, me["n"] + 1)) + (" + c" if me["offset"] else ""), ""]
        rows = []
        for i, c in enumerate(me["components"], 1):
            rows.append([i, F.pair(c["T2"], c["T2_se"], sci=True), F.interval(c["T2"], *c["T2_ci"], se=c["T2_se"], sci=True) if F.ok_unc(c["T2"], c["T2_se"]) else "undetermined",
                         F.fmt(c["A"]), F.pct(c["fraction"]) + (f" ± {F.pct(c['fraction_se'])}" if F.ok_unc(c["fraction"], c["fraction_se"]) else "")])
        L += _md_table(["#", f"{sym} (s) ± SE", f"{lvl}% interval", "amplitude", "fraction"], rows) + [""]
        stats = [("RSS", F.fmt_sci(me["rss"])), ("MSE", F.fmt_sci(me["mse"])), ("residual SD / signal range", F.pct(me["resid_sd_over_amp"], 2)),
                 ("noise σ (imaginary channel)", F.fmt(me["noise_sigma"])), ("reduced χ²", F.fmt(me["chi2_red"])), ("R²", F.fmt(me["r2"], 5)), ("degrees of freedom", me["dof"])]
        L += _md_table(["statistic", "value"], stats) + [""]
        for w in me["warnings"]:
            L.append(f"> {w}")
        if me["warnings"]:
            L.append("")
        cmp_ = res["multiexp"]["comparison"]
        if len(cmp_) > 1:
            L += ["", "Model comparison (Δ relative to the best; ΔBIC > 10 is strong evidence against a model):", ""]
            L += _md_table(["components", "reduced χ²", "ΔAIC", "ΔBIC"], [[c["n"], F.fmt(c["chi2_red"]), f"{c['d_aic']:.1f}", f"{c['d_bic']:.1f}"] for c in cmp_]) + [""]
    il = res.get("ilt")
    if il:
        L += ["## ILT", "", f"α = {F.fmt_sci(il['alpha'])}, {il['n_steps']} grid points, T2 {F.fmt_sci(il['t2_min'])} – {F.fmt_sci(il['t2_max'])} s; "
              f"RMSE / signal range {F.pct(il['rmse_over_amp'], 2)}; χ² per point {F.fmt(il['chi2_per_point'])}", ""]
        L += _md_table(["peak T2 (s)", "mean T2 of peak (s)", "fraction"], [[F.fmt_sci(p["T2_peak"]), F.fmt_sci(p["T2_mean"]), F.pct(p["fraction"])] for p in il["peaks"]] or [["none", "", ""]]) + [""]
    ft, sp = res.get("ft"), res.get("spatial") or {}
    if ft:
        L += ["## Spatial analysis (FT)", "", f"{ft['nucleus']}, G = {ft['G']} T/m, zero-fill ×2^{ft['zero_fill']} → {ft['nfft']} points (Δz = {F.fmt(ft['zf_step_um'])} µm); "
              f"natural resolution {F.fmt(ft['resolution_um'])} µm; window {F.fmt(ft['window'][0])} to {F.fmt(ft['window'][1])} µm; "
              f"+ z = + frequency = closer to the magnet. Positions below {F.fmt(ft['min_signal_pct'])}% of the profile maximum are not fitted. "
              f"Exponential fits: {ft['n_positions_kept']} of {ft['n_positions']} positions; ILT: {ft['n_ilt_positions_kept']} of {ft['n_ilt_positions']}.", ""]
    for name, r in (("MCMC", mcmc), ("Bootstrap", boot)):
        if not r:
            continue
        L += [f"## {name}", ""]
        if name == "MCMC":
            d = r["diagnostics"]
            L += [f"{r['nwalkers']} walkers × {r['steps_run']} steps (burn-in {r['burn']}); acceptance {F.fmt(d['acceptance'], 2)}; autocorrelation time {F.fmt(d['tau_max'])} steps; "
                  f"{F.fmt(d['n_eff'])} effective samples. σ {'sampled' if r['sample_sigma'] else 'fixed from the imaginary channel'}.", ""]
            rows = [[p["name"], F.interval(p["median"], p["lo"], p["hi"], p["sd"], sci=p["name"].startswith("T2") or "T2" in p["name"])] for p in r["params"] + r["derived"]]
        else:
            L += [f"{r['replicates']} replicates, noise: {r['method']}; {r['failures']} failed.", ""]
            rows = [[p["name"], F.interval(p["median"], p["lo"], p["hi"], p["sd"], sci="T2" in p["name"])] for p in ((r.get("multiexp") or {}).get("params", []) + (r.get("multiexp") or {}).get("derived", []))]
        L += _md_table(["parameter", f"median [{round(100 * r['level'])}% interval]"], rows) + [""]
        for w in r["warnings"]:
            L.append(f"> {w}")
        L.append("")
    L += ["## Provenance", "", *[f"- {line}" for line in provenance.describe(prov)], "- full detail: `provenance.json`; the settings used: `params_used.toml`", ""]
    if files:
        L += ["## Files", "", *[f"- `{f}`" for f in sorted(files)], ""]
    return "\n".join(L)
