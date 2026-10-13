// Results tab: spatial profile and per-position fits (FT mode), decay + fits, fit tables, ILT spectrum.
import { $, state } from "./state.js";
import { baseLayout, CFG, COLORS, fmt, fmtCI as fmtCIplain, fmtCIWith, fmtPair, fmtSci, isOdd, okUnc, pct, sciGroup } from "./plotutil.js";
import { ILT_EQUATION, meDefinitions, meEquation } from "./equations.js";

const LINKS = {
  RSS: "https://en.wikipedia.org/wiki/Residual_sum_of_squares",
  MSE: "https://en.wikipedia.org/wiki/Mean_squared_error",
  AIC: "https://en.wikipedia.org/wiki/Akaike_information_criterion",
  BIC: "https://en.wikipedia.org/wiki/Bayesian_information_criterion",
  CHI2: "https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic",
  NRMSD: "https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization",
};
const link = (text, href) => ({ text, href });
const d1 = (x) => (x === null || x === undefined ? "—" : x.toFixed(1));

// Rows for a (statistic, value, meaning) table; a statistic may be a link object.
function statTable(el, rows) {
  table(el, ["Statistic", "Value", "Meaning"], rows.map(([a, b, c]) => [a, b, c]));
  // table() writes plain text; turn the statistic cells into links where asked
  [...el.querySelectorAll("tr")].slice(1).forEach((tr, i) => {
    const stat = rows[i][3];
    if (stat) {
      const td = tr.firstChild;
      td.replaceChildren(Object.assign(document.createElement("a"), { textContent: rows[i][0], href: stat, target: "_blank", rel: "noopener noreferrer" }));
    }
  });
}

// headers: strings, or {text, href} for a link to an explanation
function table(el, headers, rows) {
  const mk = (tag, text) => Object.assign(document.createElement(tag), { textContent: text });
  const head = document.createElement("tr");
  headers.forEach((h) => {
    const th = mk("th", typeof h === "string" ? h : "");
    if (typeof h !== "string") {
      const a = Object.assign(document.createElement("a"), { textContent: h.text, href: h.href, target: "_blank", rel: "noopener noreferrer", title: "Open an explanation (new tab)" });
      th.appendChild(a);
    }
    head.appendChild(th);
  });
  el.replaceChildren(head, ...rows.map((r) => {
    const tr = document.createElement("tr");
    r.forEach((c) => tr.appendChild(mk("td", c)));
    return tr;
  }));
}

// error-bar arrays: missing or absurd (undetermined) uncertainties are not drawn
const bars = (se, v) => se.map((x, i) => (okUnc(v[i], x) ? x : 0));
const T2_AXIS = (linear) => (linear ? { type: "linear", tickformat: ".1e" } : { type: "log", tickformat: ".0e" });

// ── FT: profile and echo map ──────────────────────────────────────────────
function renderFt(r) {
  $("ft-results").hidden = !r.ft;
  if (!r.ft) return;
  const f = r.ft, [w0, w1] = f.window;
  const nuc = f.nucleus === "19F" ? "¹⁹F" : "¹H";
  $("ft-stats").textContent = `${nuc} (γ/2π = ${f.gamma_mhz_per_t} MHz/T), G = ${f.G} T/m, zero-fill ×2^${f.zero_fill} → ${f.nfft} points, ` +
    `echo centre at point ${f.center_point}. Window ${fmt(w0)} to ${fmt(w1)} µm${f.window_auto ? " (automatic: profile > 10% of max)" : ""}.`;
  const base = baseLayout("Spatial profile", "z (µm)", "|Σ echoes| (arb.)");
  Plotly.react("plot-profile", [{ x: f.z_um, y: f.profile, mode: "lines", name: "Profile", line: { color: COLORS.blue } }], {
    ...base, showlegend: false,
    shapes: [{ type: "rect", xref: "x", yref: "paper", x0: w0, x1: w1, y0: 0, y1: 1, fillcolor: COLORS.orange, opacity: 0.18, line: { width: 0 } }],
  }, CFG);
  const t = r.decay.t, h = f.heat;
  const base2 = baseLayout("Echo magnitude after FT", "Echo time (s)", "z (µm)");
  Plotly.react("plot-ftmap", [{ type: "heatmap", z: h.mag, x: h.echo_idx.map((i) => t[i]), y: h.z_um, colorscale: "Viridis", colorbar: { title: "|S|" }, zsmooth: false }],
    { ...base2, yaxis: { ...base2.yaxis, autorange: "reversed" } }, CFG);      // + z (+ frequency) points down, toward the magnet
  $("ftmap-note").textContent =
    `Drawn on the full zero-filled grid (${f.nfft} points, Δz = ${fmt(f.zf_step_um)} µm). The natural resolution is ${fmt(f.resolution_um)} µm: ` +
    `zero-filling interpolates and adds no resolution. + z is down (toward the magnet).`;
}

// ── FT: fits at every z position ──────────────────────────────────────────
function renderSpatial(r) {
  const sp = r.spatial;
  $("spatial-results").hidden = !sp;
  if (!sp) return;
  const f = r.ft, linear = $("sp-linear").checked;
  const il0 = sp.ilt;
  $("spatial-stats").textContent =
    (sp.multiexp ? `Exponential fits: ${f.n_positions_kept} of ${f.n_positions} positions (every ${fmt(f.spacing_um)} µm${f.spacing_auto ? ", the natural resolution" : ""}). ` : "") +
    (il0 ? `ILT: ${f.n_ilt_positions_kept} of ${f.n_ilt_positions} positions on the zero-filled grid (every ${fmt(f.zf_step_um)} µm) so the map matches the other heat maps. ` : "") +
    `Positions with signal below ${fmt(f.min_signal_pct)}% of the profile maximum are not fitted (gaps). ` +
    `Positions closer than the natural resolution of ${fmt(f.resolution_um)} µm are not independent measurements.`;
  const t2 = baseLayout("", "z (µm)", "T₂ (s)").yaxis;

  const me = sp.multiexp;
  $("card-sp-t2").hidden = $("card-sp-amp").hidden = !me;
  if (me) {
    const z = me.z_um;
    const sym = me.flag.map((fl, i) => (fl || !me.ok[i] ? "circle-open" : "circle"));
    const palette = [COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple];
    const label = (k) => (me.n === 1 ? "T₂" : `T₂,${k + 1}`);
    const traces = me.T2.map((row, k) => ({
      x: z, y: row, name: label(k), mode: "markers", marker: { color: palette[k], size: 7, symbol: sym },
      error_y: { type: "data", array: bars(me.T2_se[k], me.T2[k]), visible: true, thickness: 1, color: palette[k] },
    }));
    if (me.n > 1) traces.push({ x: z, y: me.mean_T2, name: "mean T₂", mode: "lines", line: { color: COLORS.grey, dash: "dot", width: 2 } });
    Plotly.react("plot-sp-t2", traces, baseLayout("T₂ vs position (multi-exponential)", "z (µm)", "T₂ (s)",
      { yaxis: { ...t2, ...T2_AXIS(linear) } }), CFG);

    const amps = me.A.map((row, k) => ({
      x: z, y: row, name: me.n === 1 ? "A" : `A${k + 1}`, mode: "markers", marker: { color: palette[k], size: 7, symbol: sym },
      error_y: { type: "data", array: bars(me.A_se[k], me.A[k]), visible: true, thickness: 1, color: palette[k] },
    }));
    if (me.n > 1) amps.push({ x: z, y: me.total_amplitude, name: "total", mode: "lines", line: { color: COLORS.grey, width: 2 } });
    Plotly.react("plot-sp-amp", amps, baseLayout("Amplitude vs position", "z (µm)", "Amplitude (arb.)"), CFG);
  }

  $("card-sp-chi2").hidden = !me;
  if (me) {
    const base3 = baseLayout("Fit quality vs position", "z (µm)", "Reduced χ²");
    Plotly.react("plot-sp-chi2", [{ x: me.z_um, y: me.chi2_red, mode: "lines+markers", name: "reduced χ²", marker: { size: 6, color: COLORS.blue }, line: { color: COLORS.blue }, connectgaps: false }],
      { ...base3, showlegend: false, yaxis: { ...base3.yaxis, rangemode: "tozero" },
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: COLORS.grey, dash: "dot" } }] }, CFG);
  }

  const il = sp.ilt;
  for (const id of ["card-sp-ilt", "card-sp-iltsum", "card-sp-iltfrac"]) $(id).hidden = !il;
  if (il) {
    const z = il.z_um;
    const bandColors = ["#ffffff", COLORS.orange, COLORS.purple, "#56B4E9", "#F0E442"];
    const bandName = (b) => `peak ${fmtSci(b.t2_peak)} s`;
    // One track per ILT peak: the geometric-mean T2 inside that peak's band, so the lines follow the heat-map ridges.
    const tracks = il.bands.map((b, k) => ({
      x: z, y: il.band_mean_T2[k], mode: "lines", name: bandName(b), connectgaps: false,
      line: { color: bandColors[k % bandColors.length], width: 2.5 },
    }));
    renderViewer(il, tracks);

    const palette = [COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple, "#56B4E9"];
    Plotly.react("plot-sp-iltsum", il.bands.map((b, k) => ({
      x: z, y: il.band_mean_T2[k], name: bandName(b), mode: "lines+markers", connectgaps: false,
      line: { color: palette[k % palette.length] }, marker: { size: 4 },
    })), baseLayout("ILT peak T₂ vs position (geometric mean per peak)", "z (µm)", "T₂ (s)", { yaxis: { ...t2, ...T2_AXIS(linear) } }), CFG);

    Plotly.react("plot-sp-iltfrac", il.bands.map((b, k) => ({
      x: z, y: il.band_fraction[k], name: bandName(b), mode: "lines", line: { color: palette[k % palette.length], width: 2 },
    })), baseLayout("Share of the signal in each ILT peak", "z (µm)", "Fraction", { yaxis: { ...t2, title: "Fraction", range: [0, 1], type: "linear" } }), CFG);
  }
}

// ── linked cross-section viewer: ILT map (centre), spectrum at a position (right), signal at a T2 (top) ──
const view = { key: "", i: 0, j: 0 };

function renderViewer(il, tracks) {
  const gd = $("plot-sp-ilt"), z = il.z_um, tau = il.tau;
  const np = z.length, nt = tau.length, key = `${np}x${nt}`;
  const zi = $("view-z"), tj = $("view-t2");
  if (view.key !== key) {                                   // new dataset or grid: start on the strongest position and its main peak
    const tot = il.spectra.map((row) => row.reduce((a, v) => a + (v ?? 0), 0));
    view.i = tot.indexOf(Math.max(...tot));
    const row = il.spectra[view.i].map((v) => v ?? -Infinity);
    view.j = row.indexOf(Math.max(...row));
    view.key = key;
  }
  zi.max = np - 1; zi.value = view.i;
  tj.max = nt - 1; tj.value = view.j;

  const zmap = tau.map((_, j) => il.spectra.map((row) => row[j]));            // [tau][position]
  const colAt = (j) => il.spectra.map((row) => row[j]);                        // signal vs z at one T2
  const grid = baseLayout("", "", "").xaxis;
  const withData = z.filter((_, p) => il.spectra[p].some((v) => v !== null && v > 0));
  const pad = withData.length > 1 ? 0.08 * (withData[withData.length - 1] - withData[0]) : 5;
  const zRange = withData.length ? [withData[0] - pad, withData[withData.length - 1] + pad] : [z[0], z[np - 1]];   // zoom to where there is signal
  const iTop = 1 + tracks.length, iRight = iTop + 1;
  // One fixed amplitude scale for both side panels (from the whole map), so relative amplitudes survive sweeping the sliders.
  let fMax = 0, fMin = 0;
  for (const row of il.spectra) for (const v of row) if (v !== null) { if (v > fMax) fMax = v; if (v < fMin) fMin = v; }
  const fRange = [fMin < 0 ? 1.05 * fMin : 0, 1.05 * fMax];
  const fixedScale = () => $("view-fixed").checked;
  const data = [
    { type: "heatmap", x: z, y: tau, z: zmap, colorscale: "Viridis", zsmooth: false, xaxis: "x", yaxis: "y",
      colorbar: { title: "f", len: 0.7, y: 0, yanchor: "bottom", x: 1.02 }, hovertemplate: "z %{x:.1f} µm<br>T₂ %{y:.2e} s<br>f %{z:.3g}<extra></extra>" },
    ...tracks.map((t) => ({ ...t, xaxis: "x", yaxis: "y" })),
    { x: z, y: colAt(view.j), xaxis: "x2", yaxis: "y2", mode: "lines", line: { color: COLORS.blue, width: 2 }, name: "signal at T₂", connectgaps: false },
    { x: il.spectra[view.i], y: tau, xaxis: "x3", yaxis: "y3", mode: "lines", line: { color: COLORS.green, width: 2 }, name: "spectrum at z", connectgaps: false },
  ];
  const cross = { color: "rgba(255,255,255,0.9)", width: 1, dash: "dot" };
  const mark = { color: COLORS.orange, width: 1.5, dash: "dot" };
  const L = (j) => tau[j];                    // shape coordinates on a log axis are data values
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: baseLayout("", "", "").font,
    margin: { l: 65, r: 90, t: 15, b: 50 }, showlegend: false, hovermode: "closest",
    xaxis: { ...grid, title: "z (µm)", domain: [0, 0.74], anchor: "y", range: zRange },
    yaxis: { ...grid, title: "T₂ (s)", type: "log", tickformat: ".0e", domain: [0, 0.7], anchor: "x", range: [Math.log10(tau[0]), Math.log10(tau[nt - 1])] },
    xaxis2: { ...grid, domain: [0, 0.74], anchor: "y2", matches: "x", showticklabels: false },
    yaxis2: { ...grid, title: "Signal", domain: [0.77, 1], anchor: "x2", range: fixedScale() ? fRange : undefined, autorange: !fixedScale() },
    xaxis3: { ...grid, title: "Signal", domain: [0.8, 1], anchor: "y3", range: fixedScale() ? fRange : undefined, autorange: !fixedScale() },
    yaxis3: { ...grid, domain: [0, 0.7], anchor: "x3", type: "log", matches: "y", showticklabels: false },
    shapes: [
      { type: "line", xref: "x", yref: "y domain", x0: z[view.i], x1: z[view.i], y0: 0, y1: 1, line: cross },
      { type: "line", xref: "x domain", yref: "y", x0: 0, x1: 1, y0: L(view.j), y1: L(view.j), line: cross },
      { type: "line", xref: "x2", yref: "y2 domain", x0: z[view.i], x1: z[view.i], y0: 0, y1: 1, line: mark },
      { type: "line", xref: "x3 domain", yref: "y3", x0: 0, x1: 1, y0: L(view.j), y1: L(view.j), line: mark },
    ],
  };
  Plotly.react(gd, data, layout, CFG);

  const update = () => {
    $("view-z-out").textContent = `${z[view.i].toFixed(1)} µm`;
    $("view-t2-out").textContent = `${fmtSci(tau[view.j])} s`;
    Plotly.restyle(gd, { x: [z], y: [colAt(view.j)] }, [iTop]);
    Plotly.restyle(gd, { x: [il.spectra[view.i]], y: [tau] }, [iRight]);
    Plotly.relayout(gd, {
      "shapes[0].x0": z[view.i], "shapes[0].x1": z[view.i], "shapes[2].x0": z[view.i], "shapes[2].x1": z[view.i],
      "shapes[1].y0": L(view.j), "shapes[1].y1": L(view.j), "shapes[3].y0": L(view.j), "shapes[3].y1": L(view.j),
    });
  };
  const nearest = (arr, v, log = false) => arr.reduce((best, x, k) => (Math.abs((log ? Math.log(x) : x) - (log ? Math.log(v) : v)) < Math.abs((log ? Math.log(arr[best]) : arr[best]) - (log ? Math.log(v) : v)) ? k : best), 0);
  $("view-fixed").onchange = () => {
    Plotly.relayout(gd, fixedScale()
      ? { "yaxis2.range": fRange, "yaxis2.autorange": false, "xaxis3.range": fRange, "xaxis3.autorange": false }
      : { "yaxis2.autorange": true, "xaxis3.autorange": true });
  };
  zi.oninput = () => { view.i = +zi.value; update(); };
  tj.oninput = () => { view.j = +tj.value; update(); };
  gd.removeAllListeners?.("plotly_click");
  gd.on("plotly_click", (ev) => {                          // click the map to move both cuts; click a side panel to move one
    const pt = ev.points && ev.points[0];
    if (!pt) return;
    const axis = pt.data.xaxis || "x";                     // heatmap and peak tracks are on "x"; the top panel is "x2", the spectrum "x3"
    if (axis === "x") { view.i = nearest(z, pt.x); view.j = nearest(tau, pt.y, true); }
    else if (axis === "x2") view.i = nearest(z, pt.x);
    else if (axis === "x3") view.j = nearest(tau, pt.y, true);
    zi.value = view.i; tj.value = view.j;
    update();
  });
  $("view-z-out").textContent = `${z[view.i].toFixed(1)} µm`;
  $("view-t2-out").textContent = `${fmtSci(tau[view.j])} s`;
}

// ── decay + fits (integrated decay, or the FT window sum) ────────────────
function renderFit(r) {
  const { t, y } = r.decay;
  const traces = [{ x: t, y, name: "Data", mode: "markers", marker: { size: 4, color: COLORS.grey }, xaxis: "x", yaxis: "y" }];
  const me = r.multiexp && r.multiexp.selected;
  if (me) {
    traces.push({ x: me.t_dense, y: me.fit_dense, name: `${me.n}-exp fit`, mode: "lines", line: { color: COLORS.orange, width: 2 } });
    traces.push({ x: t, y: me.resid, name: `${me.n}-exp residual`, mode: "lines", line: { color: COLORS.orange, width: 1 }, yaxis: "y2", showlegend: false });
  }
  if (r.ilt) {
    traces.push({ x: t, y: r.ilt.pred, name: "ILT fit", mode: "lines", line: { color: COLORS.green, width: 2, dash: "dash" } });
    traces.push({ x: t, y: r.ilt.resid, name: "ILT residual", mode: "lines", line: { color: COLORS.green, width: 1 }, yaxis: "y2", showlegend: false });
  }
  const ft = r.mode === "ft";
  $("fit-heading").textContent = ft ? "Window-summed decay and fits" : "Decay and fits";
  const base = baseLayout(ft ? "Echo decay (FT, summed over z-window)" : "Echo decay (phased, integrated)", "Echo time (s)", "Signal (arb.)");
  Plotly.react("plot-fit", traces, {
    ...base,
    margin: { ...base.margin, b: 70 },
    xaxis: { ...base.xaxis, anchor: "y2" },
    yaxis: { ...base.yaxis, domain: [0.3, 1], type: $("fit-log").checked ? "log" : "linear" },
    yaxis2: { ...base.yaxis, title: "Residual", domain: [0, 0.22] },
    legend: { orientation: "h", y: -0.22 },
  }, CFG);
}

function renderMultiexp(r) {
  const card = $("me-results");
  card.hidden = !r.multiexp;
  if (!r.multiexp) return;
  const me = r.multiexp.selected;
  const lvl = Math.round(me.ci_level * 100);
  $("me-equation").innerHTML = meEquation(me.n, me.offset);
  $("me-defs").innerHTML = meDefinitions(me.n);
  $("me-defs").hidden = me.n < 2;

  // Each value, its SE and its interval share one base exponent (e.g. 1.99e-2 ± 0.35e-2, [1.31e-2, 2.67e-2]).
  const rows = me.components.map((c, i) => {
    const okT = okUnc(c.T2, c.T2_se), okA = okUnc(c.A, c.A_se);
    const gT = sciGroup(c.T2, okT ? c.T2_se : null);
    const sciA = isOdd(c.A), gA = sciGroup(c.A, okA ? c.A_se : null);
    return [
      String(i + 1), gT.fmt(c.T2), okT ? "± " + gT.fmt(c.T2_se) : "undetermined", okT ? fmtCIWith(gT, c.T2_ci) : "—",
      sciA ? gA.fmt(c.A) : fmt(c.A), !okA ? "—" : sciA ? fmtCIWith(gA, c.A_ci) : fmtCIplain(c.A_ci),
      okUnc(c.fraction, c.fraction_se) ? `${pct(c.fraction)} ± ${pct(c.fraction_se)}` : pct(c.fraction),
    ];
  });
  table($("me-table"), ["#", "T₂ (s)", "SE", `${lvl}% CI`, "Amplitude", `${lvl}% CI`, "Fraction"], rows);

  const notes = [...me.warnings];
  if (r.multiexp.recommended_n !== me.n) {
    notes.push(`BIC prefers ${r.multiexp.recommended_n} component${r.multiexp.recommended_n > 1 ? "s" : ""} over the ${me.n} chosen (see model comparison).`);
  }
  $("me-warnings").replaceChildren(...notes.map((text) => Object.assign(document.createElement("p"), { className: "message warn", textContent: text })));

  const extra = [];
  if (me.offset) extra.push(`offset ${fmtPair(me.offset_value, me.offset_se)}`);
  extra.push(`total amplitude ${fmtPair(me.total_amplitude, me.total_amplitude_se)}`);
  if (okUnc(me.mean_T2, me.mean_T2_se)) {
    const gm = sciGroup(me.mean_T2, me.mean_T2_se);
    extra.push(`amplitude-weighted mean T₂ ${gm.fmt(me.mean_T2)} ± ${gm.fmt(me.mean_T2_se)} s`);
  } else {
    extra.push(`amplitude-weighted mean T₂ ${fmtSci(me.mean_T2)} s (uncertainty undetermined)`);
  }
  $("me-stats").textContent = extra.join(" · ");

  const sig = me.noise_sigma;
  statTable($("me-quality"), [
    ["RSS", fmtSci(me.rss), "Σ residual²; grows with amplitude² and with the number of echoes", LINKS.RSS],
    ["MSE", fmtSci(me.mse), "RSS / N; removes the echo count but still scales with amplitude²", LINKS.MSE],
    ["Residual SD", fmt(me.resid_sd), "√(RSS / (N − k)), in signal units"],
    ["Residual SD / amplitude", me.resid_sd_over_amp === null ? "—" : pct(me.resid_sd_over_amp, 2),
      "residual SD ÷ the fitted signal range (max − min over the echoes); independent of amplitude and echo count", LINKS.NRMSD],
    ["Noise σ", sig ? fmt(sig) : "—", sig ? "from the imaginary channel after phasing (model-free)" : "unavailable: the data has no imaginary channel"],
    ["Reduced χ²", me.chi2_red === null ? "—" : fmt(me.chi2_red), "RSS / (σ²(N − k)); ≈ 1 when the model explains everything the noise allows", LINKS.CHI2],
    ["R²", fmt(me.r2, 5), "fraction of variance explained (depends on how much of the decay was sampled)"],
    ["Degrees of freedom", String(me.dof), "N − k"],
  ]);
  $("quality-note").textContent = me.chi2_red === null ? "" :
    me.chi2_red > 2 ? "Reduced χ² well above 1: the model is missing structure in the decay (or the noise estimate is too low)."
    : me.chi2_red < 0.5 ? "Reduced χ² well below 1: the noise estimate is probably too high, or the fit is over-parameterised."
    : "Reduced χ² is close to 1: the residuals are consistent with the measured noise.";

  const best = r.multiexp.recommended_n;
  table($("me-compare-table"),
    ["Components", "Parameters", link("RSS", LINKS.RSS), link("MSE", LINKS.MSE), link("Residual SD / amplitude", LINKS.NRMSD),
     link("Reduced χ²", LINKS.CHI2), link("ΔAIC", LINKS.AIC), link("ΔBIC", LINKS.BIC), ""],
    r.multiexp.comparison.map((c) => [String(c.n), String(c.n_params), fmtSci(c.rss), fmtSci(c.mse),
      c.resid_sd_over_amp === null ? "—" : pct(c.resid_sd_over_amp, 2), c.chi2_red === null ? "—" : fmt(c.chi2_red),
      d1(c.d_aic), d1(c.d_bic),
      [c.n === me.n ? "shown above" : "", c.n === best ? "lowest BIC" : ""].filter(Boolean).join(", ")]));
}

function renderIlt(r) {
  $("ilt-results").hidden = !r.ilt;
  if (!r.ilt) return;
  $("ilt-equation").innerHTML = ILT_EQUATION;
  const il = r.ilt;
  const traces = [{ x: il.tau, y: il.spectrum, name: "ILT", mode: "lines", line: { color: COLORS.green, width: 2 } }];
  if (il.peaks.length) {
    traces.push({
      x: il.peaks.map((p) => p.T2_peak), y: il.peaks.map((p) => il.spectrum[il.tau.findIndex((t) => t === p.T2_peak)]),
      name: "Peaks", mode: "markers+text", marker: { color: COLORS.green, size: 8 },
      text: il.peaks.map((p) => fmtSci(p.T2_peak)), textposition: "top center", showlegend: false,
    });
  }
  const me = r.multiexp && r.multiexp.selected;
  if (me) {
    traces.push({
      x: me.components.map((c) => c.T2), y: me.components.map(() => 0), name: `${me.n}-exp fit T₂`, mode: "markers",
      marker: { symbol: "diamond", size: 10, color: COLORS.orange },
    });
  }
  const base = baseLayout("T₂ spectrum", "T₂ (s)", "Amplitude");
  // x in seconds, one labelled tick per decade: 1e-5, 1e-4, 1e-3, ...
  const xaxis = { ...base.xaxis, type: "log", dtick: 1, tickformat: ".0e", range: [Math.log10(il.t2_min), Math.log10(il.t2_max)] };
  Plotly.react("plot-spectrum", traces, { ...base, xaxis }, CFG);
  $("ilt-stats").textContent = `α = ${fmtSci(il.alpha)} · ${il.n_steps} grid points (max ${r.settings.ilt.n_steps}) · T₂ ${fmtSci(il.t2_min)}–${fmtSci(il.t2_max)} s · RMSE ${fmt(il.rmse)}`;
  statTable($("ilt-quality"), [
    ["RSS", fmtSci(il.rss), "Σ residual²", LINKS.RSS],
    ["MSE", fmtSci(il.mse), "RSS / N", LINKS.MSE],
    ["RMSE / amplitude", il.rmse_over_amp === null ? "—" : pct(il.rmse_over_amp, 2), "RMSE ÷ the fitted signal range (max − min over the echoes)", LINKS.NRMSD],
    ["Noise σ", il.noise_sigma ? fmt(il.noise_sigma) : "—", "from the imaginary channel"],
    ["χ² per point", il.chi2_per_point === null ? "—" : fmt(il.chi2_per_point),
      "RSS / (σ² N); regularisation has no clean parameter count, so this is not a reduced χ²", LINKS.CHI2],
  ]);
  table($("ilt-peaks"), ["Peak T₂ (s)", "Mean T₂ of peak (s)", "Fraction of signal"],
    il.peaks.length ? il.peaks.map((p) => [fmtSci(p.T2_peak), fmtSci(p.T2_mean), pct(p.fraction)]) : [["no resolved peaks", "", ""]]);

  $("lcurve-card").hidden = !il.sweep;
  if (il.sweep) {
    const base2 = baseLayout("α sweep (L-curve)", "Fit RMSE", "Spectrum roughness ‖D₂x‖/‖x‖");
    Plotly.react("plot-lcurve", [{
      x: il.sweep.map((s) => s.rmse), y: il.sweep.map((s) => s.roughness), mode: "lines+markers+text",
      text: il.sweep.map((s) => "α=" + fmtSci(s.alpha, 1)), textposition: "top right", line: { color: COLORS.purple },
      marker: { color: COLORS.purple }, showlegend: false,
    }], { ...base2, xaxis: { ...base2.xaxis, type: "log" }, yaxis: { ...base2.yaxis, type: "log" } }, CFG);
  }
}

export function renderResults() {
  const r = state.results;
  if (!r) return;
  $("results-summary").textContent =
    `${r.mode === "ft" ? "FT of each echo" : "Integrated each echo"} (autophase rotation ${fmt(r.decay.phase_deg, 4)}°) · ${r.decay.t.length} echoes`;
  renderFt(r);
  renderSpatial(r);
  renderFit(r);
  renderMultiexp(r);
  renderIlt(r);
}

export function initResults() {
  $("fit-log").addEventListener("change", renderResults);
  $("sp-linear").addEventListener("change", renderResults);
}
