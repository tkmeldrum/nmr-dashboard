// T2-D results: the map with linked slices, the regions table, and data / fit / residual maps.
import { $, state } from "./state.js";
import { baseLayout, CFG, COLORS, fmt, fmtSci, ink, pct } from "./plotutil.js";

const view = { key: "", i: 0, j: 0 };
const LOGTICKS = { type: "log", tickformat: ".0e", dtick: 1 };      // one labelled tick per decade, as on the ILT plot

function table(el, headers, rows) {
  const mk = (tag, text) => Object.assign(document.createElement(tag), { textContent: text });
  const head = document.createElement("tr");
  headers.forEach((h) => head.appendChild(mk("th", h)));
  el.replaceChildren(head, ...rows.map((r) => {
    const tr = document.createElement("tr");
    r.forEach((c) => tr.appendChild(mk("td", c)));
    return tr;
  }));
}

export function renderT2d(r) {
  const box = $("t2d-results");
  box.hidden = !r.t2d;
  if (!r.t2d) return;
  const t = r.t2d, skip = r.decay.skip || 0;
  const [lo, hi] = t.d_sensitive, bmin = Math.min(...t.b_vec) * 1e9, bmax = Math.max(...t.b_vec) * 1e9;

  $("t2d-stats").textContent =
    `${t.n_T2} T₂ points ${fmtSci(t.t2_min)}–${fmtSci(t.t2_max)} s × ${t.n_D} D points ${fmtSci(t.d_min)}–${fmtSci(t.d_max)} m²/s · α = ${fmtSci(t.alpha)} · ` +
    `RMSE / signal range ${t.rmse_over_amp === null ? "—" : pct(t.rmse_over_amp, 2)} · noise σ ${t.noise_sigma ? fmt(t.noise_sigma) : "—"} (imaginary channel) · ` +
    `χ² per point ${t.chi2_per_point === null ? "—" : fmt(t.chi2_per_point)}` + (skip ? ` · first ${skip} echoes excluded` : "");

  const msgs = [];
  if (t.fraction_in_sensitive_window !== null && t.fraction_in_sensitive_window < 0.9) {
    msgs.push(`Only ${pct(t.fraction_in_sensitive_window, 0)} of the spectrum lies where the data can constrain it. Your b-values (${fmtSci(bmin)} – ${fmtSci(bmax)} s/m²) can tell apart diffusion ` +
      `coefficients from about ${fmtSci(lo)} to ${fmtSci(hi)} m²/s; signal the regularised solution places outside that range (the hatched band on the map) is extrapolation, not measurement. ` +
      "Use “Use the range the b-values can resolve” on the Process tab and run again to see only what the data support."
      + " To probe diffusion coefficients above this range (faster diffusion), decrease b by decreasing the gradient encoding time τ or the diffusion time Δ; to probe below it (slower diffusion), increase b by increasing τ or Δ.");
  }
  const edge = t.peaks.filter((p) => p.D_peak <= t.D[0] * 1.0001 || p.D_peak >= t.D[t.D.length - 1] * 0.9999 || p.T2_peak <= t.T2[0] * 1.0001 || p.T2_peak >= t.T2[t.T2.length - 1] * 0.9999);
  if (edge.length) msgs.push(`${edge.length} region${edge.length > 1 ? "s peak" : " peaks"} at the edge of the grid: signal may extend beyond the range you chose, so widen it to check.`);
  $("t2d-warnings").replaceChildren(...msgs.map((text) => Object.assign(document.createElement("p"), { className: "message warn", textContent: text })));

  table($("t2d-regions"), ["#", "Fraction", "Fraction of the resolvable part", "T₂ (s), mean", "D (m²/s), mean", "T₂ at max (s)", "D at max (m²/s)", "Constrained by the data"],
    t.peaks.length ? t.peaks.map((p, i) => [String(i + 1), pct(p.fraction), p.fraction_in_window === null ? "—" : pct(p.fraction_in_window), fmtSci(p.T2_mean), fmtSci(p.D_mean), fmtSci(p.T2_peak), fmtSci(p.D_peak), p.constrained ? "yes" : "no"])
      : [["–", "no regions", "", "", "", "", "", ""]]);

  renderViewer(t, [lo, hi]);
  renderMaps(t);
  renderT2dModel(r);
}

const POP_COLORS = [COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple];
const finite = (x) => x !== null && x !== undefined && Number.isFinite(x);

// The few-population model: table, comparison and the populations drawn on the map.
function renderT2dModel(r) {
  const m = r.t2d && r.t2d.model;
  $("t2d-model").hidden = !m;
  if (!m) return;
  $("t2d-model-note").textContent = `${m.K} population${m.K > 1 ? "s" : ""} (BIC prefers ${m.recommended_k}); χ² per point ${fmt(m.chi2_per_point)}; ` +
    `D allowed from ${fmtSci(m.d_window[0])} to ${fmtSci(m.d_window[1])} m²/s; intervals are least-squares 95% intervals.`;
  $("t2d-model-warnings").replaceChildren(...m.warnings.map((text) => Object.assign(document.createElement("p"), { className: "message warn", textContent: text })));
  const val = (v, ci, bound) => (bound ? `${fmtSci(v)} (a bound only)` : finite(ci[0]) ? `${fmtSci(v)} [${fmtSci(ci[0])}, ${fmtSci(ci[1])}]` : `${fmtSci(v)} (interval undetermined)`);
  table($("t2d-model-table"), ["#", "Fraction", "T₂ (s)", "D (m²/s)", "Amplitude", "Note"], m.populations.map((p, i) => [
    String(i + 1), finite(p.fraction_se) ? `${pct(p.fraction)} ± ${pct(p.fraction_se)}` : pct(p.fraction),
    val(p.T2, p.T2_ci, p.T2_at_limit), val(p.D, p.D_ci, p.D_at_limit), fmt(p.A),
    [p.D_at_limit ? `D at the ${p.D_at_limit} limit` : "", p.T2_at_limit ? `T₂ at the ${p.T2_at_limit} limit` : "", !p.supported && !p.D_at_limit && !p.T2_at_limit ? "not supported by the data" : ""].filter(Boolean).join("; ") || "–",
  ]));
  table($("t2d-model-compare"), ["Populations", "Parameters", "χ² per point", "ΔAIC", "ΔBIC", ""], m.comparison.map((c) => [
    String(c.k), String(c.n_params), fmt(c.chi2_per_point), fmt(c.d_aic, 3), fmt(c.d_bic, 3), c.k === m.K ? "chosen" : c.k === m.recommended_k ? "BIC's choice" : "",
  ]));
  renderT2dPopulations();
}

// The populations on the regularised map, with bootstrap / MCMC samples when they exist (redrawn when those finish).
export function renderT2dPopulations() {
  const r = state.results, m = r && r.t2d && r.t2d.model;
  if (!m) return;
  const t = r.t2d, traces = [{ type: "heatmap", x: t.T2, y: t.D, z: t.spectrum, colorscale: "Greys", reversescale: true, showscale: false, opacity: 0.6, hoverinfo: "skip", zsmooth: false }];
  const xs = m.populations.flatMap((p) => [p.T2, ...(finite(p.T2_ci[0]) ? p.T2_ci : [])]);
  const u = state.unc || {};
  for (const [src, label, size] of [[u.boot, "bootstrap", 3], [u.mcmc, "MCMC", 3]]) {
    const sc = src && src.scatter;
    if (!sc) continue;
    sc.T2.forEach((arr, k) => {
      xs.push(Math.min(...arr), Math.max(...arr));
      traces.push({ x: arr, y: sc.D[k], mode: "markers", marker: { size, color: POP_COLORS[k % 4], opacity: 0.3 }, name: label, legendgroup: label, showlegend: k === 0, hoverinfo: "skip" });
    });
  }
  m.populations.forEach((p, i) => {
    const bound = p.D_at_limit || p.T2_at_limit;
    traces.push({
      x: [p.T2], y: [p.D], mode: "markers+text", text: [`${i + 1}: ${Math.round(100 * p.fraction)}%`], textposition: "top right", showlegend: false,
      marker: { size: 11, color: p.supported && !bound ? POP_COLORS[i % 4] : "rgba(0,0,0,0)", symbol: p.D_at_limit === "low" ? "triangle-down" : p.D_at_limit === "high" ? "triangle-up" : "circle", line: { color: ink(), width: 2 } },
      error_x: finite(p.T2_ci[0]) ? { type: "data", symmetric: false, array: [p.T2_ci[1] - p.T2], arrayminus: [p.T2 - p.T2_ci[0]], color: ink(), thickness: 1.5 } : undefined,
      error_y: finite(p.D_ci[0]) ? { type: "data", symmetric: false, array: [p.D_ci[1] - p.D], arrayminus: [p.D - p.D_ci[0]], color: ink(), thickness: 1.5 } : undefined,
      name: `population ${i + 1}`,
    });
  });
  const [lo, hi] = m.d_window, base = baseLayout("", "T₂ (s)", "D (m²/s)", { margin: { l: 70, r: 15, t: 10, b: 50 } });
  const band = (y0, y1) => ({ type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0, y1, fillcolor: "rgba(122,130,140,0.25)", line: { width: 0 }, layer: "above" });
  Plotly.react("plot-t2d-pop", traces, {
    ...base,
    xaxis: { ...base.xaxis, ...LOGTICKS, range: [Math.log10(Math.max(t.T2[0], Math.min(...xs) / 4)), Math.log10(Math.min(t.T2[t.T2.length - 1], Math.max(...xs) * 4))] },
    yaxis: { ...base.yaxis, ...LOGTICKS, range: [Math.log10(t.D[0] / 4), Math.log10(t.D[t.D.length - 1])] },     // room below the lowest D for a population sitting on the limit
    shapes: [band(t.D[0] / 10, lo), band(hi, t.D[t.D.length - 1] * 10)],
    legend: { orientation: "h", y: -0.2 },
  }, CFG);
}

function renderViewer(t, window) {
  const gd = $("plot-t2d"), T2 = t.T2, D = t.D, S = t.spectrum;            // S[iD][iT2]
  const key = `${S.length}x${T2.length}`;
  const sd = $("t2d-view-d"), st = $("t2d-view-t2");
  if (view.key !== key) {                                                  // start at the strongest bin
    let best = -1;
    S.forEach((row, i) => row.forEach((v, j) => { if (v > best) { best = v; view.i = i; view.j = j; } }));
    view.key = key;
  }
  sd.max = D.length - 1; sd.value = view.i;
  st.max = T2.length - 1; st.value = view.j;
  let fMax = 0;
  for (const row of S) for (const v of row) if (v > fMax) fMax = v;
  const grid = baseLayout("", "", "").xaxis;
  const proj = () => $("t2d-proj").checked;
  const topY = () => (proj() ? t.projection_T2 : S[view.i]);
  const rightX = () => (proj() ? t.projection_D : S.map((row) => row[view.j]));
  const cross = { color: "rgba(255,255,255,0.9)", width: 1, dash: "dot" }, mark = { color: COLORS.orange, width: 1.5, dash: "dot" };
  const markers = {
    x: t.peaks.map((p) => p.T2_mean), y: t.peaks.map((p) => p.D_mean), mode: "markers+text", showlegend: false,
    text: t.peaks.map((p) => pct(p.fraction, 0)), textposition: "top right", textfont: { color: "white", size: 11 },
    marker: { symbol: t.peaks.map((p) => (p.constrained ? "circle-open" : "x-open")), size: 11, color: "white", line: { width: 2, color: "white" } },
    hovertemplate: "T₂ %{x:.2e} s<br>D %{y:.2e} m²/s<br>%{text}<extra>region</extra>",
  };
  const data = [
    { type: "heatmap", x: T2, y: D, z: S, colorscale: "Viridis", zsmooth: false, xaxis: "x", yaxis: "y", colorbar: { title: "f", len: 0.7, y: 0, yanchor: "bottom", x: 1.02 },
      hovertemplate: "T₂ %{x:.2e} s<br>D %{y:.2e} m²/s<br>f %{z:.3g}<extra></extra>" },
    { ...markers, xaxis: "x", yaxis: "y" },
    { x: T2, y: topY(), xaxis: "x2", yaxis: "y2", mode: "lines", line: { color: COLORS.blue, width: 2 }, name: "top" },
    { x: rightX(), y: D, xaxis: "x3", yaxis: "y3", mode: "lines", line: { color: COLORS.green, width: 2 }, name: "right" },
  ];
  const band = (a, b) => ({ type: "rect", xref: "x domain", yref: "y", x0: 0, x1: 1, y0: a, y1: b, fillcolor: "rgba(255,255,255,0.35)", line: { width: 0 }, layer: "above" });
  const shapes = [
    { type: "line", xref: "x", yref: "y domain", x0: T2[view.j], x1: T2[view.j], y0: 0, y1: 1, line: cross },
    { type: "line", xref: "x domain", yref: "y", x0: 0, x1: 1, y0: D[view.i], y1: D[view.i], line: cross },
    { type: "line", xref: "x2", yref: "y2 domain", x0: T2[view.j], x1: T2[view.j], y0: 0, y1: 1, line: mark },
    { type: "line", xref: "x3 domain", yref: "y3", x0: 0, x1: 1, y0: D[view.i], y1: D[view.i], line: mark },
  ];
  if (D[0] < window[0]) shapes.push(band(D[0] / 1.5, window[0]));           // below what the b-values can resolve
  if (D[D.length - 1] > window[1]) shapes.push(band(window[1], D[D.length - 1] * 1.5));   // above it
  const fixedRange = () => (!proj() && $("t2d-fixed").checked ? [0, 1.05 * fMax] : undefined);
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: baseLayout("", "", "").font,
    margin: { l: 70, r: 90, t: 15, b: 50 }, showlegend: false, hovermode: "closest",
    xaxis: { ...grid, ...LOGTICKS, title: "T₂ (s)", domain: [0, 0.74], anchor: "y", range: [Math.log10(T2[0]), Math.log10(T2[T2.length - 1])] },
    yaxis: { ...grid, ...LOGTICKS, title: "D (m²/s)", domain: [0, 0.7], anchor: "x", range: [Math.log10(D[0]), Math.log10(D[D.length - 1])] },
    xaxis2: { ...grid, ...LOGTICKS, domain: [0, 0.74], anchor: "y2", matches: "x", showticklabels: false },
    yaxis2: { ...grid, title: proj() ? "Σ over D" : "f at D", domain: [0.77, 1], anchor: "x2", autorange: !fixedRange(), range: fixedRange() },
    xaxis3: { ...grid, title: proj() ? "Σ over T₂" : "f at T₂", domain: [0.8, 1], anchor: "y3", autorange: !fixedRange(), range: fixedRange() },
    yaxis3: { ...grid, ...LOGTICKS, domain: [0, 0.7], anchor: "x3", matches: "y", showticklabels: false },
    shapes,
  };
  Plotly.react(gd, data, layout, CFG);

  const update = () => {
    $("t2d-view-d-out").textContent = `${fmtSci(D[view.i])} m²/s`;
    $("t2d-view-t2-out").textContent = `${fmtSci(T2[view.j])} s`;
    Plotly.restyle(gd, { x: [T2], y: [topY()] }, [2]);
    Plotly.restyle(gd, { x: [rightX()], y: [D] }, [3]);
    const r = fixedRange();
    Plotly.relayout(gd, {
      "shapes[0].x0": T2[view.j], "shapes[0].x1": T2[view.j], "shapes[2].x0": T2[view.j], "shapes[2].x1": T2[view.j],
      "shapes[1].y0": D[view.i], "shapes[1].y1": D[view.i], "shapes[3].y0": D[view.i], "shapes[3].y1": D[view.i],
      "yaxis2.title.text": proj() ? "Σ over D" : "f at D", "xaxis3.title.text": proj() ? "Σ over T₂" : "f at T₂",
      "yaxis2.autorange": !r, "xaxis3.autorange": !r, ...(r ? { "yaxis2.range": r, "xaxis3.range": r } : {}),
    });
  };
  const nearest = (arr, v) => arr.reduce((best, x, k) => (Math.abs(Math.log(x / v)) < Math.abs(Math.log(arr[best] / v)) ? k : best), 0);
  sd.oninput = () => { view.i = +sd.value; update(); };
  st.oninput = () => { view.j = +st.value; update(); };
  $("t2d-proj").onchange = update;
  $("t2d-fixed").onchange = update;
  gd.removeAllListeners?.("plotly_click");
  gd.on("plotly_click", (ev) => {                               // click the map to move both cuts; click a side panel to move one
    const pt = ev.points && ev.points[0];
    if (!pt) return;
    const axis = pt.data.xaxis || "x";
    if (axis === "x") { view.j = nearest(T2, pt.x); view.i = nearest(D, pt.y); }
    else if (axis === "x2") view.j = nearest(T2, pt.x);
    else if (axis === "x3") view.i = nearest(D, pt.y);
    sd.value = view.i; st.value = view.j;
    update();
  });
  $("t2d-view-d-out").textContent = `${fmtSci(D[view.i])} m²/s`;
  $("t2d-view-t2-out").textContent = `${fmtSci(T2[view.j])} s`;
}

function renderMaps(t) {
  const b = t.b_vec.map((x) => x * 1e9), e = t.echo_s;
  const pred = t.pred, resid = t.resid, data = pred.map((row, i) => row.map((v, j) => v + resid[i][j]));
  let lo = Infinity, hi = -Infinity, rmax = 0;
  data.forEach((row) => row.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
  resid.forEach((row) => row.forEach((v) => { if (Math.abs(v) > rmax) rmax = Math.abs(v); }));
  const lay = (title) => {
    const base = baseLayout(title, "Echo time (s)", "b (s/m²)", { margin: { l: 70, r: 10, t: 35, b: 45 } });
    return { ...base, yaxis: { ...base.yaxis, type: "log", tickformat: ".0e" } };
  };
  const hm = (z, extra) => [{ type: "heatmap", x: e, y: b, z, zsmooth: false, ...extra }];
  Plotly.react("plot-t2d-data", hm(data, { colorscale: "Viridis", zmin: lo, zmax: hi, colorbar: { thickness: 12 } }), lay("Data"), CFG);
  Plotly.react("plot-t2d-fit", hm(pred, { colorscale: "Viridis", zmin: lo, zmax: hi, colorbar: { thickness: 12 } }), lay("Fit"), CFG);
  Plotly.react("plot-t2d-resid", hm(resid, { colorscale: "RdBu", reversescale: true, zmin: -rmax, zmax: rmax, colorbar: { thickness: 12 } }), lay("Residual"), CFG);
}
