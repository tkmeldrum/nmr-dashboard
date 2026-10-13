// Results tab: decay + fits, fit tables, ILT spectrum.
import { $, state } from "./state.js";
import { baseLayout, CFG, COLORS, fmt, fmtCI, pct } from "./plotutil.js";

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

function renderFit(r) {
  const { t, y } = r.decay;
  const grey = COLORS.grey;
  const traces = [{ x: t, y, name: "Data", mode: "markers", marker: { size: 4, color: grey }, xaxis: "x", yaxis: "y" }];
  const me = r.multiexp && r.multiexp.selected;
  if (me) {
    traces.push({ x: me.t_dense, y: me.fit_dense, name: `${me.n}-exp fit`, mode: "lines", line: { color: COLORS.orange, width: 2 } });
    traces.push({ x: t, y: me.resid, name: `${me.n}-exp residual`, mode: "lines", line: { color: COLORS.orange, width: 1 }, yaxis: "y2", showlegend: false });
  }
  if (r.ilt) {
    traces.push({ x: t, y: r.ilt.pred, name: "ILT fit", mode: "lines", line: { color: COLORS.green, width: 2, dash: "dash" } });
    traces.push({ x: t, y: r.ilt.resid, name: "ILT residual", mode: "lines", line: { color: COLORS.green, width: 1 }, yaxis: "y2", showlegend: false });
  }
  const base = baseLayout("Echo decay (phased, integrated)", "Echo time (s)", "Signal (arb.)");
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
  const rows = me.components.map((c, i) => [
    String(i + 1), fmt(c.T2), "± " + fmt(c.T2_se), fmtCI(c.T2_ci),
    fmt(c.A), fmtCI(c.A_ci), `${pct(c.fraction)} ± ${pct(c.fraction_se)}`,
  ]);
  table($("me-table"), ["#", "T₂ (s)", "SE", `${lvl}% CI`, "Amplitude", `${lvl}% CI`, "Fraction"], rows);

  const w = $("me-warnings");
  const notes = [...me.warnings];
  if (r.multiexp.recommended_n !== me.n) {
    notes.push(`BIC prefers ${r.multiexp.recommended_n} component${r.multiexp.recommended_n > 1 ? "s" : ""} over the ${me.n} chosen (see model comparison).`);
  }
  w.replaceChildren(...notes.map((text) => Object.assign(document.createElement("p"), { className: "message warn", textContent: text })));

  const extra = [];
  if (me.offset) extra.push(`offset ${fmt(me.offset_value)} ± ${fmt(me.offset_se)}`);
  extra.push(`total amplitude ${fmt(me.total_amplitude)} ± ${fmt(me.total_amplitude_se)}`);
  extra.push(`amplitude-weighted mean T₂ ${fmt(me.mean_T2)} ± ${fmt(me.mean_T2_se)} s`);
  extra.push(`residual SD ${fmt(me.resid_sd)}`, `R² ${fmt(me.r2, 5)}`, `${me.dof} degrees of freedom`);
  $("me-stats").textContent = extra.join(" · ");

  const cmp = r.multiexp.comparison;
  const best = r.multiexp.recommended_n;
  table($("me-compare-table"), ["Components", "Parameters", "RSS", "AIC", "BIC", ""],
    cmp.map((c) => [String(c.n), String(c.n_params), fmt(c.rss), fmt(c.aic, 5), fmt(c.bic, 5),
      [c.n === me.n ? "shown above" : "", c.n === best ? "lowest BIC" : ""].filter(Boolean).join(", ")]));
}

function renderIlt(r) {
  $("ilt-results").hidden = !r.ilt;
  if (!r.ilt) return;
  const il = r.ilt;
  const traces = [{ x: il.tau, y: il.spectrum, name: "ILT", mode: "lines", line: { color: COLORS.green, width: 2 } }];
  if (il.peaks.length) {
    traces.push({
      x: il.peaks.map((p) => p.T2_peak), y: il.peaks.map((p) => il.spectrum[il.tau.findIndex((t) => t === p.T2_peak)]),
      name: "Peaks", mode: "markers+text", marker: { color: COLORS.green, size: 8 },
      text: il.peaks.map((p) => fmt(p.T2_peak)), textposition: "top center", showlegend: false,
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
  Plotly.react("plot-spectrum", traces, { ...base, xaxis: { ...base.xaxis, type: "log" } }, CFG);
  $("ilt-stats").textContent = `α = ${fmt(il.alpha)} · ${il.n_steps} grid points (max ${fmt(r.settings.ilt.n_steps)}) · T₂ ${fmt(il.t2_min)}–${fmt(il.t2_max)} s · RMSE ${fmt(il.rmse)}`;
  table($("ilt-peaks"), ["Peak T₂ (s)", "Mean T₂ of peak (s)", "Fraction of signal"],
    il.peaks.length ? il.peaks.map((p) => [fmt(p.T2_peak), fmt(p.T2_mean), pct(p.fraction)]) : [["no resolved peaks", "", ""]]);

  $("lcurve-card").hidden = !il.sweep;
  if (il.sweep) {
    const base2 = baseLayout("α sweep (L-curve)", "Fit RMSE", "Spectrum roughness ‖D₂x‖/‖x‖");
    Plotly.react("plot-lcurve", [{
      x: il.sweep.map((s) => s.rmse), y: il.sweep.map((s) => s.roughness), mode: "lines+markers+text",
      text: il.sweep.map((s) => "α=" + fmt(s.alpha, 1)), textposition: "top right", line: { color: COLORS.purple },
      marker: { color: COLORS.purple }, showlegend: false,
    }], { ...base2, xaxis: { ...base2.xaxis, type: "log" }, yaxis: { ...base2.yaxis, type: "log" } }, CFG);
  }
}

export function renderResults() {
  const r = state.results;
  if (!r) return;
  $("results-summary").textContent =
    `Integrated each echo (autophase rotation ${fmt(r.decay.phase_deg, 4)}°) · ${r.decay.t.length} echoes`;
  renderFit(r);
  renderMultiexp(r);
  renderIlt(r);
}

export function initResults() {
  $("fit-log").addEventListener("change", renderResults);
}
