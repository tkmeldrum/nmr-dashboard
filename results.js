// Results tab: decay + fits, fit tables, ILT spectrum.
import { $, state } from "./state.js";
import { baseLayout, CFG, COLORS, fmt, fmtCI as fmtCIplain, fmtCISci, fmtPair, fmtSci, pct } from "./plotutil.js";

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

function renderFt(r) {
  $("ft-results").hidden = !r.ft;
  if (!r.ft) return;
  const f = r.ft, [w0, w1] = f.window;
  const nuc = f.nucleus === "19F" ? "¹⁹F" : "¹H";
  $("ft-stats").textContent = `${nuc} (γ/2π = ${f.gamma_mhz_per_t} MHz/T), G = ${f.G} T/m, zero-fill ×2^${f.zero_fill} → ${f.nfft} points, ` +
    `echo centre at point ${f.center_point}. Decay = sum over z from ${fmt(w0)} to ${fmt(w1)} µm${f.window_auto ? " (automatic: profile > 10% of max)" : ""}.`;
  const base = baseLayout("Spatial profile", "z (µm)", "|Σ echoes| (arb.)");
  Plotly.react("plot-profile", [{ x: f.z_um, y: f.profile, mode: "lines", name: "Profile", line: { color: COLORS.blue } }], {
    ...base, showlegend: false,
    shapes: [{ type: "rect", xref: "x", yref: "paper", x0: w0, x1: w1, y0: 0, y1: 1, fillcolor: COLORS.orange, opacity: 0.18, line: { width: 0 } }],
  }, CFG);
  const t = r.decay.t, h = f.heat;
  const base2 = baseLayout("Echo magnitude after FT", "Echo time (s)", "z (µm)");
  Plotly.react("plot-ftmap", [{ type: "heatmap", z: h.mag, x: h.echo_idx.map((i) => t[i]), y: h.z_um, colorscale: "Viridis", colorbar: { title: "|S|" } }], base2, CFG);
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
  const base = baseLayout(r.mode === "ft" ? "Echo decay (FT, summed over z-window)" : "Echo decay (phased, integrated)", "Echo time (s)", "Signal (arb.)");
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
    String(i + 1), fmtSci(c.T2), "± " + fmtSci(c.T2_se), fmtCISci(c.T2_ci),
    fmt(c.A), fmtCIplain(c.A_ci), `${pct(c.fraction)} ± ${pct(c.fraction_se)}`,
  ]);
  table($("me-table"), ["#", "T₂ (s)", "SE", `${lvl}% CI`, "Amplitude", `${lvl}% CI`, "Fraction"], rows);

  const w = $("me-warnings");
  const notes = [...me.warnings];
  if (r.multiexp.recommended_n !== me.n) {
    notes.push(`BIC prefers ${r.multiexp.recommended_n} component${r.multiexp.recommended_n > 1 ? "s" : ""} over the ${me.n} chosen (see model comparison).`);
  }
  w.replaceChildren(...notes.map((text) => Object.assign(document.createElement("p"), { className: "message warn", textContent: text })));

  const extra = [];
  if (me.offset) extra.push(`offset ${fmtPair(me.offset_value, me.offset_se)}`);
  extra.push(`total amplitude ${fmtPair(me.total_amplitude, me.total_amplitude_se)}`);
  extra.push(`amplitude-weighted mean T₂ ${fmtSci(me.mean_T2)} ± ${fmtSci(me.mean_T2_se)} s`);
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
  renderFit(r);
  renderMultiexp(r);
  renderIlt(r);
}

export function initResults() {
  $("fit-log").addEventListener("change", renderResults);
}
