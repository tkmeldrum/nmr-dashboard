// Raw-data plots (tab 2).
import { $, state } from "./state.js";
import { baseLayout, CFG, COLORS } from "./plotutil.js";

function renderRawT1(raw) {
  // saturation recovery: one curve, delay vs amplitude (the heat map and the single-echo views do not apply)
  for (const id of ["plot-heat", "plot-echo"]) $(id).closest(".plot-card").hidden = true;
  const base = baseLayout("Saturation recovery (as measured)", "Delay (s)", "Amplitude (arb.)");
  Plotly.react("plot-decay", [{ x: raw.t, y: raw.y, mode: "lines+markers", name: "Data", line: { color: COLORS.blue }, marker: { size: 6 } }], {
    ...base, showlegend: false,
    xaxis: { ...base.xaxis, ...($("decay-logx").checked ? { type: "log", tickformat: ".0e", dtick: 1 } : { type: "linear" }) },
    yaxis: { ...base.yaxis, type: $("decay-log").checked ? "log" : "linear" },
  }, CFG);
}

function renderRawT2D(raw) {
  // T2-D: echo trains at several diffusion delays. Map of everything, signal vs b at one echo (slider), and echo decays at three b-values.
  for (const id of ["plot-heat", "plot-echo"]) $(id).closest(".plot-card").hidden = false;
  const b = raw.b_vec.map((x) => x * 1e9), t = raw.echo_time_s, e = +$("echo-idx").value;
  const bAxis = { ...baseLayout("", "", "").xaxis, title: "b (s/m²)" };
  Plotly.react("plot-heat", [{ type: "heatmap", x: t, y: b, z: raw.data, colorscale: "Viridis", colorbar: { title: "signal" }, zsmooth: false }],
    (() => { const l = baseLayout("Data: echo × diffusion delay", "Echo time (s)", "b (s/m²)"); return { ...l, yaxis: { ...l.yaxis, type: "log", tickformat: ".0e", dtick: 1 } }; })(), CFG);          // dtick 1: one labelled tick per decade, not 2 to 9
  $("echo-idx-out").textContent = `${e + 1} of ${raw.n_echoes} (t = ${+t[e].toPrecision(4)} s)`;
  const l2 = baseLayout(`Signal vs b at echo ${e + 1}`, "b (s/m²)", "Signal (arb.)");
  Plotly.react("plot-echo", [{ x: b, y: raw.data.map((row) => row[e]), mode: "lines+markers", line: { color: COLORS.blue }, marker: { size: 5 } }],
    { ...l2, showlegend: false, xaxis: { ...l2.xaxis, ...($("decay-logx").checked ? { type: "log", tickformat: ".0e", dtick: 1 } : { type: "linear" }) } }, CFG);
  const pick = [0, Math.floor((b.length - 1) / 2), b.length - 1];
  const palette = [COLORS.blue, COLORS.orange, COLORS.green];
  const l3 = baseLayout("Echo decays at three b-values", "Echo time (s)", "Signal (arb.)");
  Plotly.react("plot-decay", pick.map((i, k) => ({ x: t, y: raw.data[i], mode: "lines", name: `b = ${b[i].toExponential(1)} s/m²`, line: { color: palette[k], width: 1.5 } })),
    { ...l3, yaxis: { ...l3.yaxis, type: $("decay-log").checked ? "log" : "linear" } }, CFG);
}

export function renderRaw() {
  const raw = state.result && state.result.raw;
  if (!raw) return;
  if (state.result.kind === "t1") return renderRawT1(raw);
  if (state.result.kind === "t2d") return renderRawT2D(raw);
  for (const id of ["plot-heat", "plot-echo"]) $(id).closest(".plot-card").hidden = false;
  const t = raw.echo_time_s;
  const mag = raw.real.map((row, p) => row.map((re, e) => Math.hypot(re, raw.imag[p][e])));

  Plotly.react("plot-heat", [{
    type: "heatmap", z: mag, x: t, y: mag.map((_, i) => i + 1), colorscale: "Viridis",
    colorbar: { title: "|S|" },
  }], baseLayout("Echo magnitude", "Echo time (s)", "Point within echo"), CFG);

  const e = +$("echo-idx").value;
  $("echo-idx-out").textContent = `${e + 1} of ${raw.n_echoes} (t = ${+t[e].toPrecision(4)} s)`;
  const pts = raw.real.map((_, i) => i + 1);
  // y-range fixed from the second echo (the first can be distorted) plus 10% headroom, so the
  // echo shape is comparable across the slider.
  const ref = Math.min(1, raw.n_echoes - 1);
  const refVals = raw.real.flatMap((r, p) => [r[ref], raw.imag[p][ref], mag[p][ref]]);
  const pad = 0.1 * Math.max(...refVals.map(Math.abs));
  const echoRange = [Math.min(...refVals) - pad, Math.max(...refVals) + pad];
  Plotly.react("plot-echo", [
    { x: pts, y: raw.real.map((r) => r[e]), name: "Real", mode: "lines+markers", line: { color: COLORS.blue } },
    { x: pts, y: raw.imag.map((r) => r[e]), name: "Imaginary", mode: "lines+markers", line: { color: COLORS.orange } },
    { x: pts, y: mag.map((r) => r[e]), name: "Magnitude", mode: "lines+markers", line: { color: COLORS.green } },
  ], baseLayout("Single echo", "Point within echo", "Signal (arb.)",
     { yaxis: { ...baseLayout("", "", "").yaxis, title: "Signal (arb.)", range: echoRange } }), CFG);

  Plotly.react("plot-decay", [
    { x: t, y: raw.summed, mode: "lines+markers", name: "Summed", line: { color: COLORS.blue }, marker: { size: 4 } },
  ], baseLayout("Summed echo decay (phased real part)", "Echo time (s)", "Signal (arb.)",
     { yaxis: { ...baseLayout("", "", "").yaxis, title: "Signal (arb.)", type: $("decay-log").checked ? "log" : "linear" } }), CFG);
}

export function initRaw() {
  $("echo-idx").addEventListener("input", renderRaw);
  $("decay-log").addEventListener("change", renderRaw);
  $("decay-logx").addEventListener("change", renderRaw);
}
