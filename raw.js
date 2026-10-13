// Raw-data plots (tab 2).
import { $, state } from "./state.js";
import { baseLayout, CFG, COLORS } from "./plotutil.js";

export function renderRaw() {
  const raw = state.result && state.result.raw;
  if (!raw) return;
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
}
