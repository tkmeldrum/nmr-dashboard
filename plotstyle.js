// Plot settings shared by the on-screen plots and the saved figures. The page applies them as each plot is drawn;
// the same values go to matplotlib when results are saved.
import { $ } from "./state.js";

export const DEFAULTS = { font_size: 10, line_scale: 1, marker_scale: 1, cmap: "viridis", width: "", height: "", legend: true, title: "auto", title_text: "", time_unit: "s" };
const KEY = "nmrdash.plotstyle";
const PLOTLY_CMAP = { viridis: "Viridis", plasma: "Plasma", magma: "Magma", cividis: "Cividis", inferno: "Inferno", turbo: "Turbo", gray: "Greys" };
const FIELDS = { font_size: "ps-font", line_scale: "ps-line", marker_scale: "ps-marker", cmap: "ps-cmap", width: "ps-width", height: "ps-height", title: "ps-title", title_text: "ps-title-text", time_unit: "ps-time" };

let style = { ...DEFAULTS };
export const plotStyle = () => ({ ...style });

// What the saved figures should use (blank size boxes mean "each figure's own size").
export function figureStyle() {
  const num = (v) => (v === "" || v === null || !Number.isFinite(+v) ? null : +v);
  return { ...style, width: num(style.width), height: num(style.height), font_size: +style.font_size || 10, line_scale: +style.line_scale || 1, marker_scale: +style.marker_scale || 1 };
}

function read() {
  const s = { ...DEFAULTS };
  for (const [k, id] of Object.entries(FIELDS)) s[k] = $(id).value;
  s.legend = $("ps-legend").checked;
  return s;
}

function write(s) {
  for (const [k, id] of Object.entries(FIELDS)) $(id).value = s[k];
  $("ps-legend").checked = !!s.legend;
  $("ps-title-text").disabled = s.title !== "custom";
}

// Wrap Plotly.react so every plot picks the settings up as it is drawn. At the defaults nothing is touched.
function patchPlotly() {
  const react = Plotly.react.bind(Plotly);
  Plotly.react = (gd, data, layout, cfg) => react(gd, styleData(data || []), styleLayout(layout || {}), cfg);
}

function styleData(data) {
  const ls = +style.line_scale || 1, ms = +style.marker_scale || 1, cm = PLOTLY_CMAP[style.cmap];
  const same = ls === 1 && ms === 1 && (!cm || cm === "Viridis");
  if (same) return data;
  return data.map((tr) => {
    const t = { ...tr };
    const mode = t.mode || (t.type === "scatter" || !t.type ? "lines" : "");
    if (ls !== 1 && (mode.includes("lines") || t.line)) t.line = { ...(t.line || {}), width: ((t.line && t.line.width) || 2) * ls };
    if (ms !== 1 && (mode.includes("markers") || t.marker)) {
      const sz = t.marker && t.marker.size;
      t.marker = { ...(t.marker || {}), size: Array.isArray(sz) ? sz.map((v) => v * ms) : (sz || 6) * ms };
    }
    if (t.type === "heatmap" && t.colorscale === "Viridis" && cm) t.colorscale = cm;
    return t;
  });
}

function styleLayout(layout) {
  const L = { ...layout }, fs = +style.font_size || 10;
  if (fs !== 10) {
    L.font = { ...(L.font || {}), size: 12 * fs / 10 };
    if (L.title && typeof L.title === "object") L.title = { ...L.title, font: { ...(L.title.font || {}), size: 14 * fs / 10 } };
  }
  if (style.title === "none") L.title = { ...(typeof L.title === "object" ? L.title : {}), text: "" };
  else if (style.title === "custom" && style.title_text) L.title = { ...(typeof L.title === "object" ? L.title : {}), text: style.title_text };
  if (!style.legend) L.showlegend = false;
  return L;
}

export function initPlotStyle(redraw) {
  try { style = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch (_) { style = { ...DEFAULTS }; }
  write(style);
  patchPlotly();
  const changed = () => {
    style = read();
    $("ps-title-text").disabled = style.title !== "custom";
    try { localStorage.setItem(KEY, JSON.stringify(style)); } catch (_) { /* private window: fine */ }
    redraw();
  };
  for (const id of [...Object.values(FIELDS), "ps-legend"]) $(id).addEventListener("change", changed);
  $("ps-reset").addEventListener("click", () => { write(DEFAULTS); changed(); });
}
