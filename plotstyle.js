// Plot settings shared by the on-screen plots and the saved figures. The page applies them as each plot is drawn;
// the same values go to matplotlib when results are saved.
import { $ } from "./state.js";

export const DEFAULTS = { font_size: 10, line_scale: 1, marker_scale: 1, cmap: "viridis", width: "", height: "", legend: true, title: "auto", title_text: "", time_unit: "s" };
const KEY = "nmrdash.plotstyle";
const PLOTLY_CMAP = { viridis: "Viridis", plasma: "Plasma", magma: "Magma", cividis: "Cividis", inferno: "Inferno", turbo: "Turbo", gray: "Greys" };
const FIELDS = { font_size: "ps-font", line_scale: "ps-line", marker_scale: "ps-marker", cmap: "ps-cmap", width: "ps-width", height: "ps-height", title: "ps-title", title_text: "ps-title-text", time_unit: "ps-time" };

let style = { ...DEFAULTS, figures: {} };
const FIG_FIELDS = { title: "ps-fig-title", xmin: "ps-fig-xmin", xmax: "ps-fig-xmax", ymin: "ps-fig-ymin", ymax: "ps-fig-ymax", xscale: "ps-fig-xscale", yscale: "ps-fig-yscale" };
export const plotStyle = () => ({ ...style });

// What the saved figures should use (blank size boxes mean "each figure's own size").
export function figureStyle() {
  const num = (v) => (v === "" || v === null || !Number.isFinite(+v) ? null : +v);
  const figures = {};
  for (const [name, o] of Object.entries(style.figures || {})) {
    figures[name] = { title: o.title || "", xlim: [num(o.xmin), num(o.xmax)], ylim: [num(o.ymin), num(o.ymax)], xscale: o.xscale || null, yscale: o.yscale || null };
  }
  return { ...style, figures, width: num(style.width), height: num(style.height), font_size: +style.font_size || 10, line_scale: +style.line_scale || 1, marker_scale: +style.marker_scale || 1 };
}

function read() {
  const s = { ...DEFAULTS, figures: style.figures || {} };
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

function showFigure() {
  const o = (style.figures || {})[$("ps-fig").value] || {};
  for (const [k, id] of Object.entries(FIG_FIELDS)) $(id).value = o[k] ?? "";
}

// Take plot settings from outside (a params file): the keys of figures.style_from, with figures[name] = {title, xlim, ylim, xscale, yscale}.
let applyOutside = () => {};
export const setStyle = (s) => applyOutside(s);

export function initPlotStyle(redraw) {
  try { style = { ...DEFAULTS, figures: {}, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch (_) { style = { ...DEFAULTS, figures: {} }; }
  write(style);
  showFigure();
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(style)); } catch (_) { /* private window: fine */ } };
  $("ps-fig").addEventListener("change", showFigure);
  for (const id of Object.values(FIG_FIELDS)) {            // saved figures only: nothing on the page to redraw
    $(id).addEventListener("change", () => {
      const o = {};
      for (const [k, i] of Object.entries(FIG_FIELDS)) o[k] = $(i).value;
      style.figures = { ...(style.figures || {}), [$("ps-fig").value]: o };
      save();
    });
  }
  patchPlotly();
  const changed = () => {
    style = read();
    $("ps-title-text").disabled = style.title !== "custom";
    save();
    redraw();
  };
  for (const id of [...Object.values(FIELDS), "ps-legend"]) $(id).addEventListener("change", changed);
  applyOutside = (s) => {
    const figures = {};
    for (const [name, o] of Object.entries(s.figures || {})) {
      figures[name] = { title: o.title || "", xmin: o.xlim[0] ?? "", xmax: o.xlim[1] ?? "", ymin: o.ylim[0] ?? "", ymax: o.ylim[1] ?? "", xscale: o.xscale || "", yscale: o.yscale || "" };
    }
    style = { ...DEFAULTS, ...s, width: s.width ?? "", height: s.height ?? "", figures };
    write(style);
    showFigure();
    changed();
  };
  $("ps-reset").addEventListener("click", () => { style.figures = {}; write(DEFAULTS); showFigure(); changed(); });
}
