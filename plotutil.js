// Plot styling shared by the raw-data and results views.
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export const COLORS = {           // Okabe–Ito: colour-blind safe
  blue: "#0072B2", orange: "#D55E00", green: "#009E73", purple: "#CC79A7", grey: "#7a828c",
};
export const CFG = { responsive: true, displaylogo: false };

export function baseLayout(title, xt, yt, extra = {}) {
  const fg = css("--plot-fg"), grid = css("--plot-grid");
  const axis = (t) => ({ title: t, gridcolor: grid, zerolinecolor: grid, automargin: true });
  return {
    title: { text: title, font: { size: 14 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg },
    margin: { l: 60, r: 15, t: 40, b: 50 },
    xaxis: axis(xt),
    yaxis: axis(yt),
    legend: { orientation: "h", y: -0.25 },
    ...extra,
  };
}

// Number formatting: 3 significant figures, exponent form for very large/small values.
export function fmt(x, sig = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (x === 0) return "0";
  const a = Math.abs(x);
  return a >= 1e4 || a < 1e-3 ? x.toExponential(sig - 1) : String(+x.toPrecision(sig));
}
export const fmtCI = (ci) => (ci && ci[0] !== null && ci[1] !== null ? `[${fmt(ci[0])}, ${fmt(ci[1])}]` : "—");
export const pct = (x, d = 1) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%");

// Always scientific, so values and their uncertainties compare at a glance (e.g. 1.98e-2 ± 4.87e-4).
export const fmtSci = (x, sig = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x === 0 ? "0" : x.toExponential(sig - 1));
export const fmtCISci = (ci) => (ci && ci[0] !== null && ci[1] !== null ? `[${fmtSci(ci[0])}, ${fmtSci(ci[1])}]` : "—");

// A value and its uncertainty share one notation: scientific if either needs it, else plain.
export function fmtPair(v, se) {
  const odd = (x) => x !== null && x !== undefined && x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4);
  const f = odd(v) || odd(se) ? fmtSci : fmt;
  return `${f(v)} ± ${f(se)}`;
}
