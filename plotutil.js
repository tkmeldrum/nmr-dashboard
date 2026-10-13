// Plot styling shared by the raw-data and results views.
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export const ink = () => css("--plot-fg");     // the page's text colour (follows light / dark mode)

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

// Scientific notation with ONE base exponent for a value, its uncertainty and its interval limits,
// e.g. 1.99e-2 ± 0.35e-2 and [1.31e-2, 2.67e-2]. The uncertainty is shown to 2 significant figures.
export function sciGroup(v, se) {
  const ref = v ? Math.abs(v) : Math.abs(se || 0);
  if (!ref) return { fmt: (x) => fmtSci(x) };
  const decFor = (e) => {
    const m = se > 0 ? se / 10 ** e : null;
    return m ? Math.min(6, Math.max(2, 1 - Math.floor(Math.log10(m)))) : 2;
  };
  let e = Math.floor(Math.log10(ref));
  if (Number((ref / 10 ** e).toFixed(decFor(e))) >= 10) e += 1;      // 9.996 rounds up to 10.00: carry
  const dec = decFor(e);
  return {
    e, dec,
    fmt: (x) => (x === null || x === undefined || Number.isNaN(x) ? "—" : `${(x / 10 ** e).toFixed(dec)}e${e < 0 ? "-" : "+"}${Math.abs(e)}`),
  };
}
export const fmtCIWith = (g, ci) => (ci && ci[0] !== null && ci[1] !== null ? `[${g.fmt(ci[0])}, ${g.fmt(ci[1])}]` : "—");

// "value ± uncertainty" in one notation: shared-exponent scientific if either needs it, else plain.
export function fmtPair(v, se) {
  if (!okUnc(v, se)) return `${fmt(v)} (uncertainty undetermined)`;
  const odd = (x) => x !== null && x !== undefined && x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4);
  if (odd(v) || odd(se)) {
    const g = sciGroup(v, se);
    return `${g.fmt(v)} ± ${g.fmt(se)}`;
  }
  return `${fmt(v)} ± ${fmt(se)}`;
}
export const isOdd = (x) => x !== null && x !== undefined && x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4);

// An uncertainty far larger than its value (or missing) means the parameter is not determined by the data.
export const okUnc = (v, se) => se !== null && se !== undefined && Number.isFinite(se) && (v === 0 || se <= 10 * Math.abs(v));
