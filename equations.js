// The model forms shown next to the fit settings and results (HTML strings; inputs are integers only).

export function meEquation(n, offset, kind = "decay") {
  const d = kind === "saturation" ? "1" : "2";                // T₁ for recovery data, T₂ for decays
  const A = (i) => (n === 1 ? "<i>A</i>" : `<i>A</i><sub>${i}</sub>`);
  const T = (i) => (n === 1 ? `<i>T</i><sub>${d}</sub>` : `<i>T</i><sub>${d},${i}</sub>`);
  const term = (k) => (kind === "saturation" ? `${A(k)}&thinsp;(1 − e<sup>−<i>t</i>/${T(k)}</sup>)` : `${A(k)}&thinsp;e<sup>−<i>t</i>/${T(k)}</sup>`);
  const terms = Array.from({ length: n }, (_, k) => term(k + 1)).join(" + ");
  return `<i>S</i>(<i>t</i>) = ${terms}${offset ? " + <i>c</i>" : ""}`;
}

export function meDefinitions(n, kind = "decay") {
  if (n < 2) return "";
  const d = kind === "saturation" ? "1" : "2";
  return "Fraction <i>f</i><sub><i>i</i></sub> = <i>A</i><sub><i>i</i></sub> / Σ<i>A</i><sub><i>j</i></sub>;&emsp;" +
    `mean <i>T</i><sub>${d}</sub> = Σ<i>A</i><sub><i>i</i></sub><i>T</i><sub>${d},<i>i</i></sub> / Σ<i>A</i><sub><i>i</i></sub>`;
}

export const ILT_EQUATION =
  "<i>S</i>(<i>t</i><sub><i>k</i></sub>) = Σ<sub><i>j</i></sub> <i>f</i>(<i>T</i><sub>2,<i>j</i></sub>)&thinsp;" +
  "e<sup>−<i>t</i><sub><i>k</i></sub>/<i>T</i><sub>2,<i>j</i></sub></sup>,&emsp;<i>f</i> ≥ 0,&emsp;" +
  "minimising ‖<b>K</b><i>f</i> − <i>S</i>‖² + <i>λ</i>²‖<b>D</b><sub>2</sub><i>f</i>‖² &nbsp;(<i>λ</i> set by α)";

export const T2D_EQUATION =
  "<i>S</i>(<i>t</i><sub><i>i</i></sub>, <i>b</i><sub><i>j</i></sub>) = Σ<sub><i>k,l</i></sub> <i>f</i>(<i>T</i><sub>2,<i>k</i></sub>, <i>D</i><sub><i>l</i></sub>)&thinsp;" +
  "e<sup>−<i>t</i><sub><i>i</i></sub>/<i>T</i><sub>2,<i>k</i></sub></sup>&thinsp;e<sup>−<i>b</i><sub><i>j</i></sub><i>D</i><sub><i>l</i></sub></sup>,&emsp;<i>f</i> ≥ 0,&emsp;with curvature regularisation set by α";
