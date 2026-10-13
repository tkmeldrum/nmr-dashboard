// The model forms shown next to the fit settings and results (HTML strings; inputs are integers only).

export function meEquation(n, offset) {
  const A = (i) => (n === 1 ? "<i>A</i>" : `<i>A</i><sub>${i}</sub>`);
  const T = (i) => (n === 1 ? "<i>T</i><sub>2</sub>" : `<i>T</i><sub>2,${i}</sub>`);
  const terms = Array.from({ length: n }, (_, k) => `${A(k + 1)}&thinsp;e<sup>−<i>t</i>/${T(k + 1)}</sup>`).join(" + ");
  return `<i>S</i>(<i>t</i>) = ${terms}${offset ? " + <i>c</i>" : ""}`;
}

export function meDefinitions(n) {
  if (n < 2) return "";
  return "Fraction <i>f</i><sub><i>i</i></sub> = <i>A</i><sub><i>i</i></sub> / Σ<i>A</i><sub><i>j</i></sub>;&emsp;" +
    "mean <i>T</i><sub>2</sub> = Σ<i>A</i><sub><i>i</i></sub><i>T</i><sub>2,<i>i</i></sub> / Σ<i>A</i><sub><i>i</i></sub>";
}

export const ILT_EQUATION =
  "<i>S</i>(<i>t</i><sub><i>k</i></sub>) = Σ<sub><i>j</i></sub> <i>f</i>(<i>T</i><sub>2,<i>j</i></sub>)&thinsp;" +
  "e<sup>−<i>t</i><sub><i>k</i></sub>/<i>T</i><sub>2,<i>j</i></sub></sup>,&emsp;<i>f</i> ≥ 0,&emsp;" +
  "minimising ‖<b>K</b><i>f</i> − <i>S</i>‖² + <i>λ</i>²‖<b>D</b><sub>2</sub><i>f</i>‖² &nbsp;(<i>λ</i> set by α)";
