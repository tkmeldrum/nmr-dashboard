// The analysis a dataset gets (CPMG -> exponential decay, T1Sat -> saturation recovery) decides what the page offers.
import { $, state } from "./state.js";
import { meEquation } from "./equations.js";

export const isT1 = () => state.result && state.result.kind === "t1";

// Show only the controls that belong to this kind of data (elements marked data-for="cpmg" / "t1").
export function applyKind(kind) {
  const prev = state.kind;
  state.kind = kind;
  document.querySelectorAll("[data-for]").forEach((el) => { el.hidden = !el.dataset.for.split(" ").includes(kind); });
  const n = $("me-n");
  [...n.options].forEach((o) => { o.disabled = kind === "t1" && +o.value > 2; });      // recovery data: one or two components
  if (kind !== prev) {
    n.value = kind === "t1" ? "1" : "2";                                                // T1: one component is the norm; CPMG: two
    $("me-offset").checked = true;                                                      // recovery curves rarely start at zero
    $("me-compare").checked = true;
  }
  updateEquation();
}

export function updateEquation() {
  $("me-eq-form").innerHTML = meEquation(+$("me-n").value, $("me-offset").checked, state.kind === "t1" ? "saturation" : "decay");
}
