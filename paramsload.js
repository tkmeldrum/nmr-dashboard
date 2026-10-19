// Fill the Process controls (including Advanced settings) from a params file (the same file the command-line script reads).
import { $, state } from "./state.js";
import { rpc } from "./rpc.js";
import { withBusy } from "./busy.js";
import { setStyle } from "./plotstyle.js";

function set(id, value) {
  const el = $(id);
  if (!el) return;
  if (el.type === "checkbox") el.checked = !!value;
  else el.value = value === null || value === undefined ? "" : value;
  el.dispatchEvent(new Event("change", { bubbles: true }));       // dependent controls and the equation follow
}
const pick = (name, value) => { const el = document.querySelector(`input[name="${name}"][value="${value}"]`); if (el) { el.checked = true; el.dispatchEvent(new Event("change", { bubbles: true })); } };
const log10 = (x) => +Math.log10(x).toFixed(3);
// A file that turns an advanced option on (or off its default) opens the Advanced block that holds it, so nothing is hidden.
const showAdvanced = (id) => { const d = $(id) && $(id).closest("details.adv, details.advanced-settings"); if (d) d.open = true; };

function applyDecay(s) {
  set("me-n", s.multiexp.n); set("me-offset", s.multiexp.offset); set("me-compare", s.multiexp.compare);
  const ci = String(s.multiexp.ci_level);
  const option = [...$("me-ci").options].find((o) => +o.value === +ci);
  if (option) set("me-ci", option.value);
  if (!s.multiexp.compare) showAdvanced("me-compare");
  if (option && +option.value !== 0.95) showAdvanced("me-ci");
  return option ? null : `The confidence level ${ci} is not one of the page's choices (68, 95, 99%); it was left as it was.`;
}

async function apply(out) {
  const s = out.settings, o = out.options || {}, notes = [...(out.notes || [])];
  set("exclude-first", s.exclude_first || 0);
  if (out.kind === "t2d") {
    const t = s.t2d;
    set("t2d-gradient", t.gradient);
    set("t2d-t2min", log10(t.t2_min)); set("t2d-t2max", log10(t.t2_max)); set("t2d-t2steps", t.t2_steps); set("t2d-dsteps", Math.min(t.d_steps, state.result.raw.n_tau));
    let dmin = t.d_min, dmax = t.d_max;
    if (dmin === "auto" || dmax === "auto") {
      const ax = await rpc("t2d_axes", { t2d: { gradient: t.gradient } });
      [dmin, dmax] = ax.d_window.map((v) => +v.toPrecision(2));
    }
    set("t2d-dmin", +Math.log10(dmin).toFixed(3)); set("t2d-dmax", +Math.log10(dmax).toFixed(3)); set("t2d-alpha", log10(t.alpha)); set("t2d-peakthr", t.peak_threshold_pct);
    const tm = s.t2d_model || {};
    set("t2d-model-enabled", tm.enabled !== false); set("t2d-model-k", tm.k ? String(tm.k) : "auto"); set("t2d-model-window", tm.d_window || "resolvable");
  } else {
    const warn = applyDecay(s);
    if (warn) notes.push(warn);
    if (out.kind === "cpmg") {
      pick("echo", s.echo);
      if (s.ft) {
        if (s.ft.nucleus) set("ft-nucleus", s.ft.nucleus);
        set("ft-G", s.ft.G); set("ft-zf", s.ft.zero_fill); set("ft-zmin", s.ft.z_min); set("ft-zmax", s.ft.z_max);
        set("ft-spacing", s.ft.spacing_um); set("ft-ilt-spacing", s.ft.ilt_spacing_um); set("ft-minsig", s.ft.min_signal_pct);
      }
      set("me-enabled", s.multiexp.enabled);
      const il = s.ilt;
      set("ilt-enabled", il.enabled); set("ilt-t2min", log10(il.t2_min)); set("ilt-t2max", log10(il.t2_max)); set("ilt-steps", il.n_steps);
      set("ilt-alpha", log10(il.alpha)); set("ilt-sweep", il.sweep); set("ilt-peakthr", il.peak_threshold_pct);
      set("ilt-sweep-lo", il.sweep_lo); set("ilt-sweep-hi", il.sweep_hi);
      if (il.sweep) showAdvanced("ilt-sweep");
    }
  }
  if (o.mcmc) {
    set("mcmc-enabled", o.mcmc.enabled); set("mcmc-walkers", o.mcmc.nwalkers); set("mcmc-steps", o.mcmc.steps); set("mcmc-burn", o.mcmc.burn);
    set("mcmc-sigma", o.mcmc.sample_sigma);
    if (o.mcmc.enabled || o.bootstrap.enabled) showAdvanced("mcmc-enabled");
    if (out.kind === "cpmg") { pick("unc-target", o.mcmc.target); set("unc-z", o.mcmc.z_um); if (o.mcmc.target === "position") showAdvanced("unc-z"); }
    const b = o.bootstrap;
    set("boot-enabled", b.enabled); set("boot-n", b.replicates); set("boot-method", b.method || "auto"); set("boot-me", b.do_multiexp);
    if (out.kind === "cpmg") set("boot-ilt", b.do_ilt);
  }
  set("fit-log", out.style.decay_logy);
  setStyle(out.style);
  return notes;
}

export function initParamsLoad() {
  $("params-btn").addEventListener("click", () => $("params-file").click());
  $("params-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const msg = $("process-msg");
    try {
      if (!state.result || !state.result.supported) throw new Error("Load a dataset first.");
      const out = await withBusy("Reading the params file…", async () => rpc("params_for_page", await file.text(), file.name));
      const notes = await apply(out);
      msg.hidden = true;
      $("run-status").textContent = `Settings filled in from ${file.name}.` + (notes.length ? " " + notes.join(" ") : "") + " Check them, then Run processing.";
    } catch (err) {
      msg.textContent = `Could not use ${file.name}: ${err.message}`;
      msg.className = "message error";
      msg.hidden = false;
    }
  });
}
