// Processing tab: form defaults, validation, and starting a run.
import { $, state } from "./state.js";

const DEFAULT_MAX_STEPS = 100;   // user-editable; Python also caps at the echo count
const STEPS_LIMIT = 1000;        // keep in step with nmrdash/ilt_tools.py

function showMsg(text) {
  const m = $("process-msg");
  m.textContent = text; m.className = "message error"; m.hidden = !text;
}

// Sensible starting values from the echo-time axis.
export function initProcessDefaults(raw) {
  const t = raw.echo_time_s;
  $("ilt-t2min").value = 10 ** Math.floor(Math.log10(t[0]));
  $("ilt-t2max").value = 10 ** Math.ceil(Math.log10(10 * t[t.length - 1]));
  $("ilt-steps").value = DEFAULT_MAX_STEPS;
  const d = state.result.defaults;
  if (d) {
    $("ft-nucleus").value = d.nucleus;           // from b1Freq in acqu.par
    $("ft-G").value = d.G;
    $("ft-zf").value = d.zero_fill;
    $("ft-nucleus-note").textContent = "Nucleus preset from acqu.par: " + d.nucleus_reason + ".";
  }
  $("ft-zmin").value = $("ft-zmax").value = "";
  showMsg("");
  $("run-status").textContent = "";
}

function num(id, label) {
  const v = parseFloat($(id).value);
  if (!Number.isFinite(v)) throw new Error(`${label} must be a number.`);
  return v;
}

export function readSettings() {
  const settings = { echo: document.querySelector('input[name="echo"]:checked').value };
  if (settings.echo === "ft") {
    const G = num("ft-G", "G"), zf = Math.round(num("ft-zf", "Zero-fill"));
    if (G <= 0) throw new Error("G must be positive.");
    if (zf < 0 || zf > 6) throw new Error("Zero-fill must be between 0 and 6.");
    const zmin = $("ft-zmin").value.trim(), zmax = $("ft-zmax").value.trim();
    if ((zmin === "") !== (zmax === "")) throw new Error("Give both ends of the z-window, or leave both blank for automatic.");
    if (zmin !== "" && +zmin >= +zmax) throw new Error("The z-window minimum must be below its maximum.");
    settings.ft = { nucleus: $("ft-nucleus").value, G, zero_fill: zf,
                    z_min: zmin === "" ? null : +zmin, z_max: zmax === "" ? null : +zmax };
  }
  const me = $("me-enabled").checked, il = $("ilt-enabled").checked;
  if (!me && !il) throw new Error("Choose at least one decay analysis.");
  settings.multiexp = {
    enabled: me, n: +$("me-n").value, offset: $("me-offset").checked,
    ci_level: +$("me-ci").value, compare: $("me-compare").checked,
  };
  const nEchoes = state.result.raw.n_echoes;
  if (me && 2 * settings.multiexp.n + settings.multiexp.offset >= nEchoes - 1) {
    throw new Error(`${settings.multiexp.n} components need more echoes than this dataset has (${nEchoes}).`);
  }
  const t2min = num("ilt-t2min", "T₂ min"), t2max = num("ilt-t2max", "T₂ max");
  const steps = Math.round(num("ilt-steps", "Max grid points")), alpha = num("ilt-alpha", "α");
  const thr = num("ilt-peakthr", "Peak threshold"), lo = num("ilt-sweep-lo", "α sweep start"), hi = num("ilt-sweep-hi", "α sweep end");
  if (il) {
    if (t2min <= 0 || t2max <= t2min) throw new Error("T₂ min must be positive and smaller than T₂ max.");
    if (steps < 10 || steps > STEPS_LIMIT) throw new Error(`Max grid points must be between 10 and ${STEPS_LIMIT}.`);
    if (thr < 0 || thr > 100) throw new Error("Peak threshold must be between 0 and 100%.");
    if ($("ilt-sweep").checked && lo >= hi) throw new Error("The α sweep must start below where it ends.");
    if (alpha <= 0) throw new Error("α must be positive.");
  }
  settings.ilt = {
    enabled: il, t2_min: t2min, t2_max: t2max,
    n_steps: steps, alpha, sweep: $("ilt-sweep").checked,
    peak_threshold_pct: thr, sweep_lo: lo, sweep_hi: hi,
  };
  return settings;
}

export function setRunning(on, text = "") {
  state.running = on;
  $("run-btn").disabled = on;
  $("run-status").textContent = text;
}

export function processFailed(text) {
  setRunning(false);
  showMsg("Processing failed: " + text);
}

export function initProcess() {
  for (const [box, fields] of [["me-enabled", "me-fields"], ["ilt-enabled", "ilt-fields"]]) {
    const sync = () => $(fields).querySelectorAll("input,select").forEach((el) => { el.disabled = !$(box).checked; });
    $(box).addEventListener("change", sync);
    sync();
  }
  const syncFt = () => $("ft-fields").querySelectorAll("input,select").forEach((el) => {
    el.disabled = document.querySelector('input[name="echo"]:checked').value !== "ft";
  });
  document.querySelectorAll('input[name="echo"]').forEach((r) => r.addEventListener("change", syncFt));
  syncFt();
  $("process-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.running || !state.result || !state.result.is_cpmg) return;
    let settings;
    try { settings = readSettings(); } catch (err) { return showMsg(err.message); }
    showMsg("");
    setRunning(true, "Running… the 4-component comparison can take several seconds.");
    state.worker.postMessage({ type: "process", settings });
  });
}
