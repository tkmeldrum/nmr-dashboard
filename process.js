// Processing tab: form defaults, validation, and starting a run.
import { $, state } from "./state.js";
import { ILT_EQUATION, T2D_EQUATION } from "./equations.js";
import { applyKind, updateEquation } from "./kinds.js";
import { rpc } from "./rpc.js";
import { setBusy, setStoppable } from "./busy.js";
import { download } from "./provenance.js";
import { ask, duration, exportOptions } from "./uncertainty.js";
import { figureStyle } from "./plotstyle.js";

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
  const zw = d && d.z_window_um;                                          // default window: −200 to +200 µm
  $("ft-zmin").value = zw ? zw[0] : "";
  $("ft-zmax").value = zw ? zw[1] : "";
  $("ft-spacing").value = $("ft-ilt-spacing").value = "";
  $("ft-minsig").value = 5;
  showMsg("");
  $("run-status").textContent = "";
  $("exclude-first").value = 0;
}

export function initT2dDefaults(r) {
  const d = r.defaults;
  $("t2d-t2min").value = d.t2_min; $("t2d-t2max").value = d.t2_max; $("t2d-t2steps").value = d.t2_steps;
  $("t2d-dmin").value = d.d_min; $("t2d-dmax").value = d.d_max; $("t2d-dsteps").value = Math.min(d.d_steps, r.raw.n_tau);
  $("t2d-alpha").value = +Math.log10(d.alpha).toFixed(3); $("t2d-peakthr").value = d.peak_threshold_pct;
  $("t2d-equation").innerHTML = T2D_EQUATION;
  $("t2d-gradient").value = d.gradient;
  refreshT2dAxes();
  showMsg("");
  $("run-status").textContent = "";
}

// The b-values follow from the gradient setting (calculated in Python, physics.py): the hint under the T2-D fields and the "resolvable range" button use them.
export async function refreshT2dAxes() {
  if (!state.result || state.result.kind !== "t2d") return null;
  const g = parseFloat($("t2d-gradient").value);
  if (!(g > 0)) { $("t2d-window-hint").textContent = "Enter the gradient G (T/m) to calculate the b-values."; return null; }
  const ax = await rpc("t2d_axes", { t2d: { gradient: g } });
  const b = ax.b_vec.map((x) => x * 1e9), [lo, hi] = ax.d_window;
  const kea = (ax.b_check.compared.find((c) => c.kind === "shape" && c.implied_gradient_t_per_m) || {}).implied_gradient_t_per_m;
  state.t2dAxes = ax;
  $("t2d-window-hint").textContent = `With G = ${g} T/m, Δ and τ from acqu.par, your b-values (${Math.min(...b).toExponential(2)} – ${Math.max(...b).toExponential(2)} s/m²) can tell apart D from about ${lo.toExponential(1)} to ${hi.toExponential(1)} m²/s. ` +
    "The range above is yours to choose, but outside this one the solution is regularisation, not data." +
    " To probe diffusion coefficients above this range (faster diffusion), decrease b by decreasing the gradient encoding time τ or the diffusion time Δ; to probe below it (slower diffusion), increase b by increasing τ or Δ." +
    (kea ? ` (For reference, the spectrometer's own b-values correspond to G = ${kea.toFixed(2)} T/m.)` : "");
  return ax;
}

function num(id, label) {
  const v = parseFloat($(id).value);
  if (!Number.isFinite(v)) throw new Error(`${label} must be a number.`);
  return v;
}

function excludeFirst() {
  const n = state.result.kind === "t1" ? state.result.raw.n_points : state.result.raw.n_echoes;
  const v = Math.round(num("exclude-first", "The number of excluded points"));
  if (v < 0) throw new Error("The number of excluded points cannot be negative.");
  if (v && n - v < 8) throw new Error(`Excluding ${v} of ${n} points leaves only ${n - v}; at least 8 are needed.`);
  return v;
}

export function readSettings() {
  if (state.result.kind === "t2d") {
    const t = {
      t2_min: num("t2d-t2min", "T₂ min"), t2_max: num("t2d-t2max", "T₂ max"), t2_steps: Math.round(num("t2d-t2steps", "T₂ points")),
      gradient: num("t2d-gradient", "Gradient G"), d_min: num("t2d-dmin", "D min"), d_max: num("t2d-dmax", "D max"), d_steps: Math.round(num("t2d-dsteps", "D points")),
      alpha: 10 ** num("t2d-alpha", "log10 α"), peak_threshold_pct: num("t2d-peakthr", "Region threshold"),
    };
    if (!(t.gradient > 0)) throw new Error("The gradient G must be a positive number (T/m).");
    if (t.t2_min <= 0 || t.t2_max <= t.t2_min) throw new Error("T₂ min must be positive and smaller than T₂ max.");
    if (t.d_min <= 0 || t.d_max <= t.d_min) throw new Error("D min must be positive and smaller than D max.");
    if (t.t2_steps < 10 || t.t2_steps > 400) throw new Error("T₂ points must be between 10 and 400.");
    if (t.d_steps < 5 || t.d_steps > 200) throw new Error("D points must be between 5 and 200.");
    if (t.alpha <= 0) throw new Error("α must be a finite positive number.");
    if (t.peak_threshold_pct < 0 || t.peak_threshold_pct > 100) throw new Error("The region threshold must be between 0 and 100%.");
    const k = $("t2d-model-k").value;
    return { kind: "t2d", exclude_first: excludeFirst(), t2d: t,
             t2d_model: { enabled: $("t2d-model-enabled").checked, k: k === "auto" ? null : +k, d_window: $("t2d-model-window").value } };
  }
  if (state.result.kind === "t1") {                           // saturation recovery: one or two components, no echo processing, no ILT
    const n = +$("me-n").value, offset = $("me-offset").checked;
    if (2 * n + offset + 2 > state.result.raw.n_points) throw new Error(`${n} component${n > 1 ? "s" : ""} need more delays than this dataset has (${state.result.raw.n_points}).`);
    return { kind: "t1", exclude_first: excludeFirst(), multiexp: { enabled: true, n, offset, ci_level: +$("me-ci").value, compare: $("me-compare").checked } };
  }
  const settings = { exclude_first: excludeFirst(), echo: document.querySelector('input[name="echo"]:checked').value };
  if (settings.echo === "ft") {
    const G = num("ft-G", "G"), zf = Math.round(num("ft-zf", "Zero-fill"));
    if (G <= 0) throw new Error("G must be positive.");
    if (zf < 0 || zf > 6) throw new Error("Zero-fill must be between 0 and 6.");
    const zmin = $("ft-zmin").value.trim(), zmax = $("ft-zmax").value.trim();
    if ((zmin === "") !== (zmax === "")) throw new Error("Give both ends of the z-window, or leave both blank for automatic.");
    if (zmin !== "" && +zmin >= +zmax) throw new Error("The z-window minimum must be below its maximum.");
    const sp = $("ft-spacing").value.trim();
    if (sp !== "" && !(+sp > 0)) throw new Error("The z spacing must be positive (or blank for automatic).");
    const minsig = num("ft-minsig", "Min signal");
    if (minsig < 0 || minsig > 100) throw new Error("Min signal must be between 0 and 100%.");
    const isp = $("ft-ilt-spacing").value.trim();
    if (isp !== "" && !(+isp > 0)) throw new Error("The ILT map spacing must be positive (or blank for every zero-filled point).");
    settings.ft = { nucleus: $("ft-nucleus").value, G, zero_fill: zf,
                    z_min: zmin === "" ? null : +zmin, z_max: zmax === "" ? null : +zmax,
                    spacing_um: sp === "" ? null : +sp, ilt_spacing_um: isp === "" ? null : +isp,
                    min_signal_pct: minsig };
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
  const steps = Math.round(num("ilt-steps", "Max grid points")), alpha = 10 ** num("ilt-alpha", "log10 α");
  const thr = num("ilt-peakthr", "Peak threshold"), lo = num("ilt-sweep-lo", "α sweep start"), hi = num("ilt-sweep-hi", "α sweep end");
  if (il) {
    if (t2min <= 0 || t2max <= t2min) throw new Error("T₂ min must be positive and smaller than T₂ max.");
    if (steps < 10 || steps > STEPS_LIMIT) throw new Error(`Max grid points must be between 10 and ${STEPS_LIMIT}.`);
    if (thr < 0 || thr > 100) throw new Error("Peak threshold must be between 0 and 100%.");
    if ($("ilt-sweep").checked && lo >= hi) throw new Error("The α sweep must start below where it ends.");
    if (alpha <= 0) throw new Error("α must be a finite positive number.");
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
  if (on) state.runT0 = performance.now();
  else if (text === "Done." && state.runT0) {                 // a finished single run: what Batch compares the current settings with
    state.lastRun = { label: $("folder-name").textContent, seconds: (performance.now() - state.runT0) / 1000, key: state.runKey };
  }
  $("run-btn").disabled = on;
  $("run-status").textContent = text;
  $("run-progress").hidden = true;
}

export function setProgress(done, total) {
  const bar = $("run-progress");
  bar.hidden = false; bar.max = total; bar.value = done;
  $("run-status").textContent = `Fitting z position ${done} of ${total}…`;
}

export function processFailed(text) {
  setBusy(null);
  setRunning(false);
  showMsg("Processing failed: " + text);
}

export function initProcess() {
  $("ilt-eq-form").innerHTML = ILT_EQUATION;
  applyKind("cpmg");
  $("me-n").addEventListener("change", updateEquation);
  $("me-offset").addEventListener("change", updateEquation);
  updateEquation();
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
  $("export-btn").addEventListener("click", async () => {
    try {
      showMsg("");
      const settings = readSettings();
      const toml = await rpc("export_params", settings, exportOptions(), $("folder-name").textContent, { ...figureStyle(), decay_logy: $("fit-log").checked });
      download("params.toml", toml);
      download("process_nmr.py", await rpc("export_script"));
      $("run-status").textContent = "Downloaded params.toml and process_nmr.py (your browser may ask to allow two downloads).";
    } catch (err) {
      showMsg("Could not export: " + err.message);
    }
  });
  $("process-form").addEventListener("submit", (e) => e.preventDefault());   // Enter in a field must not reload the page
  $("t2d-use-window").addEventListener("click", async () => {
    const ax = await refreshT2dAxes();
    if (!ax) return;
    const two = (v) => +v.toPrecision(2);
    $("t2d-dmin").value = two(ax.d_window[0]); $("t2d-dmax").value = two(ax.d_window[1]);
  });
  $("t2d-gradient").addEventListener("change", () => refreshT2dAxes().catch((err) => showMsg(err.message)));
  $("run-btn").addEventListener("click", async () => {
    if (state.running || !state.result || !state.result.supported) return;
    let settings;
    try { settings = readSettings(); } catch (err) { return showMsg(err.message); }
    showMsg("");
    state.runKey = JSON.stringify(settings);
    state.runOptions = exportOptions();            // what this run used, for the provenance record
    let busyText = "Processing… the 4-component comparison and per-position fits can take a while.", stoppable = false;
    if (settings.kind === "t2d") {                  // one big dense solve with no progress to report: estimate it first, offer to stop it
      setRunning(true, "Estimating how long the inversion will take…");
      try {
        const est = await rpc("t2d_estimate", settings);
        if (est.seconds > 20 && !(await ask(`The ${est.n_T2} × ${est.n_D} T₂–D inversion (${est.unknowns} unknowns) is estimated to take about ${duration(est.seconds)} on this computer, ` +
            "based on a short timed trial. The page cannot show progress during the solve, but you can stop it (this restarts Python and keeps your data and settings). Run it?"))) {
          setRunning(false, "");
          return;
        }
        busyText = `Solving the ${est.n_T2} × ${est.n_D} T₂–D inversion (about ${duration(est.seconds)})…`;
        stoppable = true;
      } catch (err) {
        return processFailed(err.message);
      }
    }
    setBusy(busyText);
    setStoppable(stoppable);
    setRunning(true, stoppable ? "Solving… use Stop in the grey bar above to cancel." : "Running… the 4-component comparison can take several seconds.");
    state.worker.postMessage({ type: "process", settings });
  });
}
