// MCMC and bootstrap: options, confirmation, chunked run with progress and Stop, and the results views.
import { $, state } from "./state.js";
import { rpc } from "./rpc.js";
import { baseLayout, CFG, COLORS, fmt, fmtSci, isOdd, okUnc, pct, sciGroup } from "./plotutil.js";
import { renderResults } from "./results.js";

const run = { stop: false, running: false };
const Z_FOR = { 0.68: 0.994, 0.95: 1.96, 0.99: 2.576 };   // normal quantiles, for least-squares intervals when no t value is at hand

export function resetUncertainty() {
  run.stop = true;                     // a job in flight should wind down quietly
  state.unc = null;
  $("unc-results").hidden = true;
}

export function initUncertainty() {
  for (const [box, fields] of [["mcmc-enabled", "mcmc-fields"], ["boot-enabled", "boot-fields"]]) {
    const sync = () => $(fields).querySelectorAll("input,select").forEach((el) => { el.disabled = !$(box).checked; });
    $(box).addEventListener("change", sync);
    sync();
  }
  $("unc-stop").addEventListener("click", () => { run.stop = true; $("unc-status").textContent = "Stopping after the current chunk…"; });
}

function readOptions() {
  const zRaw = $("unc-z").value.trim();
  return {
    target: document.querySelector('input[name="unc-target"]:checked').value,
    z_um: zRaw === "" ? null : +zRaw,
    level: +$("me-ci").value,
    mcmc: { enabled: $("mcmc-enabled").checked, nwalkers: Math.round(+$("mcmc-walkers").value), steps: Math.round(+$("mcmc-steps").value),
            burn: Math.round(+$("mcmc-burn").value), sample_sigma: $("mcmc-sigma").checked },
    boot: { enabled: $("boot-enabled").checked, replicates: Math.round(+$("boot-n").value), method: $("boot-method").value,
            do_multiexp: $("boot-me").checked, do_ilt: $("boot-ilt").checked },
  };
}

function validate(o) {
  if (o.mcmc.enabled) {
    if (!(o.mcmc.steps >= 100)) throw new Error("MCMC needs at least 100 steps.");
    if (!(o.mcmc.nwalkers >= 8)) throw new Error("MCMC needs at least 8 walkers.");
    if (!(o.mcmc.burn >= 0 && o.mcmc.burn < o.mcmc.steps)) throw new Error("The burn-in must be smaller than the number of steps.");
  }
  if (o.boot.enabled) {
    if (!(o.boot.replicates >= 20)) throw new Error("The bootstrap needs at least 20 replicates.");
    if (!o.boot.do_multiexp && !o.boot.do_ilt) throw new Error("Choose the multi-exponential fit and/or the ILT for the bootstrap.");
  }
  if (o.target === "position" && state.results.mode !== "ft") throw new Error("Choosing a z position needs the FT echo processing.");
}

const duration = (s) => (s < 1.5 ? "a second" : s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);
const msg = (text, kind = "") => { const m = $("unc-message"); m.textContent = text; m.className = "message " + kind; m.hidden = !text; };

function ask(text) {
  return new Promise((resolve) => {
    const d = $("unc-dialog");
    $("unc-dialog-text").textContent = text;
    d.returnValue = "";
    d.addEventListener("close", () => resolve(d.returnValue === "run"), { once: true });
    d.showModal();
  });
}

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ── orchestration ─────────────────────────────────────────────────────────
export async function runUncertainty() {
  const o = readOptions();
  if (!o.mcmc.enabled && !o.boot.enabled) return;
  if (run.running) return;
  const box = $("unc-results");
  box.hidden = false;
  for (const id of ["unc-compare", "mcmc-results", "boot-results"]) $(id).hidden = true;
  msg("");
  state.unc = { mcmc: null, boot: null, level: o.level, targetKind: o.target };
  run.stop = false; run.running = true;
  const bar = $("unc-bar");
  try {
    validate(o);
    if (o.mcmc.enabled && !state.results.multiexp) throw new Error("MCMC needs the multi-exponential fit: enable it on the Process tab and run again.");
    bar.hidden = false; $("unc-stop").hidden = true; $("unc-progress").value = 0;
    $("unc-status").textContent = "Preparing…";

    const jobs = [];
    if (o.mcmc.enabled) {
      $("unc-status").textContent = "Loading the sampler (the first use downloads the small emcee package)…";
      await rpc("install_emcee");
      $("unc-status").textContent = "Setting up the MCMC and timing a short trial…";
      const info = await rpc("mcmc_start", { target: o.target, z_um: o.z_um, nwalkers: o.mcmc.nwalkers, steps: o.mcmc.steps,
                                             burn: o.mcmc.burn, ci_level: o.level, sample_sigma: o.mcmc.sample_sigma });
      jobs.push({ kind: "mcmc", info, total: info.steps, done: 0, est: info.est_seconds });
    }
    if (o.boot.enabled) {
      $("unc-status").textContent = "Setting up the bootstrap and timing a few replicates…";
      const info = await rpc("boot_start", { target: o.target, z_um: o.z_um, replicates: o.boot.replicates,
                                             method: o.boot.method === "auto" ? null : o.boot.method, do_multiexp: o.boot.do_multiexp,
                                             do_ilt: o.boot.do_ilt, ci_level: o.level });
      jobs.push({ kind: "boot", info, total: info.replicates, done: info.done, est: info.est_seconds });
    }
    const target = jobs[0].info.target;
    state.unc.target = target;
    $("unc-target-note").textContent = `Analysing the ${target}.`;

    const est = jobs.reduce((a, j) => a + j.est, 0);
    const what = jobs.map((j) => (j.kind === "mcmc"
      ? `MCMC with ${j.info.nwalkers} walkers × ${j.total} steps`
      : `a bootstrap of ${j.total} replicates (${j.info.method === "noise" ? "measured" : j.info.method} noise)`)).join(" and ");
    const go = await ask(`This will run ${what} on the ${target}. It may take a while: about ${duration(est)} on this computer, ` +
                         `based on a timed trial. The page stays usable, and you can stop at any time and keep what has run so far.`);
    if (!go) { bar.hidden = true; msg("Skipped. Change the options on the Uncertainty tab and run the processing again to retry."); return; }

    $("unc-stop").hidden = false;
    const totalEst = Math.max(est, 1e-6);
    const update = (label) => {
      $("unc-progress").value = jobs.reduce((a, j) => a + (j.done / j.total) * j.est, 0) / totalEst;
      $("unc-status").textContent = label;
    };
    for (const j of jobs) {
      const perUnit = j.est / j.total;
      const [lo, hi] = j.kind === "mcmc" ? [10, 500] : [1, 50];
      const chunk = Math.min(hi, Math.max(lo, Math.round(0.4 / Math.max(perUnit, 1e-4))));   // ≈ 0.4 s per chunk keeps Stop responsive
      while (j.done < j.total && !run.stop) {
        const r = await rpc(j.kind === "mcmc" ? "mcmc_step" : "boot_step", chunk);
        j.done = r.done;
        update(j.kind === "mcmc" ? `MCMC: step ${r.done} of ${r.total}` : `Bootstrap: replicate ${r.done} of ${r.total}`);
      }
      if (j.kind === "mcmc" && j.done < 10) { msg("Stopped before the MCMC had run enough steps to summarise.", "warn"); continue; }
      if (j.kind === "boot" && j.done < 5) { msg("Stopped before enough bootstrap replicates had run to summarise.", "warn"); continue; }
      $("unc-status").textContent = j.kind === "mcmc" ? "Summarising the MCMC (drawing the corner plot)…" : "Summarising the bootstrap…";
      state.unc[j.kind] = await rpc(j.kind === "mcmc" ? "mcmc_finish" : "boot_finish", ...(j.kind === "mcmc" ? [{ fg: css("--plot-fg"), grid: css("--plot-grid") }] : []));
      renderUncertainty();
    }
    bar.hidden = true;
    renderResults();                 // redraw the decay with the posterior-predictive band
  } catch (err) {
    bar.hidden = true;
    msg("Uncertainty analysis failed: " + err.message, "error");
  } finally {
    run.running = false;
  }
}

// ── rendering ─────────────────────────────────────────────────────────────
function table(el, headers, rows) {
  const mk = (tag, text) => Object.assign(document.createElement(tag), { textContent: text });
  const head = document.createElement("tr");
  headers.forEach((h) => head.appendChild(mk("th", h)));
  el.replaceChildren(head, ...rows.map((r) => {
    const tr = document.createElement("tr");
    r.forEach((c) => tr.appendChild(mk("td", c)));
    return tr;
  }));
}

// One notation per row: shared-exponent scientific if any number needs it, otherwise 3 significant figures.
function rowFormat(v, sd, sci = false) {          // sci: T₂ quantities are always scientific, as elsewhere on the page
  if (sci || isOdd(v) || isOdd(sd)) { const g = sciGroup(v, okUnc(v, sd) ? sd : null); return (x) => g.fmt(x); }
  return (x) => fmt(x);
}
const interval = (f, v, lo, hi) => `${f(v)} [${f(lo)}, ${f(hi)}]`;
const warnings = (el, list) => el.replaceChildren(...list.map((text) => Object.assign(document.createElement("p"), { className: "message warn", textContent: text })));
const lvl = (r) => Math.round(r.level * 100);

function lsRows() {
  // least-squares reference values by name, from the fit shown on this page (only valid for the whole-decay target)
  const fit = state.results.multiexp && state.results.multiexp.selected;
  if (!fit || state.unc.targetKind !== "bulk") return {};
  const out = {}, n = fit.n, one = n === 1, z = fit.t_crit;
  fit.components.forEach((c, i) => {
    out[one ? "A" : `A${i + 1}`] = { v: c.A, se: c.A_se };
    out[one ? "T2" : `T2,${i + 1}`] = { v: c.T2, se: c.T2_se };
    out[one ? "fraction" : `fraction ${i + 1}`] = { v: c.fraction, se: c.fraction_se };
  });
  if (fit.offset) out.offset = { v: fit.offset_value, se: fit.offset_se };
  out["total amplitude"] = { v: fit.total_amplitude, se: fit.total_amplitude_se };
  out["mean T2"] = { v: fit.mean_T2, se: fit.mean_T2_se };
  out.__z = z;
  return out;
}

export function renderUncertainty() {
  const u = state.unc;
  if (!u) return;
  if (u.mcmc) renderMcmc(u.mcmc);
  if (u.boot) renderBoot(u.boot);
  renderCompare(u);
}

function renderCompare(u) {
  const names = [], ls = lsRows();
  const add = (n) => { if (!names.includes(n)) names.push(n); };
  (u.mcmc ? [...u.mcmc.params, ...u.mcmc.derived] : []).forEach((p) => add(p.name));
  if (u.boot && u.boot.multiexp) [...u.boot.multiexp.params, ...u.boot.multiexp.derived].forEach((p) => add(p.name));
  $("unc-compare").hidden = !(u.mcmc || (u.boot && u.boot.multiexp));
  if ($("unc-compare").hidden) return;
  const find = (r, n) => (r ? [...r.params, ...r.derived].find((p) => p.name === n) : null);
  const bootMe = u.boot && u.boot.multiexp;
  const rows = names.map((n) => {
    const m = find(u.mcmc, n), b = find(bootMe, n), l = ls[n];
    const ref = m || b;
    const f = rowFormat(ref.median, m ? m.sd : b.sd, /T2/.test(n));
    const z = ls.__z || Z_FOR[u.level] || 1.96;
    const lsCell = !l ? (ref.ls !== undefined && ref.ls !== null ? f(ref.ls) : "—")
      : okUnc(l.v, l.se) ? interval(f, l.v, l.v - z * l.se, l.v + z * l.se) : `${f(l.v)} (uncertainty undetermined)`;
    return [n, lsCell, b ? interval(f, b.median, b.lo, b.hi) : "—", m ? interval(f, m.median, m.lo, m.hi) : "—"];
  });
  table($("unc-compare-table"), ["Parameter", `Least squares (±${(Z_FOR[u.level] || 1.96).toFixed(2)} SE)`, `Bootstrap (${Math.round(u.level * 100)}%)`, `MCMC (${Math.round(u.level * 100)}% credible)`], rows);
}

function renderMcmc(r) {
  $("mcmc-results").hidden = false;
  const d = r.diagnostics;
  $("mcmc-diag").textContent =
    `${r.nwalkers} walkers × ${r.steps_run} steps (burn-in ${r.burn}); acceptance ${fmt(d.acceptance, 2)}; autocorrelation time ${fmt(d.tau_max, 3)} steps ` +
    `→ ${fmt(d.n_eff, 3)} effective samples (chain = ${fmt(d.steps_per_tau, 3)} autocorrelation times). ` +
    (r.sample_sigma ? "The noise σ was sampled." : `The noise σ was fixed at ${fmt(r.sigma)} (imaginary channel).`);
  warnings($("mcmc-warnings"), r.warnings);
  const rows = [...r.params, ...r.derived].map((p) => {
    const f = rowFormat(p.median, p.sd, /T2/.test(p.name));
    return [p.name, p.ls !== undefined && p.ls !== null ? `${f(p.ls)}${p.ls_se !== null && okUnc(p.ls, p.ls_se) ? " ± " + f(p.ls_se) : ""}` : "—",
            f(p.median), `[${f(p.lo)}, ${f(p.hi)}]`, p.sd === null ? "—" : f(p.sd)];
  });
  table($("mcmc-table"), ["Parameter", "Least squares", "Posterior median", `${lvl(r)}% credible interval`, "Posterior SD"], rows);

  // trace plot: one row per parameter, a few walkers each, burn-in shaded
  const k = r.trace.names.length, vals = r.trace.values, steps = r.trace.step, nW = vals[0].length;
  const palette = [COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple, "#56B4E9", "#F0E442", "#999999", "#000000"];
  const traces = [], shapes = [];
  const layout = baseLayout("", "", "", { showlegend: false, margin: { l: 70, r: 15, t: 10, b: 45 } });
  for (let j = 0; j < k; j++) {
    const ax = j === 0 ? "" : String(j + 1);
    for (let w = 0; w < nW; w++) {
      traces.push({ x: steps, y: vals.map((row) => row[w][j]), xaxis: "x" + ax, yaxis: "y" + ax, mode: "lines",
                    line: { width: 0.8, color: palette[w % palette.length] }, opacity: 0.7, hoverinfo: "skip" });
    }
    const top = 1 - j / k, bot = 1 - (j + 1) / k + 0.015;
    layout["yaxis" + ax] = { ...layout.yaxis, title: r.trace.names[j], domain: [bot, top - 0.015], anchor: "x" + ax };
    layout["xaxis" + ax] = { ...layout.xaxis, matches: j === 0 ? undefined : "x", anchor: "y" + ax, showticklabels: j === k - 1, title: j === k - 1 ? "step" : undefined };
    shapes.push({ type: "rect", xref: "x" + ax, yref: `y${ax} domain`, x0: 0, x1: r.trace.burn, y0: 0, y1: 1, fillcolor: COLORS.grey, opacity: 0.2, line: { width: 0 } });
  }
  layout.shapes = shapes;
  $("plot-mcmc-trace").style.height = `${Math.max(16, 6.5 * k)}rem`;
  Plotly.react("plot-mcmc-trace", traces, layout, CFG);
  $("mcmc-corner").src = "data:image/png;base64," + r.corner_png_b64;
}

function renderBoot(r) {
  $("boot-results").hidden = false;
  const methodText = { noise: "the measured imaginary-channel noise, resampled", residual: "the fit residuals, resampled", parametric: "Gaussian noise with σ from the imaginary channel" }[r.method];
  $("boot-diag").textContent = `${r.replicates} replicates using ${methodText}; ${r.failures} failed to fit. Intervals are percentile intervals at ${lvl(r)}%.`;
  warnings($("boot-warnings"), r.warnings);

  const me = r.multiexp;
  $("boot-table").hidden = !me;
  $("boot-hists").hidden = !me;
  if (me) {
    const rows = [...me.params, ...me.derived].map((p) => {
      const f = rowFormat(p.median, p.sd, /T2/.test(p.name));
      return [p.name, p.ls !== undefined && p.ls !== null ? f(p.ls) : "—", f(p.median), `[${f(p.lo)}, ${f(p.hi)}]`, p.sd === null ? "—" : f(p.sd)];
    });
    table($("boot-table"), ["Parameter", "Least squares", "Bootstrap median", `${lvl(r)}% interval`, "Bootstrap SD"], rows);
    const host = $("boot-hists");
    host.replaceChildren();
    me.samples.names.forEach((name, j) => {
      const div = document.createElement("div");
      host.appendChild(div);
      const p = me.params[j];
      const base = baseLayout(name, "", "", { margin: { l: 35, r: 10, t: 30, b: 30 }, showlegend: false });
      Plotly.newPlot(div, [{ type: "histogram", x: me.samples.values.map((row) => row[j]), marker: { color: COLORS.blue }, nbinsx: 25 }], {
        ...base, title: { text: name, font: { size: 12 } },
        shapes: [p.ls, p.lo, p.hi].map((x, i) => ({ type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color: i === 0 ? COLORS.orange : COLORS.grey, dash: i === 0 ? "solid" : "dot", width: 1.5 } })),
      }, CFG);
    });
  }

  const il = r.ilt;
  $("boot-ilt-results").hidden = !il;
  if (il) {
    const base = baseLayout("ILT spectrum with bootstrap band", "T₂ (s)", "Amplitude");
    Plotly.react("plot-boot-ilt", [
      { x: il.tau, y: il.hi, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x: il.tau, y: il.lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(0,158,115,0.25)", name: `${lvl(r)}% band` },
      { x: il.tau, y: il.median, mode: "lines", line: { color: COLORS.green, width: 2 }, name: "bootstrap median" },
      { x: il.tau, y: il.spectrum, mode: "lines", line: { color: COLORS.blue, width: 1.5, dash: "dash" }, name: "ILT of the data" },
    ], { ...base, xaxis: { ...base.xaxis, type: "log", dtick: 1, tickformat: ".0e" } }, CFG);
    table($("boot-peaks"), ["Peak T₂ (s)", "T₂ of the data (s)", `T₂ ${lvl(r)}% interval (s)`, "Fraction of the data", `Fraction ${lvl(r)}% interval`],
      il.bands.map((b) => {
        const f = rowFormat(b.T2.median, b.T2.sd, true);
        return [fmtSci(b.t2_peak), b.base_mean_T2 === null ? "—" : f(b.base_mean_T2), `[${f(b.T2.lo)}, ${f(b.T2.hi)}]`, pct(b.base_fraction), `[${pct(b.fraction.lo)}, ${pct(b.fraction.hi)}]`];
      }));
  }
}
