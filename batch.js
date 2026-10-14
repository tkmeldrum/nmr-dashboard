// Batch mode: the settings on the Process tab, applied to every dataset found in the folder, one after another.
import { $, state } from "./state.js";
import { rpc } from "./rpc.js";
import { readSettings } from "./process.js";
import { exportOptions, duration } from "./uncertainty.js";
import { figureStyle } from "./plotstyle.js";
import { bytes, writeInto } from "./save.js";
import { makeZip } from "./zip.js";

const run = { active: false, stop: false };
let reopen = async () => {};

function confirm(title, text, button) {
  return new Promise((resolve) => {
    const d = $("batch-dialog");
    $("batch-dialog-title").textContent = title;
    $("batch-dialog-text").textContent = text;
    $("batch-dialog-run").textContent = button;
    d.returnValue = "";
    d.addEventListener("close", () => resolve(d.returnValue === "run"), { once: true });
    d.showModal();
  });
}

const count = () => (state.datasets || []).length;

export function refreshBatch() {
  const n = count();
  $("batch-banner").hidden = n < 2;
  if (n >= 2) {
    $("batch-banner-text").textContent = `${n} datasets were found in this folder and are shown one at a time. To process all of them with the same settings, ` +
      "first try your settings on a single dataset (choose it above, then run it on the Process tab), then use Batch at the bottom of the Process tab.";
  }
  $("batch-card").hidden = n < 2 || !(state.result && state.result.supported);
  if ($("batch-card").hidden) return;
  let now = null;
  try { now = JSON.stringify(readSettings()); } catch (_) { /* invalid settings: the run will say so */ }
  const last = state.lastRun;
  $("batch-readiness").textContent = !last
    ? "You have not run these settings on a single dataset yet. Run one first (Run processing, above) and check the results before starting the batch."
    : last.key === now
      ? `✓ These exact settings were tried on ${last.label} (${duration(last.seconds)}).`
      : `The settings have changed since your last single run (on ${last.label}). Run one dataset again to check them before the batch.`;
}

function log(text, cls = "") {
  const li = Object.assign(document.createElement("li"), { textContent: text, className: cls });
  $("batch-log").append(li);
  li.scrollIntoView({ block: "nearest" });
}

async function runBatch() {
  if (run.active || state.running) return;
  const msg = $("process-msg");
  let settings;
  try { settings = readSettings(); } catch (err) { msg.textContent = err.message; msg.className = "message error"; msg.hidden = false; return; }
  msg.hidden = true;
  const sets = state.datasets, n = sets.length, kind = state.result.kind, options = exportOptions();
  const last = state.lastRun, fresh = last && last.key === JSON.stringify(settings);
  const folderMode = typeof window.showDirectoryPicker === "function" && !!state.dirHandle && sets.every((d) => d.dir.handle);
  const unc = options.mcmc.enabled || options.bootstrap.enabled;
  const text = [`${n} datasets (those of the same kind as ${$("folder-name").textContent} are processed; others are skipped).`,
    last ? `Your single run took ${duration(last.seconds)}, so roughly ${duration(last.seconds * n)} in all${unc ? ", plus the MCMC / bootstrap time for each dataset" : ""}.`
         : "I cannot estimate the time: nothing has been run on a single dataset yet.",
    fresh ? "These settings were checked on a single dataset." : "WARNING: these settings have not been checked on a single dataset. Consider cancelling and running one first.",
    folderMode ? "Results go into a new processed_… folder inside each dataset's folder, and a batch_… summary folder next to them." : "Results are collected into one .zip to download at the end.",
    "You can stop between datasets; the ones already done are kept."].join("\n\n");
  if (!(await confirm("Run the batch?", text, "Run batch"))) return;

  run.active = true; run.stop = false;
  $("batch-log").replaceChildren();
  for (const id of ["batch-run", "run-btn"]) $(id).disabled = true;
  $("batch-stop").hidden = false; $("batch-stop").disabled = false;
  const bar = $("batch-progress"), status = $("batch-status");
  bar.hidden = false; bar.max = n; bar.value = 0;
  const zipEntries = [], counts = { ok: 0, skipped: 0, failed: 0 };
  const t0 = performance.now();
  try {
    if (folderMode && (await state.dirHandle.requestPermission({ mode: "readwrite" })) !== "granted") throw new Error("Writing to that folder was not allowed; allow it, or use a browser that cannot write folders to get a .zip instead.");
    if (unc) { status.textContent = "Fetching the emcee package…"; await rpc("install_emcee"); }
    const output = { csv: $("batch-csv").checked, figures: $("batch-figs").checked, samples: true, figure_format: $("save-format").value, dpi: +$("save-dpi").value || 150 };
    await rpc("batch_start", { settings, options, style: { ...figureStyle(), decay_logy: $("fit-log").checked }, kind, output, root_name: state.dirHandle ? state.dirHandle.name : "datasets" });
    for (let i = 0; i < n; i++) {
      if (run.stop) { log("Stopped by you; the datasets above are done.", "bad"); break; }
      const ds = sets[i];
      status.textContent = `Dataset ${i + 1} of ${n}: ${ds.label}…`;
      let out;
      try {
        const { files, timestampMs } = await state.readDatasetFiles(ds);
        await rpc("stage_dataset", i, files);
        out = await rpc("batch_run_one", { i, label: ds.label, timestamp_ms: timestampMs });
      } catch (err) {
        out = { status: "failed", reason: err.message, files: {}, folder: null };
      }
      counts[out.status] = (counts[out.status] || 0) + 1;
      if (out.status === "ok") log(`✓ ${ds.label} (${duration(out.seconds)})${out.warnings && out.warnings.length ? " — " + out.warnings[0] : ""}`);
      else log(`${out.status === "skipped" ? "–" : "✗"} ${ds.label}: ${out.status}${out.reason ? " — " + out.reason : ""}`, out.status === "failed" ? "bad" : "");
      if (out.folder) {
        if (folderMode) await writeInto(ds.dir.handle, out.folder, out.files);
        else for (const [name, b64] of Object.entries(out.files)) zipEntries.push({ path: `${ds.label}/${out.folder}/${name}`, bytes: bytes(b64) });
      }
      bar.value = i + 1;
    }
    status.textContent = "Writing the batch summary…";
    const fin = await rpc("batch_finish", {});
    if (folderMode) await writeInto(state.dirHandle, fin.folder, fin.files);
    else for (const [name, b64] of Object.entries(fin.files)) zipEntries.push({ path: `${fin.folder}/${name}`, bytes: bytes(b64) });
    if (!folderMode) {
      const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(makeZip(zipEntries)), download: `${fin.folder}.zip` });
      document.body.append(a); a.click(); a.remove();
    }
    status.textContent = `Done in ${duration((performance.now() - t0) / 1000)}: ${counts.ok} processed, ${counts.skipped} skipped, ${counts.failed} failed. ` +
      (folderMode ? `Results are in each dataset's folder and in ${fin.folder}/.` : `Downloaded ${fin.folder}.zip: unzip it next to your data.`);
  } catch (err) {
    status.textContent = "The batch stopped: " + err.message;
    log("✗ " + err.message, "bad");
  } finally {
    run.active = false;
    bar.hidden = true;
    $("batch-stop").hidden = true;
    for (const id of ["batch-run", "run-btn"]) $(id).disabled = false;
    await reopen(state.datasetIndex || 0);          // the batch used Python for its own datasets: bring back the one on screen
  }
}

export function initBatch(reopenDataset) {
  reopen = reopenDataset;
  $("batch-run").addEventListener("click", runBatch);
  $("batch-stop").addEventListener("click", () => { run.stop = true; $("batch-stop").disabled = true; $("batch-status").textContent = "Stopping after this dataset…"; });
  $("batch-export").addEventListener("click", () => $("export-btn").click());
  document.addEventListener("datasetchange", refreshBatch);
  for (const id of ["process-form"]) $(id).addEventListener("change", refreshBatch);
}
