// Saving the results: into a folder (Chrome / Edge, through the File System Access API) or as one .zip download.
import { $, state } from "./state.js";
import { rpc } from "./rpc.js";
import { withBusy } from "./busy.js";
import { exportOptions } from "./uncertainty.js";
import { figureStyle } from "./plotstyle.js";

const canWrite = typeof window.showDirectoryPicker === "function";

export function bytes(b64) {
  const s = atob(b64), a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  return a;
}

function choices() {
  const u = state.unc || {};
  return {
    csv: $("save-csv").checked, figures: $("save-figs").checked, samples: $("save-samples").checked,
    figure_format: $("save-format").value, dpi: Math.max(50, Math.min(1200, +$("save-dpi").value || 150)),
    style: { ...figureStyle(), decay_logy: $("fit-log").checked },     // what the decay plot on screen shows
    mcmc: !!u.mcmc, boot: !!u.boot, options: exportOptions(),
    page: { dataset: $("folder-name").textContent, uncertainty: [u.mcmc && "mcmc", u.boot && "bootstrap"].filter(Boolean),
            browser: { user_agent: navigator.userAgent } },
  };
}

function say(text, bad = false) {
  const el = $("save-status");
  el.textContent = text;
  el.classList.toggle("error-text", bad);
}

export async function writeInto(dir, folder, files) {
  const sub = await dir.getDirectoryHandle(folder, { create: true });
  for (const [name, b64] of Object.entries(files)) {
    const w = await (await sub.getFileHandle(name, { create: true })).createWritable();
    await w.write(bytes(b64));
    await w.close();
  }
}

async function saveToFolder(dir) {
  const out = await withBusy("Making the figures and tables…", () => rpc("save_bundle", choices()));
  await writeInto(dir, out.folder, out.files);
  say(`Saved ${Object.keys(out.files).length} files to ${dir.name}/${out.folder}/`);
}

async function download() {
  const out = await withBusy("Making the figures and tables…", () => rpc("save_bundle", { ...choices(), zip: true }));
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([bytes(out.zip_b64)], { type: "application/zip" })), download: out.zip_name });
  document.body.append(a);
  a.click();
  a.remove();
  say(`Downloaded ${out.zip_name} (${out.files.length} files inside one folder). Unzip it next to your data.`);
}

async function guarded(fn) {
  if (!state.results) return say("Run the processing first.", true);
  try { await fn(); } catch (err) { if (err.name !== "AbortError") say("Could not save: " + (err.message || err), true); }
}

export function initSave() {
  const hasFolder = () => canWrite && !!state.dataDir;
  const sync = () => { $("save-folder").hidden = !hasFolder(); $("save-other").hidden = !canWrite; };
  sync();
  document.addEventListener("datasetchange", sync);
  $("save-folder").addEventListener("click", () => guarded(async () => {
    // asked first, while the click still counts: the permission prompt needs it
    if ((await state.dataDir.requestPermission({ mode: "readwrite" })) !== "granted") return say("Writing to that folder was not allowed.", true);
    await saveToFolder(state.dataDir);
  }));
  $("save-other").addEventListener("click", () => guarded(async () => saveToFolder(await window.showDirectoryPicker({ mode: "readwrite" }))));
  $("save-zip").addEventListener("click", () => guarded(download));
  $("save-format").addEventListener("change", () => { $("save-dpi-row").hidden = $("save-format").value !== "png"; });
  $("save-hint").textContent = canWrite
    ? "The files go into a new processed_… folder, so your raw data is never touched."
    : "This browser cannot write into folders, so results come as one .zip to unzip next to your data. (Chrome and Edge can save straight into the folder.)";
}
