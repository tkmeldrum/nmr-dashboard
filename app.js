// Main thread: worker plumbing, folder loading, tab flow.
import { $, state } from "./state.js";
import { initTabs, setTabEnabled, showTab } from "./tabs.js";
import { initRaw, renderRaw } from "./raw.js";
import { initProcess, initProcessDefaults, processFailed, setProgress, setRunning } from "./process.js";
import { initResults, renderResults } from "./results.js";

// ── worker ────────────────────────────────────────────────────────────────
const worker = new Worker("worker.js");
state.worker = worker;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") $("status").textContent = m.text;
  else if (m.type === "ready") {
    state.ready = true;
    $("status").textContent = "Python ready.";
    if (state.pending) { sendLoad(state.pending); state.pending = null; }
  } else if (m.type === "loaded") {
    state.result = m.result;
    $("status").textContent = "Python ready.";
    showResult(m.result);
  } else if (m.type === "progress") {
    setProgress(m.done, m.total);
  } else if (m.type === "processed") {
    state.results = m.result;
    setRunning(false, "Done.");
    setTabEnabled("results", true);
    showTab("results");
  } else if (m.type === "error") {
    $("status").textContent = "Python ready.";
    if (state.running) processFailed(m.text);
    else showMessage("Error: " + m.text, "error");
  }
};
worker.postMessage({ type: "init" });

function sendLoad(payload) {
  $("status").textContent = "Reading data…";
  worker.postMessage(
    { type: "load", files: payload.files, timestampMs: payload.timestampMs },
    Object.values(payload.files)
  );
}

// ── folder abstraction: Dir = { name, list() -> {files:[{name,getFile}], dirs:[Dir]} } ──
function dirFromHandle(h) {
  return {
    name: h.name, handle: h,
    async list() {
      const files = [], dirs = [];
      for await (const [name, child] of h.entries()) {
        if (child.kind === "file") files.push({ name, getFile: () => child.getFile() });
        else dirs.push(dirFromHandle(child));
      }
      return { files, dirs };
    },
  };
}

function dirFromEntry(entry) {
  return {
    name: entry.name,
    async list() {
      const reader = entry.createReader(), all = [];
      for (;;) {   // readEntries returns results in batches
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        all.push(...batch);
      }
      const files = [], dirs = [];
      for (const c of all) {
        if (c.isFile) files.push({ name: c.name, getFile: () => new Promise((res, rej) => c.file(res, rej)) });
        else dirs.push(dirFromEntry(c));
      }
      return { files, dirs };
    },
  };
}

function dirFromFileList(fileList) {   // <input webkitdirectory> fallback
  const build = (name, items, depth) => {
    const files = [], sub = new Map();
    for (const f of items) {
      const parts = f.webkitRelativePath.split("/").slice(depth + 1);
      if (parts.length === 1) files.push({ name: parts[0], getFile: async () => f });
      else if (!sub.has(parts[0])) sub.set(parts[0], [f]); else sub.get(parts[0]).push(f);
    }
    return {
      name,
      async list() { return { files, dirs: [...sub].map(([n, fs]) => build(n, fs, depth + 1)) }; },
    };
  };
  const arr = [...fileList];
  return build(arr[0].webkitRelativePath.split("/")[0], arr, 0);
}

// Find every folder (up to maxDepth below `dir`) that directly contains acqu.par.
async function findDatasets(dir, prefix = "", depth = 0, maxDepth = 3) {
  const { files, dirs } = await dir.list();
  const here = [];
  if (files.some((f) => f.name === "acqu.par")) here.push({ label: prefix || dir.name, dir, files });
  if (depth < maxDepth) {
    for (const d of dirs) here.push(...await findDatasets(d, (prefix || dir.name) + "/" + d.name, depth + 1, maxDepth));
  }
  return here;
}

const byLabel = (a, b) => a.label.localeCompare(b.label, undefined, { numeric: true });

// ── loading ───────────────────────────────────────────────────────────────
let datasets = [];

function resetSections() {
  hideMessage();
  $("params-section").hidden = true;
  state.result = state.results = null;
  for (const t of ["raw", "process", "results"]) setTabEnabled(t, false);
  showTab("load");
}

async function openFolder(dir) {
  resetSections();
  state.dirHandle = dir.handle || null;
  try {
    datasets = (await findDatasets(dir)).sort(byLabel);
  } catch (err) {
    return showMessage("Could not read that folder: " + err.message, "error");
  }
  if (!datasets.length) {
    return showMessage("No acqu.par found in that folder or the folders inside it. Is it a Kea data folder?", "warn");
  }
  const sel = $("dataset-select");
  sel.replaceChildren(...datasets.map((d, i) => Object.assign(document.createElement("option"), { value: i, textContent: d.label })));
  $("chooser").hidden = datasets.length === 1;
  await openDataset(0);
}

async function openDataset(i) {
  const ds = datasets[i];
  hideMessage();
  const want = ["acqu.par", "data.2d"];
  const files = {};
  let timestampMs = null;
  for (const name of want) {
    const entry = ds.files.find((f) => f.name === name);
    if (!entry) continue;
    const file = await entry.getFile();
    files[name] = await file.arrayBuffer();
    if (name === "data.2d") timestampMs = file.lastModified;   // when the data was written
  }
  $("folder-name").textContent = ds.label;
  const payload = { files, timestampMs };
  if (state.ready) sendLoad(payload);
  else { state.pending = payload; $("status").textContent = "Python is still loading; your data will open as soon as it's ready."; }
}

$("dataset-select").addEventListener("change", (e) => openDataset(+e.target.value));

// ── bundled sample data ("Explore test data") ─────────────────────────────
let samples = [];

async function initSamples() {
  try {
    const r = await fetch("test_data/manifest.json", { cache: "no-store" });
    if (!r.ok) return;
    samples = await r.json();
  } catch { return; }
  if (!samples.length) return;
  const sel = $("sample-select");
  sel.replaceChildren(
    Object.assign(document.createElement("option"), { value: "", textContent: "Choose…" }),
    ...samples.map((s, i) => Object.assign(document.createElement("option"), { value: i, textContent: s.label }))
  );
  $("sample-btn").hidden = false;
}

$("sample-btn").addEventListener("click", () => {
  const p = $("sample-picker");
  p.hidden = !p.hidden;
  if (!p.hidden) $("sample-select").focus();
});

$("sample-select").addEventListener("change", async (e) => {
  if (e.target.value === "") return;
  const s = samples[+e.target.value];
  resetSections();
  $("chooser").hidden = true;
  datasets = [];
  state.dirHandle = null;
  try {
    const files = {};
    for (const name of s.files.filter((f) => f !== "data.1d")) {
      const r = await fetch(`${s.path}/${name}`);
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      files[name] = await r.arrayBuffer();
    }
    $("folder-name").textContent = "sample: " + s.label;
    const payload = { files, timestampMs: null };
    if (state.ready) sendLoad(payload);
    else { state.pending = payload; $("status").textContent = "Python is still loading; the sample will open as soon as it's ready."; }
  } catch (err) {
    showMessage("Could not load the sample data: " + err.message, "error");
  }
});
initSamples();

// ── drop zone ─────────────────────────────────────────────────────────────
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  const item = e.dataTransfer.items && e.dataTransfer.items[0];
  if (!item) return;
  // Both lookups must start synchronously, before any await.
  const handleP = item.getAsFileSystemHandle ? item.getAsFileSystemHandle() : null;
  const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
  try {
    const h = handleP ? await handleP : null;
    if (h && h.kind === "directory") return openFolder(dirFromHandle(h));
    if (entry && entry.isDirectory) return openFolder(dirFromEntry(entry));
  } catch (err) {
    return showMessage("Could not read the dropped item: " + err.message, "error");
  }
  showMessage("Please drop a folder, not a file.", "warn");
});

async function browse() {
  if (window.showDirectoryPicker) {
    try { return openFolder(dirFromHandle(await window.showDirectoryPicker({ mode: "readwrite" }))); }
    catch (err) { if (err.name === "AbortError") return; throw err; }
  }
  const input = Object.assign(document.createElement("input"), { type: "file", webkitdirectory: true });
  input.onchange = () => input.files.length && openFolder(dirFromFileList(input.files));
  input.click();
}
drop.addEventListener("click", browse);
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); browse(); } });

// ── messages ──────────────────────────────────────────────────────────────
function showMessage(text, kind = "") {
  const m = $("message");
  m.textContent = text; m.className = "message " + kind; m.hidden = false;
}
function hideMessage() { $("message").hidden = true; }

// ── a dataset has been read ───────────────────────────────────────────────
function fillTable(table, rows) {
  table.replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    for (const t of [k, v]) tr.appendChild(Object.assign(document.createElement("td"), { textContent: t }));
    return tr;
  }));
}

function showResult(r) {
  fillTable($("params"), r.display.map((x) => [x.label, x.value]));
  fillTable($("params-all"), Object.entries(r.params_all).sort().map(([k, v]) => [k, Array.isArray(v) ? `[${v.join(", ")}]` : String(v)]));
  $("params-section").hidden = false;

  if (!r.is_cpmg) {
    return showMessage("Non-CPMG data: not yet supported (" + r.reason + "). Parameters are shown above.", "warn");
  }
  const idx = $("echo-idx");
  idx.max = r.raw.n_echoes - 1;
  idx.value = 0;
  initProcessDefaults(r.raw);
  setTabEnabled("raw", true);
  setTabEnabled("process", true);
}

// ── tabs ──────────────────────────────────────────────────────────────────
initTabs((name) => {
  if (name === "raw") renderRaw();       // plots drawn while hidden have no width, so draw on show
  if (name === "results") renderResults();
});
initRaw();
initProcess();
initResults();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderRaw(); renderResults(); });
