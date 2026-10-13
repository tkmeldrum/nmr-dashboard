// Main thread: folder loading, parameter display, raw-data plots.
const $ = (id) => document.getElementById(id);
const state = { ready: false, pending: null, result: null, dirHandle: null };

// ── worker ────────────────────────────────────────────────────────────────
const worker = new Worker("worker.js");
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
  } else if (m.type === "error") {
    $("status").textContent = "Python ready.";
    showMessage("Error: " + m.text, "error");
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

async function openFolder(dir) {
  hideMessage();
  $("params-section").hidden = $("raw-section").hidden = $("process-section").hidden = true;
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
    if (name === "data.2d" || timestampMs === null) timestampMs = file.lastModified;
  }
  $("folder-name").textContent = ds.label;
  const payload = { files, timestampMs };
  if (state.ready) sendLoad(payload);
  else { state.pending = payload; $("status").textContent = "Python is still loading; your data will open as soon as it's ready."; }
}

$("dataset-select").addEventListener("change", (e) => openDataset(+e.target.value));

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

// ── results ───────────────────────────────────────────────────────────────
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
    $("raw-section").hidden = true;
    $("process-section").hidden = true;
    return showMessage("Non-CPMG data: not yet supported (" + r.reason + "). Parameters are shown above.", "warn");
  }
  $("raw-section").hidden = false;
  $("process-section").hidden = false;
  const idx = $("echo-idx");
  idx.max = r.raw.n_echoes - 1;
  idx.value = 0;
  renderRaw();
}

// ── plots ─────────────────────────────────────────────────────────────────
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLORS = { re: "#0072B2", im: "#D55E00", mag: "#009E73" };   // colour-blind safe

function baseLayout(title, xt, yt, extra = {}) {
  const fg = css("--plot-fg"), grid = css("--plot-grid");
  return {
    title: { text: title, font: { size: 14 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg },
    margin: { l: 60, r: 15, t: 40, b: 50 },
    xaxis: { title: xt, gridcolor: grid, zerolinecolor: grid, automargin: true },
    yaxis: { title: yt, gridcolor: grid, zerolinecolor: grid, automargin: true },
    legend: { orientation: "h", y: -0.25 },
    ...extra,
  };
}
const CFG = { responsive: true, displaylogo: false };

function renderRaw() {
  const raw = state.result && state.result.raw;
  if (!raw) return;
  const t = raw.echo_time_s;
  const mag = raw.real.map((row, p) => row.map((re, e) => Math.hypot(re, raw.imag[p][e])));

  Plotly.react("plot-heat", [{
    type: "heatmap", z: mag, x: t, y: mag.map((_, i) => i + 1), colorscale: "Viridis",
    colorbar: { title: "|S|" },
  }], baseLayout("Echo magnitude", "Echo time (s)", "Point within echo"), CFG);

  const e = +$("echo-idx").value;
  $("echo-idx-out").textContent = `${e + 1} of ${raw.n_echoes} (t = ${+t[e].toPrecision(4)} s)`;
  const pts = raw.real.map((_, i) => i + 1);
  Plotly.react("plot-echo", [
    { x: pts, y: raw.real.map((r) => r[e]), name: "Real", mode: "lines+markers", line: { color: COLORS.re } },
    { x: pts, y: raw.imag.map((r) => r[e]), name: "Imaginary", mode: "lines+markers", line: { color: COLORS.im } },
    { x: pts, y: mag.map((r) => r[e]), name: "Magnitude", mode: "lines+markers", line: { color: COLORS.mag } },
  ], baseLayout("Single echo", "Point within echo", "Signal (arb.)"), CFG);

  Plotly.react("plot-decay", [
    { x: t, y: raw.summed, mode: "lines+markers", name: "Summed", line: { color: COLORS.re }, marker: { size: 4 } },
  ], baseLayout("Summed echo decay (phased real part)", "Echo time (s)", "Signal (arb.)",
     { yaxis: { ...baseLayout("", "", "").yaxis, title: "Signal (arb.)", type: $("decay-log").checked ? "log" : "linear" } }), CFG);
}

$("echo-idx").addEventListener("input", renderRaw);
$("decay-log").addEventListener("change", renderRaw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderRaw);
