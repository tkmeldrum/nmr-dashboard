// Runs Pyodide off the main thread so the UI never freezes during processing.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

let pyodide;

const status = (text) => postMessage({ type: "status", text });

// A Python exception arrives with its whole traceback; the last line ("ValueError: ...") is the part worth showing.
const brief = (err) => String((err && err.message) || err).trim().split("\n").map((l) => l.trim()).filter(Boolean).pop();

async function init() {
  status("Loading Python (first load ~30 MB, then cached)…");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  status("Loading numpy / scipy / matplotlib…");
  await pyodide.loadPackage(["numpy", "scipy", "matplotlib"]);
  const files = await (await fetch("nmrdash-files.json", { cache: "no-store" })).json();
  pyodide.FS.mkdirTree("/home/pyodide/nmrdash");
  for (const f of files) {
    const src = await (await fetch("nmrdash/" + f, { cache: "no-store" })).text();
    pyodide.FS.writeFile("/home/pyodide/nmrdash/" + f, src);
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  postMessage({ type: "ready" });
}

function loadDataset({ files, timestampMs }) {
  // files: { name: ArrayBuffer }. Replace whatever was loaded before.
  if (pyodide.FS.analyzePath("/data").exists) {
    for (const n of pyodide.FS.readdir("/data")) {
      if (n !== "." && n !== "..") pyodide.FS.unlink("/data/" + n);
    }
  } else {
    pyodide.FS.mkdir("/data");
  }
  for (const [name, buf] of Object.entries(files)) {
    pyodide.FS.writeFile("/data/" + name, new Uint8Array(buf));
  }
  pyodide.globals.set("_ts_ms", timestampMs ?? undefined);   // JS null would arrive as jsnull, not None
  const json = pyodide.runPython(
    "import json\nfrom nmrdash.pipeline import load\njson.dumps(load('/data', _ts_ms, fallback_to_mtime=False))"
  );
  postMessage({ type: "loaded", result: JSON.parse(json) });
}

function processDataset({ settings }) {
  pyodide.globals.set("_settings_json", JSON.stringify(settings));
  pyodide.globals.set("_progress", (done, total) => postMessage({ type: "progress", done, total }));
  const json = pyodide.runPython(
    "import json\nfrom nmrdash.pipeline import process\njson.dumps(process(json.loads(_settings_json), _progress))"
  );
  postMessage({ type: "processed", result: JSON.parse(json) });
}

// Calls the page makes while a long job runs: small functions in nmrdash.pipeline, plus the one-off emcee install.
const RPC_FUNCTIONS = new Set(["mcmc_start", "mcmc_step", "mcmc_finish", "boot_start", "boot_step", "boot_finish",
                               "get_provenance", "export_params", "export_script"]);

async function installEmcee() {
  try { pyodide.runPython("import emcee"); return { already: true }; } catch (_) { /* not installed yet */ }
  status("Fetching the emcee package from PyPI (once, about 100 KB)…");
  try {
    await pyodide.loadPackage("micropip");
    await pyodide.pyimport("micropip").install("emcee==3.1.6");
  } catch (err) {
    throw new Error("Could not download the emcee package from PyPI (are you offline?): " + (err.message || err));
  }
  return { installed: true };
}

async function handleRpc({ id, fn, args }) {
  try {
    let result;
    if (fn === "install_emcee") result = await installEmcee();
    else if (RPC_FUNCTIONS.has(fn)) {
      pyodide.globals.set("_rpc_args", JSON.stringify(args));
      result = JSON.parse(pyodide.runPython(
        `import json\nfrom nmrdash import pipeline as _p\njson.dumps(_p.${fn}(*json.loads(_rpc_args)))`));
    } else throw new Error("Unknown call: " + fn);
    postMessage({ type: "reply", id, result });
  } catch (err) {
    postMessage({ type: "reply", id, error: brief(err) });
  }
}

onmessage = async (e) => {
  if (e.data.type === "rpc") return handleRpc(e.data);
  try {
    if (e.data.type === "init") await init();
    else if (e.data.type === "load") loadDataset(e.data);
    else if (e.data.type === "process") processDataset(e.data);
  } catch (err) {
    postMessage({ type: "error", text: brief(err) });
  }
};
