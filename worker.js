// Runs Pyodide off the main thread so the UI never freezes during processing.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

let pyodide;

const status = (text) => postMessage({ type: "status", text });

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

onmessage = async (e) => {
  try {
    if (e.data.type === "init") await init();
    else if (e.data.type === "load") loadDataset(e.data);
    else if (e.data.type === "process") processDataset(e.data);
  } catch (err) {
    postMessage({ type: "error", text: String(err.message || err) });
  }
};
