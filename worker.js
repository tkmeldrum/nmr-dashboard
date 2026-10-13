const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

let pyodide;

async function init() {
  postMessage({ type: "status", text: "Loading Python (first load ~30 MB, then cached)…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage(["numpy", "scipy", "matplotlib"]);
  const files = await (await fetch("nmrdash-files.json")).json();
  pyodide.FS.mkdirTree("/home/pyodide/nmrdash");
  for (const f of files) {
    const src = await (await fetch("nmrdash/" + f)).text();
    pyodide.FS.writeFile("/home/pyodide/nmrdash/" + f, src);
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  const result = pyodide.runPython(
    "import json; from nmrdash.pipeline import hello; json.dumps(hello())"
  );
  postMessage({ type: "hello", result: JSON.parse(result) });
}

onmessage = async (e) => {
  try {
    if (e.data.type === "init") await init();
  } catch (err) {
    postMessage({ type: "error", text: String(err) });
  }
};
