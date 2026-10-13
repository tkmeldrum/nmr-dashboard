// Provenance of the results on the page, and the Export button (params.toml + runner script).
import { $, state } from "./state.js";
import { rpc } from "./rpc.js";

export function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const safe = (s) => (s || "dataset").replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 60);

function uncertaintySummary() {
  const u = state.unc, out = {};
  if (u && u.mcmc) out.mcmc = { steps_run: u.mcmc.steps_run, steps_planned: u.mcmc.steps_planned, burn: u.mcmc.burn, walkers: u.mcmc.nwalkers, interval_level: u.mcmc.level, warnings: u.mcmc.warnings };
  if (u && u.boot) out.bootstrap = { replicates: u.boot.replicates, planned: u.boot.planned, failures: u.boot.failures, method: u.boot.method, interval_level: u.boot.level, warnings: u.boot.warnings };
  if (u && u.target) out.analysed = u.target;
  return Object.keys(out).length ? out : null;
}

// What the run used: the settings are snapshotted in Python; the uncertainty options are snapshotted by process.js at click time.
export async function refreshProvenance() {
  const box = $("prov-details");
  try {
    const r = await rpc("get_provenance", {
      dataset: $("folder-name").textContent,
      browser: { userAgent: navigator.userAgent, page: location.origin + location.pathname },
      uncertainty: uncertaintySummary(),
    }, state.runOptions || {});
    state.provenance = r.record;
    $("prov-lines").replaceChildren(...r.lines.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
    box.hidden = false;
  } catch (err) {
    state.provenance = null;
    box.hidden = true;
  }
}

export function initProvenance() {
  $("prov-download").addEventListener("click", () => {
    if (!state.provenance) return;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    download(`provenance_${safe(state.provenance.dataset)}_${stamp}.json`, JSON.stringify(state.provenance, null, 2), "application/json");
  });
}
