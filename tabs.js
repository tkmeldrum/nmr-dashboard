// Left-to-right workflow tabs.
import { $ } from "./state.js";

export const TABS = ["load", "raw", "process", "uncertainty", "results"];
const NAMES = { load: "Load", raw: "Raw data", process: "Process", uncertainty: "Uncertainty", results: "Results" };
let onShow = () => {};

export function showTab(name) {
  for (const t of TABS) {
    const on = t === name;
    $("tab-" + t).setAttribute("aria-selected", String(on));
    $("tab-" + t).tabIndex = on ? 0 : -1;
    $("panel-" + t).hidden = !on;
  }
  onShow(name);
}

export function setTabEnabled(name, enabled) {
  $("tab-" + name).disabled = !enabled;
  updateNav();
}

// Back / Next buttons at the top and bottom of every tab, so the workflow can be stepped through in either direction.
function buildNav() {
  TABS.forEach((name, i) => {
    const make = () => {
      const bar = document.createElement("div");
      bar.className = "tab-nav";
      for (const [dir, j] of [["back", i - 1], ["next", i + 1]]) {
        if (j < 0 || j >= TABS.length) { bar.append(document.createElement("span")); continue; }
        const b = document.createElement("button");
        b.type = "button";
        b.className = "secondary";
        b.dataset.go = TABS[j];
        b.dataset.dir = dir;
        b.addEventListener("click", () => showTab(TABS[j]));
        bar.append(b);
      }
      return bar;
    };
    const panel = $("panel-" + name);
    panel.prepend(make());
    panel.append(make());
  });
  updateNav();
}

function updateNav() {
  document.querySelectorAll(".tab-nav button[data-go]").forEach((b) => {
    const target = b.dataset.go, off = $("tab-" + target).disabled;
    b.disabled = off;
    b.textContent = b.dataset.dir === "back" ? `← ${NAMES[target]}` : off && target === "results" ? "Results appear after you run the processing" : `${NAMES[target]} →`;
  });
}

export function currentTab() {
  return TABS.find((t) => !$("panel-" + t).hidden);
}

export function initTabs(callback) {
  onShow = callback;
  for (const t of TABS) $("tab-" + t).addEventListener("click", () => showTab(t));
  $("tab-load").tabIndex = 0;
  document.querySelector(".tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const enabled = TABS.filter((t) => !$("tab-" + t).disabled);
    const i = enabled.indexOf(currentTab());
    const next = enabled[(i + (e.key === "ArrowRight" ? 1 : enabled.length - 1)) % enabled.length];
    showTab(next);
    $("tab-" + next).focus();
  });
  buildNav();
}
