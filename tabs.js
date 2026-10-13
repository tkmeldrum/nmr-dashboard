// Left-to-right workflow tabs.
import { $ } from "./state.js";

export const TABS = ["load", "raw", "process", "uncertainty", "results"];
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
  document.querySelectorAll("[data-next]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.next)));
}
