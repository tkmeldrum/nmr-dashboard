// Feedback and problem reports. Nothing is sent by the page itself: the buttons open the user's
// own mail program or a GitHub issue, pre-filled with text the user can read and edit first.
import { $, state } from "./state.js";

const EMAIL = "tkmeldrum@wm.edu";
const ISSUES = "https://github.com/tkmeldrum/nmr-dashboard/issues/new";

function diagnostics() {
  const tab = document.querySelector('[role="tab"][aria-selected="true"]');
  const shown = [...document.querySelectorAll(".message.error:not([hidden]), .message.warn:not([hidden])")]
    .map((el) => el.textContent.trim()).filter(Boolean).slice(0, 3);
  const lines = [
    `Page: ${location.origin}${location.pathname}`,
    `Tab: ${tab ? tab.textContent.trim() : "?"}`,
    `Sequence: ${state.result ? (state.result.kind || "unknown") : "no data loaded"}`,
    `Browser: ${navigator.userAgent}`,
  ];
  if (shown.length) lines.push("Messages on screen:", ...shown.map((s) => "  " + s.slice(0, 300)));
  return lines.join("\n");
}

function body() {
  const text = $("fb-text").value.trim();
  return $("fb-diag").checked ? `${text}\n\n---\n${diagnostics()}` : text;
}

export function initFeedback() {
  const refresh = () => { $("fb-preview").textContent = $("fb-diag").checked ? diagnostics() : "(nothing added)"; };
  $("feedback").addEventListener("toggle", refresh);
  $("fb-diag").addEventListener("change", refresh);
  $("fb-mail").addEventListener("click", () => {
    location.href = `mailto:${EMAIL}?subject=${encodeURIComponent("NMR dashboard feedback")}&body=${encodeURIComponent(body())}`;
  });
  $("fb-issue").addEventListener("click", () => {
    window.open(`${ISSUES}?title=${encodeURIComponent("Feedback")}&body=${encodeURIComponent(body())}`, "_blank", "noopener");
  });
}
