// A visible "something is happening" indicator, so slow steps never look like a frozen page.
import { $ } from "./state.js";

let depth = 0;
const messages = [];

function show() {
  const el = $("busy");
  if (!messages.length) { el.hidden = true; $("busy-stop").hidden = true; return; }
  $("busy-text").textContent = messages[messages.length - 1];
  el.hidden = false;
}

export function setBusy(text) {            // setBusy("...") shows it, setBusy(null) clears everything
  messages.length = 0;
  if (text) messages.push(text);
  show();
}

// Show the message, let the browser paint it, then run the (possibly slow, synchronous) work.
export async function withBusy(text, fn) {
  messages.push(text);
  show();
  await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
  try {
    return await fn();
  } finally {
    messages.splice(messages.lastIndexOf(text), 1);
    show();
  }
}

// A long job that cannot report progress (the 2D inversion) offers a Stop button; app.js restarts Python when it is pressed.
export function setStoppable(on) {
  $("busy-stop").hidden = !on;
}
