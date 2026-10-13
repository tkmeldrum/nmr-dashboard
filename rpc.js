// Promise-style calls to the Python worker (used for the chunked MCMC / bootstrap runs).
import { state } from "./state.js";

let nextId = 1;
const pending = new Map();

export function rpc(fn, ...args) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    state.worker.postMessage({ type: "rpc", id, fn, args });
  });
}

export function handleReply(m) {
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  if (m.error) p.reject(new Error(m.error));
  else p.resolve(m.result);
}
