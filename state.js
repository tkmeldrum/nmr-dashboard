// Shared state and DOM helper.
export const $ = (id) => document.getElementById(id);

export const state = {
  worker: null,
  ready: false,       // Python finished loading
  pending: null,      // a dataset waiting for Python
  result: null,       // output of pipeline.load
  results: null,      // output of pipeline.process
  running: false,     // a processing job is in flight
  dirHandle: null,    // writable folder handle (Chromium), used when saving
};
