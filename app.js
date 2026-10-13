const status = document.getElementById("status");
const out = document.getElementById("out");
const worker = new Worker("worker.js");

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") status.textContent = m.text;
  else if (m.type === "hello") {
    status.textContent = "Python ready.";
    out.textContent = JSON.stringify(m.result, null, 2);
  } else if (m.type === "error") {
    status.textContent = "Error: " + m.text;
  }
};
worker.postMessage({ type: "init" });
