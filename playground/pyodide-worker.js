// Runs Pyodide off the main thread. A runaway Neuva program (e.g. `while true {}`)
// hangs this worker, not the page — the UI stays responsive and the Stop button
// can recover by terminating this worker and spawning a fresh one.

importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodideReadyPromise = null;

function getPyodide() {
  if (!pyodideReadyPromise) {
    pyodideReadyPromise = (async () => {
      const pyodide = await loadPyodide();
      const res = await fetch(new URL("neuva_web.py", self.location.href));
      if (!res.ok) throw new Error(`could not load neuva_web.py (${res.status})`);
      const src = await res.text();
      pyodide.runPython(src);
      return pyodide;
    })();
  }
  return pyodideReadyPromise;
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  try {
    const pyodide = await getPyodide();

    if (type === "init") {
      self.postMessage({ id, ok: true });
      return;
    }

    if (type === "run") {
      pyodide.globals.set("__neuva_user_code", payload.code);
      const result = pyodide.runPython("run_neuva(__neuva_user_code)");
      self.postMessage({ id, ok: true, result });
      return;
    }

    if (type === "registerCsv") {
      const registerCsv = pyodide.globals.get("register_uploaded_csv");
      const result = registerCsv(payload.name, payload.text);
      self.postMessage({ id, ok: true, result });
      return;
    }

    self.postMessage({ id, ok: false, error: `unknown message type '${type}'` });
  } catch (err) {
    self.postMessage({ id, ok: false, error: err && err.message ? err.message : String(err) });
  }
};
