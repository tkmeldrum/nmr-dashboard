"""Batch processing driven by the web page.

The page stages one dataset's files at a time under ROOT/<i>/ and asks for one run (run_one); when the last one is done it asks
for the combined summary (finish). Every dataset goes through runner.run_dataset, so the results are exactly what the script writes.
Files come back base64-encoded, to be written into the user's folders or a zip by the page.
"""
import base64
import datetime
import os
import shutil
from pathlib import Path

from . import outputs, params as P, report, runner

ROOT = "/tmp/nmrdash_batch"
_B = {}


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def start(a):
    """a: {settings, options, style, kind, output: {csv, figures, samples, figure_format, dpi}, root_name}."""
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    p = P.with_style(P.from_settings(a["settings"], a["options"]), a.get("style"))
    o = a.get("output") or {}
    for k, key in (("csv", "csv"), ("figures", "figures"), ("report", "report"), ("samples", "samples"), ("figure_format", "figure_format"), ("dpi", "figure_dpi")):
        if o.get(k) is not None:
            p["output"][key] = max(50, int(o[k])) if k == "dpi" else o[k]       # the params file allows 50-1200 dpi
    p = P.validate(p)
    _B.clear()
    _B.update(rows=[], params=p, kind=a.get("kind"), root_name=a.get("root_name") or "datasets",
              text=P.dump_toml(p, header=f"Batch run from the nmrdash web page ({a.get('root_name') or 'datasets'})"))
    return {"ok": True}


def run_one(a):
    """a: {i, label, timestamp_ms}. Returns the outcome and the files this dataset produced."""
    folder = Path(ROOT) / str(a["i"])
    ts = a.get("timestamp_ms")
    record = a.get("record")
    if ts:                                                      # the page knows when the data were written; the files here are fresh copies
        for name in ("data.2d", "data2D_Cp.2d", "data2D.dat", "data.dat"):
            if (folder / name).is_file():
                os.utime(folder / name, (ts / 1000.0, ts / 1000.0))
    row = runner.run_dataset(folder, _B["params"], params_text=_B["text"], rel_name=a["label"], command="nmrdash web page (batch)", only_kind=_B["kind"], record=record)
    files, name = {}, None
    if row.get("output_dir"):
        od = Path(row["output_dir"])
        name = od.name
        files = {f.name: _b64(f) for f in sorted(od.iterdir()) if f.is_file()}
        shutil.rmtree(od, ignore_errors=True)
        row["output_dir"] = f"{a['label']}/{name}"
    shutil.rmtree(folder, ignore_errors=True)
    _B["rows"].append(row)
    return {"status": row["status"], "reason": row["reason"], "seconds": row.get("seconds"), "warnings": row.get("warnings", []), "folder": name, "files": files}


def finish(a=None):
    """The combined summary: batch_summary.csv, params_used.toml, provenance.json and the time-series figure(s)."""
    rows, p = _B["rows"], _B["params"]
    stamps = [datetime.datetime.fromisoformat(r["timestamp_utc"]) if r.get("timestamp_utc") else None for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (stamps[i] is None, stamps[i] or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc), rows[i].get("record") or 0, rows[i]["dataset"]))
    rows = [rows[i] for i in order]
    stamps = [stamps[i] for i in order]
    t0 = runner._parse_t0(p["batch"]["t0"], stamps)
    for r, t in zip(rows, stamps):
        r["elapsed_min"] = (t - t0).total_seconds() / 60 if (t and t0) else None
    bdir = Path(ROOT) / "batch_out"
    bdir.mkdir(parents=True, exist_ok=True)
    runner.write_batch(bdir, rows, p, _B["text"], None, "nmrdash web page (batch)", _B["root_name"], t0)
    out = {"folder": f"batch_{runner._stamp()}", "files": {f.name: _b64(f) for f in sorted(bdir.iterdir()) if f.is_file()},
           "counts": {k: sum(r["status"] == k for r in rows) for k in ("ok", "skipped", "failed")},
           "fit_tables": [{k: t[k] for k in ("title", "note", "header", "rows")} for t in report.fit_table_data([r for r in rows if r["status"] == "ok"])]}      # shown on the page when the batch is done
    shutil.rmtree(ROOT, ignore_errors=True)
    return out
