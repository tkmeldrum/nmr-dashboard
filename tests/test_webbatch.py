import base64
import json
import os
import shutil
from pathlib import Path

from nmrdash import webbatch

DATA = Path(__file__).resolve().parent.parent / "test_data"
SETTINGS = {"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False},
            "ilt": {"enabled": False, "t2_min": 1e-5, "t2_max": 0.1, "n_steps": 40, "alpha": 1e8, "sweep": False}}
OPTIONS = {"mcmc": {"enabled": False}, "bootstrap": {"enabled": False}}


def stage(i, src):
    dest = Path(webbatch.ROOT) / str(i)
    shutil.copytree(src, dest)
    for p in dest.rglob("processed_*"):
        shutil.rmtree(p)


def test_a_batch_runs_every_dataset_like_the_script_and_summarises_them():
    webbatch.start({"settings": SETTINGS, "options": OPTIONS, "style": {}, "kind": "cpmg", "root_name": "demo",
                    "output": {"figure_format": "png", "dpi": 20, "samples": False}})
    runs = []
    for i, (label, src, ts) in enumerate([("a/2", DATA / "CPMG_EtGly" / "2", 1_700_000_600_000), ("a/1", DATA / "CPMG_EtGly" / "1", 1_700_000_000_000),
                                           ("b/T1", DATA / "T1SAT" / "1", 1_700_001_000_000)]):
        stage(i, src)
        runs.append(webbatch.run_one({"i": i, "label": label, "timestamp_ms": ts}))
    assert [r["status"] for r in runs] == ["ok", "ok", "skipped"] and "t1" in runs[2]["reason"] and not runs[2]["files"]
    assert {"decay.csv", "decay_fit.png", "summary.md", "provenance.json", "params_used.toml"} <= set(runs[0]["files"]) and runs[0]["folder"].startswith("processed_")
    assert base64.b64decode(runs[0]["files"]["decay_fit.png"])[:4] == b"\x89PNG"
    out = webbatch.finish()
    assert out["counts"] == {"ok": 2, "skipped": 1, "failed": 0} and out["folder"].startswith("batch_")
    assert {"batch_summary.csv", "params_used.toml", "provenance.json", "timeseries.png"} <= set(out["files"])
    lines = base64.b64decode(out["files"]["batch_summary.csv"]).decode().splitlines()
    head = lines[0].split(",")
    rows = [dict(zip(head, ln.split(","))) for ln in lines[1:]]
    assert [r["dataset"] for r in rows][:2] == ["a/1", "a/2"]                       # in the order they were measured, not staged
    assert float(rows[0]["elapsed_min"]) == 0 and abs(float(rows[1]["elapsed_min"]) - 10) < 1e-6      # 600 s later
    assert not os.path.exists(webbatch.ROOT)                                          # nothing left behind in the page's memory
    prov = json.loads(base64.b64decode(out["files"]["provenance.json"]))
    assert prov["kind"] == "batch" and len(prov["datasets"]) == 3


def test_the_script_batch_still_works_without_multiprocessing_imported_at_load():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "import nmrdash.runner, sys; assert 'concurrent.futures.process' not in sys.modules"], check=True)
