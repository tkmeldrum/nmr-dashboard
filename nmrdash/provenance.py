"""Provenance: a record of exactly how a set of results was produced.

The same record is written next to every result by the command-line tool and offered for download by the web
page. It ties the results to the input files (SHA-256), the fully resolved settings, the software that ran
(versions, commit), the random seeds and the machine, so an analysis can be audited and repeated.
"""
import datetime
import hashlib
import importlib
import platform
import sys
from pathlib import Path

from . import __version__

SCHEMA = 1
PACKAGES = ("numpy", "scipy", "matplotlib", "emcee")
ACQUISITION_KEYS = ("experiment", "expName", "echoTime", "nrEchoes", "nrPnts", "nrPntsT1", "nrPntsTau", "DELTA", "gradient", "dwellTime", "acqTime", "repTime", "nrScans",
                    "pulseLength", "amp90", "amp180", "b1Freq", "bandwidth", "rxGain", "rxPhase", "dataDirectory")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        try:
            return getattr(importlib.import_module(name), "__version__", None)
        except Exception:
            return None


def nmrdash_build():
    """{version, commit, dirty}: the commit comes from the deployed build, or from git for a source checkout.

    dirty is True when a source checkout has uncommitted changes, i.e. the code that ran is not exactly the commit.
    """
    commit, dirty = None, None
    try:
        from . import _buildinfo
        commit, dirty = _buildinfo.COMMIT, getattr(_buildinfo, "DIRTY", None)
    except ImportError:
        try:
            import subprocess
            root = Path(__file__).resolve().parent.parent
            if (root / ".git").exists():
                commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
                dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "nmrdash"], capture_output=True, text=True, timeout=5).stdout.strip())
        except Exception:
            pass
    return {"version": __version__, "commit": commit, "dirty": dirty}


def environment():
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation() + (" (Pyodide, in the browser)" if sys.platform == "emscripten" else ""),
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in PACKAGES},
    }


def file_record(path, label=None):
    p = Path(path)
    st = p.stat()
    return {"name": label or p.name, "bytes": st.st_size, "sha256": sha256(p),
            "modified_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(timespec="seconds")}


def build(data_dir, settings, options=None, *, params_text=None, params_path=None, command=None, dataset=None,
          acquisition=None, notes=None, extra=None, outputs=None, input_names=("acqu.par", "data.2d", "data.dat", "data2D_Cp.2d", "data2D.dat", "DiffAxis.dat", "T2Axis.dat"), modified_ms=None):
    """Assemble the record.

    data_dir : folder holding the input files (the script's data folder, or the browser's in-memory copy)
    settings / options : the resolved settings passed to pipeline.process and the uncertainty options
    params_text : the params file as run (stored verbatim, with its hash)
    modified_ms : the original modification time of data.2d (the browser cannot see it on its in-memory copy)
    """
    data_dir = Path(data_dir)
    inputs = []
    for name in input_names:
        if (data_dir / name).is_file():
            rec = file_record(data_dir / name)
            if name == next((n for n in ("data.2d", "data2D_Cp.2d", "data2D.dat", "data.dat") if (data_dir / n).is_file()), None) and modified_ms:
                rec["modified_utc"] = datetime.datetime.fromtimestamp(modified_ms / 1000, datetime.timezone.utc).isoformat(timespec="seconds")
            inputs.append(rec)
    acq = {k: acquisition[k] for k in ACQUISITION_KEYS if acquisition and k in acquisition}
    opts = options or {}
    seeds = {k: opts[k].get("seed") for k in ("mcmc", "bootstrap") if opts.get(k, {}).get("enabled")}
    rec = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset or str(data_dir),
        "nmrdash": nmrdash_build(),
        "environment": environment(),
        "command": command,
        "inputs": inputs,
        "acquisition": acq,
        "settings": settings,
        "uncertainty_options": opts,
        "random_seeds": seeds,
        "notes": list(notes or []),
        "outputs": list(outputs or []),
    }
    if params_text is not None:
        h = hashlib.sha256(params_text.encode("utf-8")).hexdigest()
        rec["params_file"] = {"path": str(params_path) if params_path else None, "sha256": h, "text": params_text}
    if extra:
        rec["browser"] = extra
    return rec


def describe(rec):
    """A few human-readable lines for a summary or the web page."""
    nm, env = rec["nmrdash"], rec["environment"]
    pk = ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)
    lines = [f"nmrdash {nm['version']}" + (f" (commit {nm['commit'][:10]}{', with uncommitted changes' if nm.get('dirty') else ''})" if nm.get("commit") else ""),
             f"{env['implementation']} {env['python']} on {env['platform']}", f"packages: {pk}"]
    for f in rec["inputs"]:
        lines.append(f"input {f['name']}: {f['bytes']} bytes, sha256 {f['sha256'][:16]}…")
    if rec.get("random_seeds"):
        lines.append("random seeds: " + ", ".join(f"{k}={v}" for k, v in rec["random_seeds"].items()))
    lines.append(f"produced {rec['created_utc']}")
    return lines
