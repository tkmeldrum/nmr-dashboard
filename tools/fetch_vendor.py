"""Download the libraries the page needs into vendor/, so the site serves everything itself.

    python tools/fetch_vendor.py            # fetch what is missing, check every file against tools/vendor.lock.json
    python tools/fetch_vendor.py --update   # first time, or after changing a version below: record the new checksums

Versions are pinned here; checksums are pinned in tools/vendor.lock.json (commit it). Pyodide's own package files are also checked
against the hashes in its pyodide-lock.json. vendor/ is not committed: CI runs this before building the site.
"""
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
LOCK = Path(__file__).resolve().parent / "vendor.lock.json"

PYODIDE = "0.27.7"
PYODIDE_BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE}/full/"
PYODIDE_CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip"]
PYODIDE_PACKAGES = ["numpy", "scipy", "matplotlib", "micropip"]            # and everything they need
PLOTLY = ("plotly.min.js", "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js")
EMCEE = ("emcee-3.1.6-py2.py3-none-any.whl",
         "https://files.pythonhosted.org/packages/f9/ef/2196b9bf88ffa1bde45853c72df021fbd07a8fa91a0f59a22d14a050dc04/emcee-3.1.6-py2.py3-none-any.whl",
         "f2d63752023bdccf744461450e512a5b417ae7d28f18e12acd76a33de87580cb")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nmr-dashboard-fetch-vendor"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def closure(lock, names):
    seen = {}

    def go(n):
        n = n.lower().replace("_", "-")
        if n not in seen:
            seen[n] = lock["packages"][n]
            for d in seen[n].get("depends", []):
                go(d)
    for n in names:
        go(n)
    return seen


def main(update=False):
    pinned = json.loads(LOCK.read_text())["files"] if LOCK.exists() else {}
    got = {}
    problems = []

    def ensure(rel, url, expect=None):
        dest = VENDOR / rel
        if not dest.exists() or (expect and sha256(dest) != expect):
            print(f"  fetching {rel}")
            get(url, dest)
        h = sha256(dest)
        got[rel] = h
        if expect and h != expect:
            problems.append(f"{rel}: checksum {h} does not match the one the source publishes ({expect})")
        elif not update and rel in pinned and pinned[rel] != h:
            problems.append(f"{rel}: checksum {h} does not match tools/vendor.lock.json ({pinned[rel]})")
        elif not update and rel not in pinned:
            problems.append(f"{rel}: not in tools/vendor.lock.json (run with --update, then review and commit it)")

    VENDOR.mkdir(exist_ok=True)
    lock_path = VENDOR / "pyodide" / "pyodide-lock.json"
    get(PYODIDE_BASE + "pyodide-lock.json", lock_path)           # the full list, every run (it is small); then cut down to what we use
    full = json.loads(lock_path.read_text())
    pkgs = closure(full, PYODIDE_PACKAGES)
    lock_path.write_text(json.dumps({"info": full["info"], "packages": pkgs}, indent=1, sort_keys=True))
    for f in PYODIDE_CORE:
        ensure(f"pyodide/{f}", PYODIDE_BASE + f)
    for name, p in sorted(pkgs.items()):
        ensure(f"pyodide/{p['file_name']}", PYODIDE_BASE + p["file_name"], p.get("sha256"))
    ensure(f"plotly/{PLOTLY[0]}", PLOTLY[1])
    ensure(f"emcee/{EMCEE[0]}", EMCEE[1], EMCEE[2])
    if problems:
        print("\n".join("PROBLEM: " + p for p in problems))
        sys.exit(1)
    if update:
        LOCK.write_text(json.dumps({"pyodide": PYODIDE, "files": dict(sorted(got.items()))}, indent=1) + "\n")
        print(f"recorded {len(got)} checksums in {LOCK.name}")
    size = sum(f.stat().st_size for f in VENDOR.rglob("*") if f.is_file())
    print(f"vendor/ ready: {len(got)} files, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main("--update" in sys.argv[1:])
