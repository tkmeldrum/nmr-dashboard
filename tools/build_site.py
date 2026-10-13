"""Assemble the static site: python tools/build_site.py <out_dir> [version]

Used by CI (deploy) and by local testing. Copies the web assets, the nmrdash
package, and the bundled sample datasets (acqu.par + data.2d/data.1d only),
and writes the two manifests the page reads. `version` is appended to asset
URLs so browsers never serve a stale app.js/style.css after a deploy.
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILES = ("acqu.par", "data.2d", "data.1d", "data.dat")


def sample_manifest(root=ROOT):
    """[{label, path, files}] for every folder under test_data/ that has an acqu.par."""
    base = Path(root) / "test_data"
    out = []
    for par in sorted(base.rglob("acqu.par")):
        d = par.parent
        out.append({
            "label": str(d.relative_to(base)),
            "path": "test_data/" + d.relative_to(base).as_posix(),
            "files": [f for f in SAMPLE_FILES if (d / f).is_file()],
        })
    return out


def commit_id():
    import os
    import subprocess
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def tree_is_dirty():
    import subprocess
    try:
        return bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "nmrdash"], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        return None


def _sub(text, old, new):
    assert old in text, f"expected {old!r} in source"
    return text.replace(old, new)


def build(out, version):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    html = (ROOT / "index.html").read_text()
    html = _sub(html, 'href="style.css"', f'href="style.css?v={version}"')
    html = _sub(html, 'src="app.js"', f'src="app.js?v={version}"')
    (out / "index.html").write_text(html)
    shutil.copy(ROOT / "style.css", out / "style.css")
    shutil.copy(ROOT / "worker.js", out / "worker.js")
    # ES modules: version every relative import so a deploy never mixes old and new modules
    for js in sorted(ROOT.glob("*.js")):
        if js.name == "worker.js":
            continue
        text = re.sub(r'(from "\./[^"]+\.js)"', rf'\1?v={version}"', js.read_text())
        if js.name == "app.js":
            text = _sub(text, 'new Worker("worker.js")', f'new Worker("worker.js?v={version}")')
        (out / js.name).write_text(text)

    pkg = out / "nmrdash"
    pkg.mkdir()
    names = sorted(p.name for p in (ROOT / "nmrdash").glob("*.py"))
    for n in names:
        shutil.copy(ROOT / "nmrdash" / n, pkg / n)
    # record which commit the deployed code came from, so provenance can name it
    (pkg / "_buildinfo.py").write_text(f'COMMIT = {commit_id()!r}\nDIRTY = {tree_is_dirty()!r}\nBUILT_VERSION = {version!r}\n')
    names = sorted(names + ["_buildinfo.py"])
    (out / "nmrdash-files.json").write_text(json.dumps(names))

    manifest = sample_manifest()
    for s in manifest:
        dest = out / s["path"]
        dest.mkdir(parents=True, exist_ok=True)
        for f in s["files"]:
            shutil.copy(ROOT / s["path"] / f, dest / f)
    (out / "test_data").mkdir(exist_ok=True)
    (out / "test_data" / "manifest.json").write_text(json.dumps(manifest))
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(int(time.time())))
