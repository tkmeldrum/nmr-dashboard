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
SAMPLE_FILES = ("acqu.par", "data.2d", "data.1d", "data.dat", "data2D_Cp.2d", "data2D.dat", "DiffAxis.dat", "T2Axis.dat")


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


# What the About page says about where the page's own code and libraries come from (kept next to the Content-Security-Policy it describes)
PRIVACY_TEXT = ("Everything the page needs, including the Python runtime and the plotting library, is served from this site, and a content-security policy built into "
                "the page forbids it from connecting to anywhere else, so it cannot send your data out even by mistake.")


def _source_ref():
    """The git ref the source links point at: the commit being built (so the link shows exactly the running code), or main for an unclean local build."""
    commit = commit_id()
    return commit if commit and not tree_is_dirty() else "main"


def _code_links(functions, ref):
    import html
    import inspect
    out = []
    for fn in functions:
        lines, start = inspect.getsourcelines(fn)
        path = Path(inspect.getsourcefile(fn)).resolve().relative_to(ROOT)
        out.append(f'<a href="https://github.com/tkmeldrum/nmr-dashboard/blob/{ref}/{path}#L{start}-L{start + len(lines) - 1}"><code>{html.escape(path.name)}: {html.escape(fn.__name__)}</code></a>')
    return ", ".join(out)


def _model_cards(entries, ref):
    import html
    cards = []
    for m in entries:
        cards.append(
            f'<div class="model"><h3>{html.escape(m["name"])}</h3>'
            f'<p class="equation">{html.escape(m["equation"])}</p>'
            f'<p>{html.escape(m["describes"])}</p>'
            f'<p class="note"><strong>Where it comes from:</strong> {html.escape(m["source"])}</p>'
            f'<p class="note"><strong>The code:</strong> {_code_links(m["functions"], ref)}</p></div>')
    return "\n".join(cards)


def physics_html():
    """The About page's physics section, generated from the registries in nmrdash/physics.py and nmrdash/inference.py."""
    import html
    sys.path.insert(0, str(ROOT))
    from nmrdash import inference, physics
    ref = _source_ref()
    not_modelled = "".join(f"<li>{html.escape(t)}</li>" for t in physics.NOT_MODELLED)
    return (f'<h3 class="sub">The signal models and geometry</h3>'
            f'<p class="note">All of these are in one short file, <a href="https://github.com/tkmeldrum/nmr-dashboard/blob/{ref}/nmrdash/physics.py"><code>nmrdash/physics.py</code></a>, which uses only NumPy; '
            f'every other part of the tool calls these functions, so what you read there is what runs.</p>\n{_model_cards(physics.MODELS, ref)}\n'
            f'<h3 class="sub">What is not modelled</h3><ul>{not_modelled}</ul>\n'
            f'<h3 class="sub">How numbers are estimated from the models</h3>'
            f'<p class="note">Noise, goodness of fit, model selection, intervals and the regularisation of the inversions are in '
            f'<a href="https://github.com/tkmeldrum/nmr-dashboard/blob/{ref}/nmrdash/inference.py"><code>nmrdash/inference.py</code></a>.</p>\n{_model_cards(inference.METHODS, ref)}')


def build(out, version, vendor=True):
    out = Path(out)
    if vendor and not (ROOT / "vendor" / "pyodide" / "pyodide.js").is_file():
        raise SystemExit("vendor/ is missing: run  python tools/fetch_vendor.py  first")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    html = (ROOT / "index.html").read_text()
    html = _sub(html, 'href="style.css"', f'href="style.css?v={version}"')
    html = _sub(html, 'src="app.js"', f'src="app.js?v={version}"')
    (out / "index.html").write_text(html)
    shutil.copy(ROOT / "style.css", out / "style.css")
    about = (ROOT / "about.html").read_text()
    about = _sub(about, "{{COMMIT}}", (commit_id() or "unknown")[:10] + ("+local changes" if tree_is_dirty() else ""))
    about = _sub(about, "{{BUILT}}", time.strftime("built %Y-%m-%d", time.gmtime()))
    about = _sub(about, "{{PRIVACY}}", PRIVACY_TEXT)
    about = _sub(about, "{{PHYSICS}}", physics_html())
    about = _sub(about, 'href="style.css"', f'href="style.css?v={version}"')
    (out / "about.html").write_text(about)
    for src, dst in (("LICENSE", "LICENSE.txt"), ("NOTICE", "NOTICE.txt"), ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.txt")):
        shutil.copy(ROOT / src, out / dst)
    shutil.copy(ROOT / "worker.js", out / "worker.js")
    # ES modules: version every relative import so a deploy never mixes old and new modules
    for js in sorted(ROOT.glob("*.js")):
        if js.name == "worker.js":
            continue
        text = re.sub(r'(from "\./[^"]+\.js)"', rf'\1?v={version}"', js.read_text())
        if js.name == "app.js":
            text = _sub(text, 'new Worker("worker.js")', f'new Worker("worker.js?v={version}")')
        (out / js.name).write_text(text)

    if vendor:
        shutil.copytree(ROOT / "vendor", out / "vendor")      # Pyodide, its packages, Plotly and emcee (tools/fetch_vendor.py)

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
