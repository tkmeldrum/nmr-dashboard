import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from build_site import build, sample_manifest  # noqa: E402

from nmrdash.pipeline import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_manifest_lists_samples():
    m = sample_manifest()
    assert any(s["label"] == "CPMG_EtGly/1" for s in m)
    for s in m:
        assert "acqu.par" in s["files"]
        assert s["path"].startswith("test_data/")


def test_build_bundles_site(tmp_path):
    out = build(tmp_path / "site", "abc123")
    html = (out / "index.html").read_text()
    assert 'style.css?v=abc123' in html and 'app.js?v=abc123' in html
    app = (out / "app.js").read_text()
    assert 'worker.js?v=abc123' in app and 'from "./state.js?v=abc123"' in app
    assert not any('.js"' in line and line.startswith("import") for line in app.splitlines())
    for js in ("state.js", "tabs.js", "raw.js", "process.js", "results.js", "plotutil.js", "rpc.js", "uncertainty.js", "provenance.js", "busy.js", "kinds.js", "t2dview.js"):
        assert (out / js).is_file()
    expected = sorted([p.name for p in (ROOT / "nmrdash").glob("*.py")] + ["_buildinfo.py"])
    assert json.loads((out / "nmrdash-files.json").read_text()) == sorted(set(expected))
    assert "COMMIT = " in (out / "nmrdash" / "_buildinfo.py").read_text()      # provenance can name the deployed commit
    manifest = json.loads((out / "test_data" / "manifest.json").read_text())
    assert manifest
    for s in manifest:
        for f in s["files"]:
            assert (out / s["path"] / f).is_file()
    # only the data the page needs is shipped, not figures/exports
    assert not list(out.rglob("*.png")) and not list(out.rglob("*.npz"))
    # a bundled sample loads
    s = next(s for s in manifest if s["label"] == "CPMG_EtGly/1")
    assert load(out / s["path"], None, fallback_to_mtime=False)["is_cpmg"]


def test_no_timestamp_mode():
    r = load(ROOT / "test_data" / "CPMG_EtGly" / "1", None, fallback_to_mtime=False)
    row = next(x for x in r["display"] if x["label"] == "File timestamp")
    assert row["value"] == "not available"


def test_html_ids_are_unique_and_every_js_lookup_exists():
    """A duplicate id silently makes getElementById hit the wrong element (it once hid a checkbox instead of a plot)."""
    import collections
    import re
    html = (ROOT / "index.html").read_text()
    ids = re.findall(r'\bid="([^"]+)"', html)
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    assert not dupes, f"duplicate ids: {dupes}"
    used = set()
    for js in ROOT.glob("*.js"):
        if js.name != "worker.js":
            used |= set(re.findall(r'\$\("([^"]+)"\)', js.read_text()))
    dynamic = {"tab-", "panel-"}                      # ids built as "tab-" + name in tabs.js
    missing = sorted(i for i in used if i not in ids and not any(i.startswith(p) for p in dynamic))
    assert not missing, f"JS looks up ids that are not in index.html: {missing}"


def test_build_ships_the_about_page_and_licence_files(tmp_path):
    out = build(tmp_path / "site", "abc123")
    about = (out / "about.html").read_text()
    assert "{{" not in about and "built 20" in about and "style.css?v=abc123" in about            # placeholders filled, assets versioned
    assert "Apache License, Version 2.0" in about and "as is" in about and "Pyodide" in about and "NumPy" in about
    assert "Apache License" in (out / "LICENSE.txt").read_text()[:400] and "Copyright" in (out / "NOTICE.txt").read_text()
    assert "Venkataramanan" in (out / "THIRD_PARTY_NOTICES.txt").read_text()
    assert 'href="about.html"' in (out / "index.html").read_text()
    assert "<script" not in about                       # a static page: no code, nothing to load


def test_the_about_page_lists_the_versions_the_site_actually_uses():
    about = (ROOT / "about.html").read_text()
    worker = (ROOT / "worker.js").read_text()
    import re
    assert re.search(r"v0\.27\.7|0\.27\.7", worker) and "0.27.7" in about
    assert "plotly.js-dist-min@2.35.2" in (ROOT / "index.html").read_text() and "2.35.2" in about
    assert "emcee==3.1.6" in worker and "3.1.6" in about
