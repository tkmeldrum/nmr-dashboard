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
    assert 'worker.js?v=abc123' in (out / "app.js").read_text()
    assert json.loads((out / "nmrdash-files.json").read_text()) == sorted(p.name for p in (ROOT / "nmrdash").glob("*.py"))
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
    assert "File timestamp" not in [x["label"] for x in r["display"]]
