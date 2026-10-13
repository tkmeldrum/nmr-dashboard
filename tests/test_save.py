import base64
import io
import json
import zipfile
from pathlib import Path

import pytest

from nmrdash import figures
from nmrdash.pipeline import load, process, save_bundle

DATA = Path(__file__).resolve().parent.parent / "test_data"
ETGLY = DATA / "CPMG_EtGly" / "1"


def run_cpmg(**extra):
    r = load(ETGLY)
    t = r["raw"]["echo_time_s"]
    return process({"echo": "integrate",
                    "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": False},
                    "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 10 * t[-1], "n_steps": 60, "alpha": 1e8, "sweep": False}, **extra})


def test_bundle_has_tables_figures_and_records():
    run_cpmg()
    out = save_bundle({"figure_format": "png", "dpi": 60, "page": {"dataset": "x"}})
    names = set(out["files"])
    assert out["folder"].startswith("processed_")
    assert {"decay_fit.png", "ilt_spectrum.png", "params_used.toml", "provenance.json", "summary.md"} <= names
    assert any(n.endswith(".csv") for n in names)
    assert base64.b64decode(out["files"]["decay_fit.png"])[:4] == b"\x89PNG"
    rec = json.loads(base64.b64decode(out["files"]["provenance.json"]))
    assert set(rec["outputs"]) == names                    # the record lists what was written, no more and no less


def test_zip_holds_one_folder_with_the_same_files():
    run_cpmg()
    out = save_bundle({"zip": True, "dpi": 50, "figures": False})
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(out["zip_b64"]))) as z:
        assert {n.split("/")[0] for n in z.namelist()} == {out["zip_name"][:-4]}
        assert sorted(n.split("/", 1)[1] for n in z.namelist()) == sorted(out["files"])
    assert not any(n.endswith(".png") for n in out["files"])


def test_formats_svg_and_pdf():
    run_cpmg()
    assert b"<svg" in base64.b64decode(save_bundle({"figure_format": "svg"})["files"]["decay_fit.svg"])[:400]
    assert base64.b64decode(save_bundle({"figure_format": "pdf", "dpi": 50})["files"]["decay_fit.pdf"])[:4] == b"%PDF"


def test_saving_needs_a_run_and_uncertainty_belongs_to_its_run():
    from nmrdash.pipeline import _STATE
    load(ETGLY)
    _STATE.pop("last_result", None)
    with pytest.raises(RuntimeError, match="Run the processing"):
        save_bundle({})
    run_cpmg()
    _STATE["mcmc"] = object()
    run_cpmg()                                             # re-processing drops the earlier MCMC run
    assert "mcmc" not in _STATE


def fig(style, builder="decay"):
    run_cpmg()
    from nmrdash.pipeline import _STATE
    res = _STATE["last_result"]
    return figures.styled(figures.decay_figure if builder == "decay" else figures.ilt_figure, style, res, name="decay_fit" if builder == "decay" else "ilt")


def test_style_title_modes():
    ax = fig({"title": "auto"}).axes[0]
    assert ax.get_title()
    assert fig({"title": "none"}).axes[0].get_title() == ""
    assert fig({"title": "custom", "title_text": "My sample"}).axes[0].get_title() == "My sample"


def test_style_font_line_marker_legend():
    base, big = fig({}), fig({"font_size": 20, "line_scale": 2, "marker_scale": 2, "legend": False})
    assert big.axes[1].get_xlabel() and big.axes[1].xaxis.label.get_fontsize() > base.axes[1].xaxis.label.get_fontsize()
    assert max(l.get_linewidth() for l in big.axes[0].lines) == pytest.approx(2 * max(l.get_linewidth() for l in base.axes[0].lines))
    assert base.axes[0].get_legend() is not None and big.axes[0].get_legend() is None


def test_style_milliseconds_relabels_axes_but_not_data():
    s, ms = fig({}), fig({"time_unit": "ms"})
    assert "(s)" in s.axes[1].get_xlabel() and "(ms)" in ms.axes[1].get_xlabel()
    ms.canvas.draw()
    ticks = [t.get_text() for t in ms.axes[1].get_xticklabels() if t.get_text()]
    x = ms.axes[1].get_xticks()[1]
    assert float(ticks[1]) == pytest.approx(x * 1e3, rel=1e-3)
    assert ms.axes[0].lines[0].get_xdata()[3] == s.axes[0].lines[0].get_xdata()[3]     # the data are untouched


def test_style_ilt_log_axis_in_ms_and_size():
    f = fig({"time_unit": "ms", "width": 5, "height": 3}, "ilt")
    assert "(ms)" in f.axes[0].get_xlabel() and f.get_size_inches().tolist() == [5, 3]


def test_style_decay_log_y_and_bad_values_are_repaired():
    assert fig({"decay_logy": True}).axes[0].get_yscale() == "log"
    s = figures.style_from({"font_size": 999, "cmap": "nope", "time_unit": "years", "title": "x", "width": "", "line_scale": -1})
    assert s["font_size"] == 40 and s["cmap"] == "viridis" and s["time_unit"] == "s" and s["title"] == "auto" and s["width"] is None and s["line_scale"] == 0.1


def test_colormap_applies_to_heat_maps():
    from matplotlib.cm import ScalarMappable
    fm = figures.styled(figures.t2d_fit_figure, {"cmap": "magma"}, _fake_t2d(), name="t2d_fit", retight=True)
    names = sorted({m.cmap.name for m in fm.findobj(lambda o: isinstance(o, ScalarMappable)) if getattr(m, "cmap", None)})
    assert names == ["RdBu_r", "magma"]                    # the diverging residual map keeps its own colours


def _fake_t2d():
    import numpy as np
    e, b = np.linspace(0.001, 0.1, 20), np.linspace(1, 5, 8)
    z = np.random.default_rng(0).normal(size=(8, 20))
    return {"t2d": {"echo_s": e.tolist(), "b_vec": b.tolist(), "pred": (z * 0 + 1).tolist(), "resid": z.tolist()}}


def test_t2d_and_t1_results_save_too():
    load(DATA / "M1115_rt_30phr" / "1")
    process({"kind": "t2d", "t2d": {"t2_steps": 12, "d_steps": 8}})
    out = save_bundle({"figure_format": "png", "dpi": 50, "style": {"cmap": "magma", "time_unit": "ms", "title": "none"}})
    assert {"t2d_spectrum.png", "t2d_fit.png", "summary.md"} <= set(out["files"])
    load(DATA / "T1SAT" / "1")
    process({"kind": "t1", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False}})
    assert "decay_fit.png" in save_bundle({"dpi": 50})["files"]


def test_saving_does_not_need_modules_the_browser_lacks():
    """Pyodide has no multiprocessing: the page's save path must not import the batch runner."""
    import subprocess
    import sys
    code = "import sys, nmrdash.pipeline as p; sys.modules.pop('_multiprocessing', None); assert 'nmrdash.runner' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
