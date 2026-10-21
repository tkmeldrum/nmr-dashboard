"""The per-analysis PDF report: its content for each kind of analysis, its layout rules, and every way it is saved."""
import base64
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("emcee")

from nmrdash import formatting as F, params as P, report, runner, webbatch  # noqa: E402
from nmrdash.pipeline import (_STATE, _provenance_record, boot_finish, boot_start, boot_step, load, mcmc_finish, mcmc_start, mcmc_step,  # noqa: E402
                              process, save_bundle)

from sample_data import CPMG_SNR100, DATA, t2d_with_a_slow_population
CPMG = {"echo": "integrate", "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": True},
        "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 0.3, "n_steps": 60, "alpha": 1e8, "sweep": True}}
FT = {"echo": "ft", "ft": {"nucleus": "1H", "G": 23.67, "zero_fill": 3, "z_min": None, "z_max": None, "spacing_um": None, "ilt_spacing_um": None, "min_signal_pct": 5.0},
      "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": True},
      "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 0.1, "n_steps": 40, "alpha": 1e8, "sweep": False}}
T1 = {"kind": "t1", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": True}}
T2D = {"kind": "t2d", "t2d": {"t2_steps": 20, "d_steps": 10}, "t2d_model": {"enabled": True}}


def analysis(sample, settings, uncertainty=False, label=None):
    load(DATA / sample)
    sample = label or sample
    res = process(settings)
    mc = bs = None
    if uncertainty:
        mcmc_start({"steps": 300, "burn": 100, "nwalkers": 24})
        mcmc_step(300)
        mc = mcmc_finish({"fg": "#222", "grid": "#aaa"}, 0)
        boot_start({"replicates": 20})
        boot_step(20)
        bs = boot_finish()
    prov, _ = _provenance_record({"dataset": sample}, {}, outputs=["report.pdf"])
    return res, prov, mc, bs, _STATE.get("raw_out")


def all_text(blocks):
    out = []
    for b in blocks:
        t = b["type"]
        if t in ("title",):
            out += [b["text"], b["sub"] or ""]
        elif t in ("h2", "para"):
            out.append(b["text"])
        elif t == "bullets":
            out += b["items"]
        elif t == "table":
            out += (b["header"] or []) + [c for r in b["rows"] for c in r]
    return "\n".join(out)


def pdf_pages(pdf):
    return len(re.findall(rb"/Type /Page\b", pdf))


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    out = {}
    for key, args in (("cpmg", ("kea_cpmg_snr100", CPMG, True)), ("ft", ("kea_cpmg_snr100", FT, False)), ("t1", ("kea_t1_snr100", T1, False)), ("t2d", (str(t2d_with_a_slow_population(tmp_path_factory.mktemp("slow") / "ds")), T2D, True, "kea_t2d_slow_population"))):
        a = analysis(*args)
        blocks = report.compose(*a, style={})
        pages, scale = report.fit(blocks)
        out[key] = (a, blocks, pages, report.build(*a, style={}))
    return out


@pytest.mark.parametrize("key", ["cpmg", "ft", "t1", "t2d"])
def test_each_kind_gives_a_valid_letter_pdf_of_two_to_four_pages(reports, key):
    a, blocks, pages, pdf = reports[key]
    assert pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF")
    assert 2 <= len(pages) <= 4 and pdf_pages(pdf) == len(pages)
    assert re.search(rb"/MediaBox \[ 0 0 612 792 \]", pdf)                                # US Letter
    assert len(pdf) < 2_000_000


def test_the_cpmg_report_says_what_was_done_and_found(reports):
    (res, prov, mc, bs, raw), blocks, _, _ = reports["cpmg"]
    text = all_text(blocks)
    me = res["multiexp"]["selected"]
    for needle in ("Spinlab Data Analysis Report", "kea_cpmg_snr100", "Settings", "integrate each echo", "Exponential fit", F.fmt_sci(me["components"][0]["T2"]), "ILT",
                   "Uncertainty: methods compared", "MCMC: 24 walkers", "Bootstrap: 20 replicates", "Warnings and notes", "How these numbers were obtained", "Provenance", "SHA-256", "Provided as is"):
        assert needle in text, needle
    assert "exp(−t / T₂)" in text                                                       # the equations that apply are listed
    assert text.index("Settings") < text.index("Exponential fit") < text.index("Warnings and notes") < text.index("Provenance")   # page 1 content first, provenance last


def test_the_report_carries_every_warning(reports):
    (res, prov, mc, bs, raw), blocks, _, _ = reports["cpmg"]
    text = all_text(blocks)
    for w in res["multiexp"]["selected"]["warnings"]:
        assert w in text
    assert report.collect_warnings(res, mc, bs)                                        # this sample's fit is poorly determined and says so


def test_the_t2d_report_has_the_model_the_b_values_and_the_bound(reports):
    (res, prov, mc, bs, raw), blocks, _, _ = reports["t2d"]
    text = all_text(blocks)
    for needle in ("Few-population model", "G = 23.67 T/m", "b-values calculated", "Regularised map: regions", "(bound)", "D at the low limit", "Uncertainty: methods compared",
                   "T₂–D correlation", "only", "Figures"):
        assert needle in text or needle in "".join(b.get("caption") or "" for b in blocks if b["type"] == "image"), needle
    assert "nmrdash/physics.py" in text and "b = (γG)² · τ² · (Δ + 2τ/3)" in text


def test_the_t1_and_ft_reports_have_their_own_sections(reports):
    t1 = all_text(reports["t1"][1])
    assert "Saturation-recovery fit" in t1 and "No imaginary channel" in t1 and "(1 − exp(−t/T1))" in t1 and "Spatial analysis" not in t1
    ft = all_text(reports["ft"][1])
    assert "Spatial analysis (FT)" in ft and "FT of each echo" in ft and "natural resolution" in ft
    caps = [b["caption"] for b in reports["ft"][1] if b["type"] == "image"]
    assert "Spatial profile and echo map" in caps and "Position-resolved fits" in caps


def test_repeated_edge_warnings_from_mcmc_and_the_bootstrap_are_said_once():
    res = {"mode": "cpmg", "multiexp": None}
    a = {"warnings": ["The least-squares fit puts D of population 2 at the low edge of the range allowed, so the data give only a bound on it: its posterior interval is cut off."]}
    b = {"warnings": ["The least-squares fit puts D of population 2 at the low edge of the range allowed, so the data give only a bound on it: its bootstrap interval is cut off."]}
    w = report.collect_warnings(res, a, b)
    assert len(w) == 1 and "bootstrap and MCMC" in w[0]


# ── layout rules, with synthetic content ─────────────────────────────────────

def img(h_over_w, optional=False, width=5.0):
    return report.image(np.ones((int(200 * h_over_w), 200, 4)), width=width, optional=optional, caption="c")


def test_tables_and_images_are_not_split_and_headings_are_not_stranded():
    rows = [[str(i), "x"] for i in range(30)]
    blocks = [report.h2("A")] + [report.table(["n", "v"], rows, [1, 1]) for _ in range(6)] + [report.h2("Last"), img(0.8)]
    pages = report.paginate(blocks)
    assert len(pages) > 1
    for p in pages:
        assert p[-1]["type"] != "h2"                                                    # a heading is never the last thing on a page
        assert sum(report.height(b) for b in p) <= report.CONTENT_H + 1e-6


def test_a_figure_shrinks_into_the_space_left_rather_than_leaving_the_page_empty():
    intro = report.para("intro")
    big = img(1.0, width=6.0)
    room = 0.7 * report.height(big)                                                     # 70% of the figure fits on the page, 100% does not
    blocks = [intro, report.spacer(report.CONTENT_H - report.height(intro) - room), big]
    pages = report.paginate(blocks)
    placed = [b for p in pages for b in p if b["type"] == "image"][0]
    assert len(pages) == 1 and 0.65 < placed["fit"] < 0.75
    assert sum(report.height(b) for b in pages[0]) <= report.CONTENT_H + 1e-6
    tight = [intro, report.spacer(report.CONTENT_H - report.height(intro) - 0.3 * report.height(big)), big]    # only 30% would fit: it moves to the next page whole
    pages = report.paginate(tight)
    assert len(pages) == 2 and "fit" not in pages[1][-1]


def test_over_long_content_loses_its_optional_figures_last_first_and_never_the_essential_ones():
    essential = [img(1.3), img(1.3)]
    optional = [img(1.3, optional=True) for _ in range(6)]
    pages, _ = report.fit([report.para("intro")] + essential + optional, max_pages=3)
    kept = [b for p in pages for b in p if b["type"] == "image"]
    assert len(pages) <= 3 and all(any(b["array"] is e["array"] for b in kept) for e in essential)


# ── every way it is saved ────────────────────────────────────────────────────

def test_the_save_bundle_includes_the_report_unless_asked_not_to_and_can_make_only_the_report():
    analysis("kea_cpmg_snr100", CPMG)
    full = save_bundle({"dpi": 50})
    assert "report.pdf" in full["files"] and base64.b64decode(full["files"]["report.pdf"])[:5] == b"%PDF-"
    import json
    rec = json.loads(base64.b64decode(full["files"]["provenance.json"]))
    assert "report.pdf" in rec["outputs"] and set(rec["outputs"]) == set(full["files"])         # the record lists exactly what was written
    assert "report.pdf" not in save_bundle({"dpi": 50, "report": False, "figures": False})["files"]
    only = save_bundle({"report_only": True})
    assert list(only["files"]) == ["report.pdf"] and only["folder"].startswith("processed_")
    z = save_bundle({"zip": True, "report": True, "figures": False, "csv": False})
    assert "report.pdf" in z["files"]


def test_the_script_writes_the_report_and_a_batch_report(tmp_path):
    root = tmp_path / "series"
    for i in range(2):
        d = root / f"run{i}"
        shutil.copytree(CPMG_SNR100, d)
        for old in d.glob("processed_*"):
            shutil.rmtree(old)
    p = P.defaults()
    p["multiexp"].update({"enabled": True, "n": 1})
    p["output"].update({"figure_dpi": 50})
    rows, bdir = runner.run_batch(root, P.validate(p), quiet=True)
    for r in rows:
        pdf = (Path(r["output_dir"]) / "report.pdf").read_bytes()
        assert pdf[:5] == b"%PDF-" and 2 <= pdf_pages(pdf) <= 4
    batch = (bdir / "batch_report.pdf").read_bytes()
    assert batch[:5] == b"%PDF-" and pdf_pages(batch) >= 1
    p["output"]["report"] = False
    rows2, bdir2 = runner.run_batch(root, P.validate(p), quiet=True)
    assert not (bdir2 / "batch_report.pdf").exists() and not (Path(rows2[0]["output_dir"]) / "report.pdf").exists()


def test_the_batch_report_lists_every_dataset_and_why_some_were_skipped():
    rows = [{"dataset": "a/1", "status": "ok", "kind": "cpmg", "timestamp_utc": "2026-10-01T09:00:00+00:00", "elapsed_min": 0.0, "warnings": ["w1"],
             "result": {"multiexp": {"n": 1, "total_amplitude": 1.0, "components": [{"T2": 0.03, "fraction": 1.0, "T2_se": 0.001, "A": 1, "A_se": 0.1}]}, "t2d": None}},
            {"dataset": "a/2", "status": "ok", "kind": "cpmg", "timestamp_utc": "2026-10-01T09:30:00+00:00", "elapsed_min": 30.0, "warnings": [],
             "result": {"multiexp": {"n": 1, "components": [{"T2": 0.031, "fraction": 1.0, "T2_se": 0.001, "A": 1, "A_se": 0.1, "T2_ci": [0, 1]}], "total_amplitude": 1.0}, "t2d": None}},
            {"dataset": "b/T1", "status": "skipped", "kind": "t1", "reason": "a t1 dataset; this batch is cpmg", "timestamp_utc": None, "elapsed_min": None}]
    assert "T₂ 3.00e-2 s (100%)" in report._row_result(rows[0]) and "this batch is cpmg" in report._row_result(rows[2])
    pdf = report.build_batch(rows, P.defaults(), "demo")
    assert pdf[:5] == b"%PDF-" and pdf_pages(pdf) >= 1
    assert "nmrdash analysis report" not in all_text(report.compose(*analysis("kea_t1_snr100", T1), style={})).lower()


def test_the_params_file_has_a_report_switch():
    assert P.defaults()["output"]["report"] is True
    with pytest.raises(ValueError, match="output.report"):
        P.validate({"output": {"report": "yes"}})
    import tomllib
    assert tomllib.loads(P.dump_toml(P.defaults()))["output"]["report"] is True


def test_the_page_batch_carries_the_report_option(tmp_path, monkeypatch):
    monkeypatch.setattr(webbatch, "ROOT", str(tmp_path / "b"))
    settings = {"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False}, "ilt": {"enabled": False, "t2_min": 1e-5, "t2_max": 0.1, "n_steps": 40, "alpha": 1e8, "sweep": False}}
    webbatch.start({"settings": settings, "options": {"mcmc": {"enabled": False}, "bootstrap": {"enabled": False}}, "style": {}, "kind": "cpmg", "root_name": "demo", "output": {"figure_format": "png", "dpi": 50, "report": True}})
    for i, ts in enumerate((1_700_000_000_000, 1_700_000_600_000)):
        dest = Path(webbatch.ROOT) / str(i)
        shutil.copytree(CPMG_SNR100, dest)
        for old in dest.glob("processed_*"):
            shutil.rmtree(old)
        out = webbatch.run_one({"i": i, "label": f"d{i}", "timestamp_ms": ts})
        assert "report.pdf" in out["files"]
    assert "batch_report.pdf" in webbatch.finish()["files"]


# ── the batch report's table of per-run fit parameters ────────────────────────

def _cells(blocks):
    return [[c for r in b["rows"] for c in r] for b in blocks if b["type"] == "table"]


def test_the_batch_report_has_a_per_run_fit_parameter_table_for_each_kind():
    comp = lambda T, f, se=1e-4: {"T2": T, "T2_se": se, "A": 100.0 * f, "A_se": 3.0, "fraction": f, "fraction_se": 0.02}
    me = lambda n, comps: {"n": n, "components": comps, "offset": 1.5, "total_amplitude": 100.0, "chi2_red": 1.02}
    rows = [{"dataset": "s.tnt [record 1]", "array_value": "90°: 10, 180°: 20", "status": "ok", "kind": "cpmg", "timestamp_utc": "2026-10-01T09:00:00+00:00", "elapsed_min": 0.0, "warnings": [],
             "result": {"multiexp": me(2, [comp(0.005, 0.4), comp(0.02, 0.6)]), "t2d": None}},
            {"dataset": "s.tnt [record 2]", "array_value": "90°: 14, 180°: 28", "status": "ok", "kind": "cpmg", "timestamp_utc": "2026-10-01T09:00:00+00:00", "elapsed_min": 0.0, "warnings": [],
             "result": {"multiexp": me(2, [comp(0.0051, 0.41, None), comp(0.021, 0.59)]), "t2d": None}},
            {"dataset": "t1.tnt", "status": "ok", "kind": "t1", "timestamp_utc": None, "elapsed_min": None, "warnings": [],
             "result": {"multiexp": {**me(1, [comp(0.8, 1.0, 0.01)]), "chi2_red": None}, "t2d": None}},
            {"dataset": "d.tnt", "status": "ok", "kind": "t2d", "timestamp_utc": None, "elapsed_min": None, "warnings": [],
             "result": {"multiexp": None, "t2d": {"peaks": [], "model": {"K": 2, "chi2_per_point": 1.01, "populations": [
                 {"T2": 0.003, "T2_se": 1e-4, "D": 1.5e-9, "D_se": 5e-11, "fraction": 0.4, "fraction_se": 0.01, "supported": True},
                 {"T2": 0.012, "T2_se": 2e-4, "D": 3e-10, "D_se": None, "fraction": 0.6, "fraction_se": 0.01, "supported": True, "D_at_limit": "low"}]}}}},
            {"dataset": "skipped", "status": "skipped", "kind": None, "reason": "no", "timestamp_utc": None, "elapsed_min": None}]
    blocks = report._fit_tables([r for r in rows if r["status"] == "ok"])
    assert blocks[0]["text"] == "Fit parameters"
    cpmg, t1, t2d = [b for b in blocks if b["type"] == "table"]
    assert cpmg["header"][:4] == ["run", "T₂,1 (s)", "amplitude", "fraction"] and cpmg["header"][-2:] == ["offset", "χ²ᵣ"] and len(cpmg["rows"]) == 2
    assert cpmg["rows"][0][0] == "s.tnt [record 1]\n90°: 10, 180°: 20" and "±" in cpmg["rows"][0][1] and cpmg["rows"][0][3] == "40%" and "n/d" in cpmg["rows"][1][1]      # no uncertainty: said so
    assert t1["header"][1] == "T₁,1 (s)" and "χ²ᵣ" not in t1["header"] and t1["header"][-1] == "offset" and len(t1["rows"]) == 1                                                 # recovery data have no noise channel, so no χ²
    assert t2d["header"][:3] == ["run", "K", "population"] and len(t2d["rows"]) == 2 and t2d["rows"][0][0] == "d.tnt" and t2d["rows"][1][0] == "" and "D at the low limit" in t2d["rows"][1][-1]
    assert "40% ± 1%" in t2d["rows"][0][5]
    assert report.build_batch(rows, P.defaults(), "demo")[:5] == b"%PDF-"


def test_a_long_batch_table_is_cut_into_pieces_that_fit_a_page():
    comp = {"T2": 0.01, "T2_se": 1e-4, "A": 1.0, "A_se": 0.1, "fraction": 1.0, "fraction_se": 0.0}
    rows = [{"dataset": f"run{i}", "status": "ok", "kind": "cpmg", "timestamp_utc": None, "elapsed_min": None, "warnings": [],
             "result": {"multiexp": {"n": 1, "components": [comp], "offset": None, "total_amplitude": 1.0, "chi2_red": 1.0}, "t2d": None}} for i in range(60)]
    tables = [b for b in report._fit_tables(rows) if b["type"] == "table"]
    assert len(tables) == 3 and sum(len(b["rows"]) for b in tables) == 60 and max(len(b["rows"]) for b in tables) <= report.ROWS_PER_TABLE
    assert all(report.height(b) < report.CONTENT_H for b in tables)
    assert len(report.paginate(tables)[0]) >= 1                                          # and they flow onto pages


def test_the_arrayed_batch_report_carries_the_fit_table(tmp_path):
    shutil.copy(DATA / "tecmag_cpmg_power_series_snr30.tnt", tmp_path / "series.tnt")
    p = P.defaults()
    p["multiexp"]["compare"] = False
    p["ilt"]["max_grid_points"] = 40
    rows, bdir = runner.run_batch(tmp_path, p, quiet=True)
    blocks = report._fit_tables([r for r in rows if r["status"] == "ok"])
    (cpmg,) = [b for b in blocks if b["type"] == "table"]
    assert len(cpmg["rows"]) == 6 and [r[0].split("\n")[1] for r in cpmg["rows"]][0] == "90°: 10, 180°: 20"
    assert (bdir / "batch_report.pdf").read_bytes()[:5] == b"%PDF-"
