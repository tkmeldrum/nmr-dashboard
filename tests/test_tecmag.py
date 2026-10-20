"""Tecmag datasets through the dashboard: what each sequence becomes, that the known truth is recovered, and every way a file can be unsuitable."""
import json
import shutil
from datetime import datetime

import numpy as np
import pytest
from sample_data import (CPMG_SNR100, DATA, TECMAG_CPMG_CLEAN, TECMAG_CPMG_SNR100, TECMAG_T1_CLEAN, TECMAG_T1_SNR100, TECMAG_T2D_CLEAN, TECMAG_T2D_SNR100, TRUTH,
                         tecmag_variant)
from tnt_writer import write_tnt

from nmrdash import physics, tecmag
from nmrdash.pipeline import _STATE, get_provenance, load, process

CPMG_FIT = {"echo": "integrate", "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": False}, "ilt": {"enabled": False}}
T1_FIT = {"kind": "t1", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False}}
T2D_FIT = {"kind": "t2d", "t2d": {"t2_steps": 24, "d_steps": 15, "gradient": 6.59}, "t2d_model": {"enabled": True}}


@pytest.mark.parametrize("name,kind", [("CPMG", "cpmg"), ("CPMG_EtOH", "cpmg"), ("cpmg2D_sample", "cpmg"), ("T1_SaturationRecovery", "t1"), ("T1_saturation_recovery", "t1"),
                                       ("StimEchoDiffusion_singleAcqPeriod_newPhaseCycle", "t2d"), ("StimEchoDiffusion_T2D", "t2d"), ("SSE-T2", "t2d"),
                                       ("SEDiff_7", None), ("T1_InversionRecovery", None), ("Pulse_acquire", None), ("", None)])
def test_the_sequence_file_name_picks_the_analysis(name, kind):
    k, reason = tecmag.kind_of_sequence(name)
    assert k == kind and bool(reason) == (kind is None)


def test_a_diffusion_series_is_declined_with_its_name_and_a_reason():
    assert "diffusion series" in tecmag.kind_of_sequence("SEDiff_1")[1] and "SEDiff_1" in tecmag.kind_of_sequence("SEDiff_1")[1]


def test_find_takes_a_file_or_the_only_tnt_in_a_folder(tmp_path):
    shutil.copy(TECMAG_T1_CLEAN, tmp_path / "a.tnt")
    assert tecmag.find(tmp_path / "a.tnt") == tmp_path / "a.tnt" and tecmag.find(tmp_path) == tmp_path / "a.tnt"
    (tmp_path / "notes.txt").write_text("x")
    assert tecmag.find(tmp_path / "notes.txt") is None
    shutil.copy(TECMAG_T1_CLEAN, tmp_path / "B.TNT")
    with pytest.raises(ValueError, match="2 .tnt files"):
        tecmag.find(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tecmag.find(empty) is None
    with pytest.raises(FileNotFoundError, match="Tecmag"):
        load(empty)


# ── what each sequence becomes ───────────────────────────────────────────────

@pytest.mark.parametrize("path,kind,expect", [
    (TECMAG_CPMG_SNR100, "cpmg", {"echoTime": 150.0, "nrEchoes": 256, "nrPnts": 32, "dwellTime": 2.0, "pulseLength": 6.0, "repTime": 1000.0, "nrScans": 64}),
    (TECMAG_T1_SNR100, "t1", {"echoTime": 150.0, "nrEchoes": 16, "nrPnts": 16, "nrPntsT1": 16, "dwellTime": 2.0, "repTime": 1000.0}),
    (TECMAG_T2D_SNR100, "t2d", {"echoTime": 60.0, "nrEchoes": 300, "nrPnts": 20, "nrPntsTau": 16, "DELTA": 10.0, "dwellTime": 2.0, "nrScans": 128}),
])
def test_each_sequence_loads_with_kea_style_parameters(path, kind, expect):
    r = load(path)
    assert r["supported"] and r["kind"] == kind and r["source_format"] == "tecmag" and r["warnings"] == []
    for k, v in expect.items():
        assert r["params_all"][k] == pytest.approx(v), k
    assert r["params_all"]["nucleus"] == "1H" and r["params_all"]["b1Freq"] == pytest.approx(13.235)
    rows = {x["label"]: x["value"] for x in r["display"]}
    assert rows["Sequence file"] and rows["Experiment date (stored in the file)"] == "2026-01-15 10:30:00" and "File timestamp" not in rows
    assert r["is_cpmg"] == (kind == "cpmg")


def test_the_gradient_default_follows_the_instrument_not_the_kind_of_data():
    assert load(TECMAG_CPMG_SNR100)["defaults"]["G"] == 6.59 and load(TECMAG_T2D_SNR100)["defaults"]["gradient"] == 6.59
    assert load(DATA / "kea_cpmg_snr100")["defaults"]["G"] == 23.67 and load(DATA / "kea_t2d_snr100")["defaults"]["gradient"] == 23.67


def test_the_nucleus_comes_from_the_file_header():
    r = load(TECMAG_CPMG_SNR100)
    assert r["defaults"]["nucleus"] == "1H" and "from the file header" in r["defaults"]["nucleus_reason"]
    assert "from the file header" not in load(CPMG_SNR100)["defaults"]["nucleus_reason"]               # Kea: from b1Freq


def test_the_timestamp_is_the_experiment_date_in_the_file_not_the_file_time():
    load(TECMAG_CPMG_SNR100, 1_700_000_000_000)                                                      # a file time the browser reports: not used
    assert datetime.fromtimestamp(_STATE["source"]["timestamp_ms"] / 1000) == datetime(2026, 1, 15, 10, 30, 0)


# ── the known truth comes back ───────────────────────────────────────────────

@pytest.mark.parametrize("path,tol", [(TECMAG_CPMG_CLEAN, 0.01), (TECMAG_CPMG_SNR100, 0.15)])
def test_cpmg_recovers_both_t2_by_integration_and_by_ft(path, tol):
    c = TRUTH["tecmag_cpmg"]
    load(path)
    for cfg in (CPMG_FIT, {**CPMG_FIT, "echo": "ft", "ft": {"nucleus": "1H", "G": 6.59, "zero_fill": 4, "z_min": None, "z_max": None}}):
        comps = process(cfg)["multiexp"]["selected"]["components"]
        assert [x["T2"] for x in comps] == pytest.approx(c["T2_s"], rel=tol)
        assert [x["fraction"] for x in comps] == pytest.approx(c["fraction"], abs=tol)


def test_cpmg_noise_is_measured_from_the_imaginary_channel():
    load(TECMAG_CPMG_SNR100)
    r = process(CPMG_FIT)
    assert r["decay"]["noise_sigma"] == pytest.approx(r["decay"]["y"][0] / 100, rel=0.3)                    # the file was made with an SNR of 100 at the first echo


@pytest.mark.parametrize("path,tol", [(TECMAG_T1_CLEAN, 1e-3), (TECMAG_T1_SNR100, 0.05)])
def test_t1_is_recovered(path, tol):
    load(path)
    c = process(T1_FIT)["multiexp"]["selected"]["components"][0]
    assert c["T2"] == pytest.approx(TRUTH["tecmag_t1"]["T1_s"], rel=tol)                                # (the field is called T2 for every decay; for T1 it is T1)


def test_t1_delays_are_the_table_plus_the_pulse_width():
    r = load(TECMAG_T1_CLEAN)
    c = TRUTH["tecmag_t1"]
    want = physics.geometric_grid(c["t_min_s"], c["t_max_s"], c["n_delays"])
    assert r["raw"]["t"] == pytest.approx(list(want), abs=1.5e-6)                                           # the table is rounded to a microsecond


@pytest.mark.parametrize("path,chi_max", [(TECMAG_T2D_CLEAN, 5.0), (TECMAG_T2D_SNR100, 1.15)])
def test_t2d_recovers_both_populations(path, chi_max):
    c = TRUTH["tecmag_t2d"]
    load(path)
    res = process(T2D_FIT)["t2d"]
    m = res["model"]
    assert m["K"] == 2 and m["chi2_per_point"] < chi_max and not any(p["D_at_limit"] for p in m["populations"])
    for p, T2, D, f in zip(m["populations"], c["T2_s"], c["D_m2s"], c["fraction"]):
        assert p["T2"] == pytest.approx(T2, rel=0.03) and p["D"] == pytest.approx(D, rel=0.03) and p["fraction"] == pytest.approx(f, abs=0.02)


def test_t2d_uses_the_cpmg_echoes_and_leaves_out_the_stimulated_echo():
    """Echo k is at k x tE; window 0 (the stimulated echo) is dropped, so the first echo is one echo time in, as in Kea data."""
    r = load(TECMAG_T2D_CLEAN)
    c = TRUTH["tecmag_t2d"]
    assert r["raw"]["echo_time_s"][0] == pytest.approx(c["echo_time_us"] * 1e-6) and r["raw"]["n_echoes"] == c["n_echoes"]
    L = _STATE["t2d"]
    te = c["echo_time_us"] * 1e-6
    b = physics.b_value_stimulated_echo(L["tau_s"][0], c["DELTA_ms"] * 1e-3, c["G_t_per_m"], "1H")
    want = c["amplitude"] * sum(f * np.exp(-te / T2) * np.exp(-b * D) for f, T2, D in zip(c["fraction"], c["T2_s"], c["D_m2s"]))
    assert L["data"][0, 0] == pytest.approx(want, rel=2e-3)


def test_t2d_tau_is_the_table_plus_the_pulse_width_and_delta_is_as_stored():
    r = load(TECMAG_T2D_CLEAN)
    c = TRUTH["tecmag_t2d"]
    assert r["raw"]["tau_s"][0] == pytest.approx(c["tau_table_us"][0] * 1e-6 + c["pulse_us"] * 1e-6) and r["raw"]["tau_s"][-1] == pytest.approx((c["tau_table_us"][1] + c["pulse_us"]) * 1e-6)
    assert r["raw"]["b_info"]["delta_ms"] == pytest.approx(10.0) and r["raw"]["b_info"]["source"] == "tecmag"


def test_t2d_b_values_follow_the_gradient_setting():
    load(TECMAG_T2D_SNR100)
    from nmrdash.pipeline import t2d_axes
    a, b = t2d_axes({"t2d": {"gradient": 6.59}}), t2d_axes({"t2d": {"gradient": 13.18}})
    assert np.allclose(np.array(b["b_vec"]) / np.array(a["b_vec"]), 4.0) and a["b_check"]["compared"] == []              # no spectrometer files to compare with


def test_a_cpmg_profile_has_the_same_orientation_as_the_kea_equivalent(tmp_path):
    """Both formats give the same z for a layer on the + side: the Tecmag data are stored conjugated and read back conjugated (as the lab's MATLAB reader does)."""
    layer = {"T2_s": (0.02,), "fraction": (1.0,)}
    from make_synthetic_data import write_cpmg
    write_cpmg(tmp_path / "kea", True, np.random.default_rng(5), "k", layers_um=((60.0, 120.0),), **layer)
    tecmag_variant(tmp_path / "t.tnt", "cpmg", layers_um=((216.0, 431.0),), **layer)
    centroids = []
    for path, g in ((tmp_path / "kea", 23.67), (tmp_path / "t.tnt", 6.59)):
        load(path)
        ft = process({**CPMG_FIT, "echo": "ft", "ft": {"nucleus": "1H", "G": g, "zero_fill": 4, "z_min": None, "z_max": None}})["ft"]
        z, p = np.array(ft["z_um"]), np.array(ft["profile"])
        centroids.append(float((z * p).sum() / p.sum()))
    assert centroids[0] > 0 and centroids[1] > 0


# ── files that are not what the dashboard analyses ───────────────────────────

def test_a_diffusion_series_is_declined_in_the_page_message(tmp_path):
    p = tmp_path / "d.tnt"
    write_tnt(p, np.ones((5, 64), complex), points_per_window=8, dwell_s=2e-6, sequence_file="SEDiff_test", variables={"nrEchoes": "8", "tE": "100u"}, locv=("tE",))
    r = load(p)
    assert not r["supported"] and r["kind"] is None and "diffusion series" in r["reason"] and r["raw"] is None


def test_an_unreadable_file_raises_a_message_not_a_crash(tmp_path):
    (tmp_path / "bad.tnt").write_bytes(b"this is not a tnt file" * 100)
    with pytest.raises(ValueError, match="not a Tecmag"):
        load(tmp_path / "bad.tnt")


def test_a_cpmg_with_several_records_is_declined(tmp_path):
    p = tmp_path / "c.tnt"
    write_tnt(p, np.ones((3, 256), complex), points_per_window=8, dwell_s=2e-6, sequence_file="CPMG", variables={"nrEchoes": "32", "tE": "100u"}, locv=("tE",))
    r = load(p)
    assert not r["supported"] and "3 records" in r["reason"]


def test_a_delay_table_that_does_not_match_the_records_is_declined(tmp_path):
    p = tmp_path / "t.tnt"
    rows = [("Delay", [("5u", {}), ("=T1Wait", {"delay": "de02"})])]
    write_tnt(p, np.ones((4, 64), complex), points_per_window=8, dwell_s=2e-6, sequence_file="T1_SaturationRecovery", rows=rows, tables={"de02": (["10", "20", "30"], "delay")},
              variables={"nrEchoes": "8", "tEcho": "100u", "pw": "6u"}, locv=("tEcho",))
    r = load(p)
    assert not r["supported"] and "3 values but the file holds 4 records" in r["reason"]


def test_a_t1_without_a_delay_table_or_a_t2d_without_delta_says_what_is_missing(tmp_path):
    p = tmp_path / "t.tnt"
    write_tnt(p, np.ones((4, 64), complex), points_per_window=8, dwell_s=2e-6, sequence_file="T1_SaturationRecovery", variables={"nrEchoes": "8", "tEcho": "100u"}, locv=("tEcho",))
    assert "no delay table" in load(p)["reason"]
    q = tmp_path / "d.tnt"
    rows = [("Delay", [("=d1", {"delay": "de02"})])]
    write_tnt(q, np.ones((2, 64), complex), points_per_window=64, dwell_s=2e-6, sequence_file="StimEchoDiffusion_x", rows=rows, tables={"de02": (["10", "20"], "delay")},
              variables={"nrEchoes": "1", "tE": "40u", "EchoAcqTime": "10u", "pw": "6u"}, locv=("tE",))
    assert "no DELTA" in load(q)["reason"]


def test_a_missing_echo_time_is_named(tmp_path):
    p = tmp_path / "c.tnt"
    write_tnt(p, np.ones(256, complex), points_per_window=8, dwell_s=2e-6, sequence_file="CPMG", variables={"nrEchoes": "32"})
    assert "echo time variable" in load(p)["reason"]


def test_an_inconsistent_echo_count_is_noted_and_the_record_wins(tmp_path):
    p = tmp_path / "c.tnt"
    write_tnt(p, np.ones(256, complex), points_per_window=8, dwell_s=2e-6, sequence_file="CPMG", variables={"nrEchoes": "10", "tE": "100u", "pw": "6u"}, locv=("tE",))
    r = load(p)
    assert r["supported"] and r["raw"]["n_echoes"] == 32 and any("nrEchoes = 10" in w and "32" in w for w in r["warnings"])


def test_an_echo_off_centre_in_its_window_is_flagged(tmp_path):
    n_pts, n_echo = 32, 40
    rec = np.zeros((n_echo, n_pts), complex)
    rec[:, 3] = 100.0                                                                                # the echo peaks at the start of the window
    rec += 0.01
    p = tmp_path / "c.tnt"
    write_tnt(p, rec.reshape(-1), points_per_window=n_pts, dwell_s=2e-6, sequence_file="CPMG", variables={"nrEchoes": str(n_echo), "tE": "100u", "pw": "6u"}, locv=("tE",))
    assert any("peaks at point 4 of 32" in w for w in load(p)["warnings"])


@pytest.mark.parametrize("when,fragment", [(datetime(1970, 1, 1), "cannot be right"), (datetime(2099, 1, 1), "cannot be right")])
def test_an_implausible_experiment_date_is_flagged_not_corrected(tmp_path, when, fragment):
    p = tmp_path / "c.tnt"
    write_tnt(p, np.ones(256, complex), points_per_window=8, dwell_s=2e-6, sequence_file="CPMG", variables={"nrEchoes": "32", "tE": "100u"}, locv=("tE",), date=when)
    r = load(p)
    assert r["supported"] and any(fragment in w for w in r["warnings"])
    assert datetime.fromtimestamp(_STATE["source"]["timestamp_ms"] / 1000) == when                       # shown as stored: the user decides


def test_provenance_lists_the_tnt_file_as_the_input():
    load(TECMAG_CPMG_SNR100)
    process(CPMG_FIT)
    rec = get_provenance({"dataset": "sample: tecmag_cpmg_snr100"}, {})["record"]
    assert [f["name"] for f in rec["inputs"]] == ["tecmag_cpmg_snr100.tnt"] and rec["acquisition"].get("nrEchoes") == 256
    json.dumps(rec)


# ── scripts, batches and the page's batch ────────────────────────────────────

def _fast():
    from nmrdash import params as P
    p = P.defaults()
    p["multiexp"]["compare"] = False
    p["ilt"]["max_grid_points"] = 40
    p["t2d"].update({"t2_steps": 20, "d_steps": 10})
    return p


@pytest.mark.parametrize("path,kind", [(TECMAG_CPMG_SNR100, "cpmg"), (TECMAG_T1_SNR100, "t1"), (TECMAG_T2D_SNR100, "t2d")])
def test_the_script_runs_a_tnt_file_and_writes_its_results_next_to_it(tmp_path, path, kind):
    from nmrdash import runner
    f = tmp_path / path.name
    shutil.copy(path, f)
    row = runner.run_dataset(f, _fast())
    assert row["status"] == "ok" and row["kind"] == kind, row.get("reason")
    assert row["dataset"] == path.name and row["timestamp_local"] == "2026-01-15T10:30:00"            # the experiment date in the file, not the copy's file time
    out = tmp_path / row["output_dir"].split("/")[-1]
    assert out.parent == tmp_path and out.name.startswith(f"processed_{path.stem}_")
    prov = json.loads((out / "provenance.json").read_text())
    assert [i["name"] for i in prov["inputs"]] == [path.name] and prov["inputs"][0]["modified_utc"].startswith("2026-01-15") and prov["acquisition"]["source"] == "tecmag"
    md = (out / "summary.md").read_text()
    assert "experiment date stored in the .tnt file" in md and ("from the sequence's delay table" in md) == (kind == "t2d") and (out / "report.pdf").read_bytes()[:5] == b"%PDF-"


def test_the_script_uses_the_instrument_gradient_when_the_params_say_auto(tmp_path):
    from nmrdash import runner
    f = tmp_path / "t.tnt"
    shutil.copy(TECMAG_T2D_SNR100, f)
    row = runner.run_dataset(f, _fast())
    assert row["settings"]["t2d"]["gradient"] == 6.59
    k = tmp_path / "kea"
    shutil.copytree(DATA / "kea_t2d_snr100", k)
    assert runner.run_dataset(k, _fast())["settings"]["t2d"]["gradient"] == 23.67
    explicit = _fast()
    explicit["t2d"]["gradient"] = 7.5
    assert runner.run_dataset(f, explicit)["settings"]["t2d"]["gradient"] == 7.5


def test_a_batch_mixes_kea_folders_and_tnt_files_in_time_order(tmp_path):
    from nmrdash import runner
    shutil.copytree(DATA / "kea_cpmg_snr100", tmp_path / "kea_a")
    (tmp_path / "sub").mkdir()
    shutil.copy(TECMAG_CPMG_SNR100, tmp_path / "sub" / "tec_b.tnt")
    shutil.copy(TECMAG_CPMG_CLEAN, tmp_path / "sub" / "tec_a.tnt")
    shutil.copy(TECMAG_T1_SNR100, tmp_path / "sub" / "tec_t1.tnt")
    assert [p.name for p in runner.discover(tmp_path)] == ["kea_a", "tec_a.tnt", "tec_b.tnt", "tec_t1.tnt"]
    rows, bdir = runner.run_batch(tmp_path, _fast(), quiet=True)
    by = {r["dataset"]: r for r in rows}
    assert set(by) == {"kea_a", "sub/tec_a.tnt", "sub/tec_b.tnt", "sub/tec_t1.tnt"} and all(r["status"] == "ok" for r in rows)
    assert by["sub/tec_a.tnt"]["timestamp_local"] == "2026-01-15T10:30:00" and by["sub/tec_b.tnt"]["kind"] == "cpmg" and by["sub/tec_t1.tnt"]["kind"] == "t1"
    assert (bdir / "batch_summary.csv").is_file()


def test_a_declined_tnt_is_skipped_with_its_reason_in_a_batch(tmp_path):
    from nmrdash import runner
    write_tnt(tmp_path / "diff.tnt", np.ones((5, 64), complex), points_per_window=8, dwell_s=2e-6, sequence_file="SEDiff_test", variables={"nrEchoes": "8", "tE": "100u"}, locv=("tE",))
    row = runner.run_dataset(tmp_path / "diff.tnt", _fast())
    assert row["status"] == "skipped" and "diffusion series" in row["reason"]


def test_a_tnt_in_a_folder_is_one_dataset_for_the_page_batch(tmp_path):
    """The page stages one dataset's files in a folder; a folder holding one .tnt is that dataset."""
    import base64
    from nmrdash import webbatch
    settings = {"kind": "cpmg", "echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False}, "ilt": {"enabled": False}}
    webbatch.start({"settings": settings, "options": {"mcmc": {"enabled": False}, "bootstrap": {"enabled": False}}, "style": {}, "kind": "cpmg", "root_name": "demo",
                    "output": {"figure_format": "png", "dpi": 50, "samples": False}})
    import os
    os.makedirs(f"{webbatch.ROOT}/0")
    shutil.copy(TECMAG_CPMG_SNR100, f"{webbatch.ROOT}/0/{TECMAG_CPMG_SNR100.name}")
    out = webbatch.run_one({"i": 0, "label": "tec/1", "timestamp_ms": 1_700_000_000_000})
    assert out["status"] == "ok" and "provenance.json" in out["files"]
    prov = json.loads(base64.b64decode(out["files"]["provenance.json"]))
    assert [i["name"] for i in prov["inputs"]] == [TECMAG_CPMG_SNR100.name] and prov["inputs"][0]["modified_utc"].startswith("2026-01-15")
    webbatch.finish()


# ── options added at Tyler's request ────────────────────────────────────────

def test_t1_can_use_the_first_echo_alone_instead_of_the_echo_train_sum():
    r = load(TECMAG_T1_CLEAN)
    assert r["raw"]["has_echoes"]
    both = {sel: process({**T1_FIT, "t1_amplitude": sel}) for sel in ("sum", "first")}
    for res in both.values():
        assert res["multiexp"]["selected"]["components"][0]["T2"] == pytest.approx(TRUTH["tecmag_t1"]["T1_s"], rel=1e-3)
    ys = {k: np.array(v["decay"]["y"]) for k, v in both.items()}
    assert ys["first"] == pytest.approx(_STATE["t1"]["echoes"][:, 0]) and ys["sum"] == pytest.approx(_STATE["t1"]["echoes"].sum(axis=1))
    assert not np.allclose(ys["first"], ys["sum"]) and process(T1_FIT)["settings"].get("t1_amplitude", "sum") == "sum"            # sum is the default
    with pytest.raises(ValueError, match="t1_amplitude"):
        process({**T1_FIT, "t1_amplitude": "last"})


def test_the_first_echo_amplitude_needs_an_echo_train_per_delay():
    r = load(DATA / "kea_t1_clean")
    assert r["raw"]["has_echoes"] is False
    with pytest.raises(ValueError, match="only available for data with an echo train"):
        process({**T1_FIT, "t1_amplitude": "first"})


def test_flip_z_mirrors_the_position_axis_and_the_profile(tmp_path):
    tecmag_variant(tmp_path / "t.tnt", "cpmg", layers_um=((216.0, 431.0),), T2_s=(0.02,), fraction=(1.0,))
    load(tmp_path / "t.tnt")
    ft = {"nucleus": "1H", "G": 6.59, "zero_fill": 4, "z_min": None, "z_max": None}
    a = process({**CPMG_FIT, "echo": "ft", "ft": ft})["ft"]
    b = process({**CPMG_FIT, "echo": "ft", "ft": {**ft, "flip_z": True}})["ft"]
    za, pa, zb, pb = (np.array(x) for x in (a["z_um"], a["profile"], b["z_um"], b["profile"]))
    assert np.allclose(zb, -za[::-1]) and np.allclose(pb, pa[::-1])
    assert (za * pa).sum() > 0 > (zb * pb).sum()                                                    # the layer is on the + side, then on the - side


def test_the_new_options_travel_through_the_params_file():
    from nmrdash import params as P
    p = P.defaults()
    assert p["t1"]["amplitude"] == "sum" and p["ft"]["flip_z"] is False
    p["t1"]["amplitude"], p["ft"]["flip_z"], p["processing"]["echo"] = "first", True, "ft"
    assert P.to_settings_t1(p)[0]["t1_amplitude"] == "first" and P.to_settings(p, [1e-3, 2e-3])[0]["ft"]["flip_z"] is True
    assert P.from_settings(P.to_settings_t1(p)[0], P.to_settings_t1(p)[1])["t1"]["amplitude"] == "first"
    assert P.from_settings(*P.to_settings(p, [1e-3, 2e-3]))["ft"]["flip_z"] is True
    with pytest.raises(ValueError, match="t1.amplitude"):
        P.validate({"t1": {"amplitude": "second"}})
