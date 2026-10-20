"""Tecmag datasets through the dashboard: what each sequence becomes, that the known truth is recovered, and every way a file can be unsuitable."""
import json
import shutil
from datetime import datetime

import numpy as np
import pytest
from sample_data import (CPMG_NOISY, DATA, TECMAG_CPMG_CLEAN, TECMAG_CPMG_NOISY, TECMAG_T1_CLEAN, TECMAG_T1_NOISY, TECMAG_T2D_CLEAN, TECMAG_T2D_NOISY, TRUTH,
                         tecmag_variant)
from tnt_writer import write_tnt

from nmrdash import physics, tecmag
from nmrdash.pipeline import _STATE, get_provenance, load, process

CPMG_FIT = {"echo": "integrate", "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": False}, "ilt": {"enabled": False}}
T1_FIT = {"kind": "t1", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False}}
T2D_FIT = {"kind": "t2d", "t2d": {"t2_steps": 24, "d_steps": 15, "gradient": 6.7}, "t2d_model": {"enabled": True}}


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
    (TECMAG_CPMG_NOISY, "cpmg", {"echoTime": 150.0, "nrEchoes": 256, "nrPnts": 32, "dwellTime": 2.0, "pulseLength": 6.0, "repTime": 1000.0, "nrScans": 64}),
    (TECMAG_T1_NOISY, "t1", {"echoTime": 150.0, "nrEchoes": 16, "nrPnts": 16, "nrPntsT1": 16, "dwellTime": 2.0, "repTime": 1000.0}),
    (TECMAG_T2D_NOISY, "t2d", {"echoTime": 60.0, "nrEchoes": 300, "nrPnts": 20, "nrPntsTau": 16, "DELTA": 10.0, "dwellTime": 2.0, "nrScans": 128}),
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
    assert load(TECMAG_CPMG_NOISY)["defaults"]["G"] == 6.7 and load(TECMAG_T2D_NOISY)["defaults"]["gradient"] == 6.7
    assert load(DATA / "kea_cpmg_noisy")["defaults"]["G"] == 23.67 and load(DATA / "kea_t2d_noisy")["defaults"]["gradient"] == 23.67


def test_the_nucleus_comes_from_the_file_header():
    r = load(TECMAG_CPMG_NOISY)
    assert r["defaults"]["nucleus"] == "1H" and "from the file header" in r["defaults"]["nucleus_reason"]
    assert "from the file header" not in load(CPMG_NOISY)["defaults"]["nucleus_reason"]               # Kea: from b1Freq


def test_the_timestamp_is_the_experiment_date_in_the_file_not_the_file_time():
    load(TECMAG_CPMG_NOISY, 1_700_000_000_000)                                                      # a file time the browser reports: not used
    assert datetime.fromtimestamp(_STATE["source"]["timestamp_ms"] / 1000) == datetime(2026, 1, 15, 10, 30, 0)


# ── the known truth comes back ───────────────────────────────────────────────

@pytest.mark.parametrize("path,tol", [(TECMAG_CPMG_CLEAN, 0.01), (TECMAG_CPMG_NOISY, 0.1)])
def test_cpmg_recovers_both_t2_by_integration_and_by_ft(path, tol):
    c = TRUTH["tecmag_cpmg"]
    load(path)
    for cfg in (CPMG_FIT, {**CPMG_FIT, "echo": "ft", "ft": {"nucleus": "1H", "G": 6.7, "zero_fill": 4, "z_min": None, "z_max": None}}):
        comps = process(cfg)["multiexp"]["selected"]["components"]
        assert [x["T2"] for x in comps] == pytest.approx(c["T2_s"], rel=tol)
        assert [x["fraction"] for x in comps] == pytest.approx(c["fraction"], abs=tol)


def test_cpmg_noise_is_measured_from_the_imaginary_channel():
    load(TECMAG_CPMG_NOISY)
    sigma = process(CPMG_FIT)["decay"]["noise_sigma"]
    assert sigma == pytest.approx(TRUTH["tecmag_cpmg"]["noise_sigma"] * np.sqrt(TRUTH["tecmag_cpmg"]["n_points"]), rel=0.2)       # the sum of 32 points per echo


@pytest.mark.parametrize("path,tol", [(TECMAG_T1_CLEAN, 1e-3), (TECMAG_T1_NOISY, 0.03)])
def test_t1_is_recovered(path, tol):
    load(path)
    c = process(T1_FIT)["multiexp"]["selected"]["components"][0]
    assert c["T2"] == pytest.approx(TRUTH["tecmag_t1"]["T1_s"], rel=tol)                                # (the field is called T2 for every decay; for T1 it is T1)


def test_t1_delays_are_the_table_plus_the_pulse_width():
    r = load(TECMAG_T1_CLEAN)
    c = TRUTH["tecmag_t1"]
    want = physics.geometric_grid(c["t_min_s"], c["t_max_s"], c["n_delays"])
    assert r["raw"]["t"] == pytest.approx(list(want), abs=1.5e-6)                                           # the table is rounded to a microsecond


@pytest.mark.parametrize("path,chi_max", [(TECMAG_T2D_CLEAN, 5.0), (TECMAG_T2D_NOISY, 1.15)])
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
    load(TECMAG_T2D_NOISY)
    from nmrdash.pipeline import t2d_axes
    a, b = t2d_axes({"t2d": {"gradient": 6.7}}), t2d_axes({"t2d": {"gradient": 13.4}})
    assert np.allclose(np.array(b["b_vec"]) / np.array(a["b_vec"]), 4.0) and a["b_check"]["compared"] == []              # no spectrometer files to compare with


def test_a_cpmg_profile_has_the_same_orientation_as_the_kea_equivalent(tmp_path):
    """Both formats give the same z for a layer on the + side: the Tecmag data are stored conjugated and read back conjugated (as the lab's MATLAB reader does)."""
    layer = {"T2_s": (0.02,), "fraction": (1.0,)}
    from make_synthetic_data import write_cpmg
    write_cpmg(tmp_path / "kea", True, np.random.default_rng(5), "k", layers_um=((60.0, 120.0),), **layer)
    tecmag_variant(tmp_path / "t.tnt", "cpmg", layers_um=((212.0, 424.0),), **layer)
    centroids = []
    for path, g in ((tmp_path / "kea", 23.67), (tmp_path / "t.tnt", 6.7)):
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
    load(TECMAG_CPMG_NOISY)
    process(CPMG_FIT)
    rec = get_provenance({"dataset": "sample: tecmag_cpmg_noisy"}, {})["record"]
    assert [f["name"] for f in rec["inputs"]] == ["tecmag_cpmg_noisy.tnt"] and rec["acquisition"].get("nrEchoes") == 256
    json.dumps(rec)
