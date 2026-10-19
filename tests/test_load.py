import shutil
from pathlib import Path

import pytest

from nmrdash.pipeline import load

DATA = Path(__file__).resolve().parent.parent / "test_data"
CPMG = sorted(p.parent for p in DATA.rglob("data.2d") if (p.parent / "acqu.par").exists())
T2D = [DATA / "M1115_rt_30phr" / "1", DATA / "S3115_rt_100phr" / "1"]
T1 = [DATA / "T1SAT" / "1", DATA / "T1SAT_dopedwater" / "1", DATA / "T1SAT_dopedwater" / "2"]


@pytest.mark.parametrize("d", CPMG, ids=lambda p: str(p.relative_to(DATA)))
def test_cpmg_loads(d):
    r = load(d)
    assert r["is_cpmg"], r["reason"]
    raw = r["raw"]
    assert len(raw["real"]) == raw["n_points"]
    assert len(raw["real"][0]) == raw["n_echoes"] == len(raw["summed"]) == len(raw["echo_time_s"])
    labels = [row["label"] for row in r["display"]]
    assert "Echo time" in labels and "Pulse length" in labels and "File timestamp" in labels


@pytest.mark.parametrize("d", T2D, ids=lambda p: str(p.relative_to(DATA)))
def test_t2d_is_read(d):
    r = load(d)
    assert r["kind"] == "t2d" and r["supported"] and not r["is_cpmg"] and "T₂–D" in r["analysis"]
    raw = r["raw"]
    assert raw["n_echoes"] == len(raw["echo_time_s"]) and raw["n_tau"] == len(raw["b_vec"]) == len(raw["data"]) and len(raw["data"][0]) == raw["n_echoes"]
    assert raw["noise_sigma"] > 0 and r["defaults"]["t2_steps"] == 50 and r["defaults"]["d_max"] == 2.2e-9


def test_a_folder_with_a_sequence_nmrdash_does_not_know_is_declined(tmp_path):
    d = tmp_path / "x"
    shutil.copytree(T2D[0], d)
    (d / "acqu.par").write_text((d / "acqu.par").read_text().replace('experiment = "SSE-T2"', 'experiment = "Diffusion"'))
    r = load(d)
    assert not r["supported"] and r["kind"] is None and "Diffusion" in r["reason"] and r["raw"] is None


def test_a_t2d_folder_missing_its_data_is_declined_with_a_reason(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    shutil.copy(T2D[0] / "acqu.par", d / "acqu.par")
    r = load(d)
    assert not r["supported"] and "data2D" in r["reason"]


@pytest.mark.parametrize("d", T1, ids=lambda p: str(p.relative_to(DATA)))
def test_t1sat_is_read_from_data_dat(d):
    r = load(d)
    assert r["kind"] == "t1" and r["supported"] and not r["is_cpmg"] and "saturation recovery" in r["analysis"]
    raw = r["raw"]
    assert raw["n_points"] == len(raw["t"]) == len(raw["y"]) >= 10 and raw["t"] == sorted(raw["t"])
    assert r["defaults"]["t1_offset"] is True and r["defaults"]["t1_components"] == 1
    assert any(row["label"] == "Number of T1 delays" for row in r["display"]) and r["display"][0] == {"label": "Sequence", "value": "T1Sat"}


def test_display_pulse_row():
    r = load(DATA / "CPMG_EtGly" / "1")
    row = next(x for x in r["display"] if x["label"] == "Pulse length")
    assert row["value"] == "2.25 µs (90°/180° amp: -10/-4 dB)"


def test_sequence_and_timestamp_are_base_rows():
    r = load(DATA / "CPMG_EtGly" / "1", 1760000000000)
    assert [x["label"] for x in r["display"]][:2] == ["Sequence", "File timestamp"]
    assert r["display"][0]["value"] == "CPMG"
    assert r["display"][1]["value"].startswith("2025-")


def test_sequence_shown_for_non_cpmg():
    r = load(DATA / "T1SAT" / "1")
    assert r["display"][0] == {"label": "Sequence", "value": "T1Sat"}
