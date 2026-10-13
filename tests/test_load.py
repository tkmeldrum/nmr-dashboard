from pathlib import Path

import pytest

from nmrdash.pipeline import load

DATA = Path(__file__).resolve().parent.parent / "test_data"
CPMG = sorted(p.parent for p in DATA.rglob("data.2d") if (p.parent / "acqu.par").exists())
NON_CPMG = [DATA / "T1SAT" / "1", DATA / "T1SAT_dopedwater" / "1"]


@pytest.mark.parametrize("d", CPMG, ids=lambda p: str(p.relative_to(DATA)))
def test_cpmg_loads(d):
    r = load(d)
    assert r["is_cpmg"], r["reason"]
    raw = r["raw"]
    assert len(raw["real"]) == raw["n_points"]
    assert len(raw["real"][0]) == raw["n_echoes"] == len(raw["summed"]) == len(raw["echo_time_s"])
    labels = [row["label"] for row in r["display"]]
    assert "Echo time" in labels and "Pulse length" in labels and "File timestamp" in labels


@pytest.mark.parametrize("d", NON_CPMG, ids=lambda p: str(p.relative_to(DATA)))
def test_non_cpmg_flagged(d):
    r = load(d)
    assert not r["is_cpmg"] and r["raw"] is None and r["reason"]


def test_display_pulse_row():
    r = load(DATA / "CPMG_EtGly" / "1")
    row = next(x for x in r["display"] if x["label"] == "Pulse length")
    assert row["value"] == "2.25 µs (90°/180° amp: -10 / -4)"
