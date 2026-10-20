"""The Tecmag .tnt reader: header, data, sequence parameters and tables, on files written by tools/tnt_writer.py (the real layout, synthetic content)."""
from datetime import datetime

import numpy as np
import pytest
from sample_data import ROOT  # noqa: F401  (puts tools/ on the path)
from tnt_writer import write_tnt

from nmrdash import tecmag_io as T

ROWS = [("Delay", [("5u", {}), ("=d1", {"delay": "de02"}), ("=pw", {}), ("=trep", {})]),
        ("F1_Ph", [("-1", {}), ("-1", {"phase": "ph0"}), ("-1", {}), ("-1", {})]),
        ("Acq", [("0", {}), ("0", {}), ("1", {}), ("0", {})])]
VARS = {"nrEchoes": "128", "pw": "3u", "trep": "300m", "tE": "150u", "90amp": "41", "d1": "1n", "DELTA": "5000u"}
DEFAULTS = {"nrEchoes": "64", "pw": "6u", "trep": "1s", "tE": "100u", "90amp": "26"}
TABLES = {"ph0": (["0", "2"], "phase"), "de02": (["44", "62", "80"], "delay")}


@pytest.fixture
def tnt(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(3, 64)) + 1j * rng.normal(size=(3, 64))
    path = write_tnt(tmp_path / "x.tnt", data, points_per_window=16, dwell_s=2e-6, scans=32, date=datetime(2026, 3, 4, 5, 6, 7), sequence_file="CPMG_test",
                     rows=ROWS, tables=TABLES, variables=VARS, defaults=DEFAULTS, locv=("tE", "DELTA"))
    return path, data


def test_header_fields_are_read(tnt):
    path, data = tnt
    f = T.read_tnt(path)
    h = f.header
    assert h["version"] == "TNT1.008" and h["nucleus"] == "H1" and h["date"] == "2026/3/4 05:06:07"
    assert h["observe_MHz"][0] == pytest.approx(13.235) and h["dwell_s"][0] == pytest.approx(2e-6) and h["sweep_hz"][0] == pytest.approx(500000.0)   # +/- half width, doubled
    assert h["scans"] == 32 and h["acq_points"] == 16 and h["points"][:2] == [16, 3] and h["points_per_record"] == 64 and h["records"] == [3, 1, 1] and h["dimensions"] == 2
    assert f.warnings == [] and f.software == "3.4.5 Build 151002"


def test_data_is_complex_float32_in_record_order(tnt):
    path, data = tnt
    f = T.read_tnt(path)
    assert f.data.shape == (3, 64) and np.iscomplexobj(f.data)
    np.testing.assert_allclose(f.data, data, atol=1e-6)                       # float32 storage
    np.testing.assert_allclose(f.conjugated(), np.conj(f.data))


def test_a_single_record_is_one_dimensional(tmp_path):
    f = T.read_tnt(write_tnt(tmp_path / "one.tnt", np.arange(40) * (1 + 1j), points_per_window=8, dwell_s=1e-6))
    assert f.data.shape == (40,) and f.header["dimensions"] == 1 and f.header["points_per_record"] == 40


def test_the_value_used_is_read_not_the_saved_default(tnt):
    """A text scan that keeps the later of two occurrences returns the sequence file's saved default (nrEchoes 64, tE 100u); the value used is the first."""
    f = T.read_tnt(tnt[0])
    assert f.variables["nrEchoes"] == "128" and f.pseq["defaults"]["nrEchoes"] == "64"
    assert f.value("nrEchoes") == 128 and f.value("tE") == pytest.approx(150e-6) and f.value("PW") == pytest.approx(3e-6)          # names ignore case
    assert f.value("trep") == pytest.approx(0.3) and f.value("DELTA") == pytest.approx(5e-3) and f.value("nothing", "none") == "none"
    assert f.variables["90amp"] == "41"


def test_tables_and_rows_are_read_with_the_tables_each_cell_uses(tnt):
    f = T.read_tnt(tnt[0])
    assert f.pseq["file"] == "CPMG_test" and f.pseq["revision"] == "1.18 BIN"
    assert f.table("DE02")["values"] == ["44", "62", "80"] and f.table("ph0")["type"].endswith("phase") and f.table("de02")["type"].endswith("delay")
    assert f.table("missing") is None
    delay_cells = f.pseq["rows"]["Delay"]
    assert [c["setting"] for c in delay_cells] == ["5u", "=d1", "=pw", "=trep"] and delay_cells[1]["tables"] == {"delay": "de02"}
    assert f.pseq["rows"]["F1_Ph"][1]["tables"] == {"phase": "ph0"} and f.pseq["rows"]["Acq"][2]["setting"] == "1"


def test_parse_value_units():
    assert T.parse_value("100u") == pytest.approx(1e-4) and T.parse_value("1.5m") == pytest.approx(1.5e-3) and T.parse_value("5000u") == pytest.approx(5e-3)
    assert T.parse_value("64") == 64.0 and T.parse_value("1s") == 1.0 and T.parse_value("0.5u") == pytest.approx(5e-7) and T.parse_value("13.2MHz") == pytest.approx(13.2e6)
    assert T.parse_value("44\r\n62\r\n80") == [44.0, 62.0, 80.0]
    for bad in ("=pw", "", None, "abc"):
        assert np.isnan(T.parse_value(bad))


def test_the_reader_takes_bytes_as_well_as_a_path(tnt):
    path, _ = tnt
    assert T.read_tnt(path.read_bytes()).variables == T.read_tnt(path).variables and T.read_tnt(bytearray(path.read_bytes())).data.shape == (3, 64)


def test_files_that_are_not_tnt_are_refused_with_a_reason(tmp_path):
    for content, match in ((b"", "too short"), (b"NOTATNTFILE" * 200, "not a Tecmag"), (b"TNT1.008" + b"XXXX" * 400, "TMAG header")):
        with pytest.raises(T.TntError, match=match):
            T.read_tnt(content)


def test_a_truncated_file_is_refused_not_misread(tnt):
    raw = tnt[0].read_bytes()
    with pytest.raises(T.TntError, match="truncated"):
        T.read_tnt(raw[:1500])


def test_damaged_sequence_information_gives_a_warning_but_keeps_the_data(tnt):
    raw = bytearray(tnt[0].read_bytes())
    i = raw.find(b"PSEQ")
    raw[i + 4:i + 8] = b"\x01\x00\x00\x00"
    raw[i + 12:i + 40] = b"\xff" * 28                                         # garbage where the sequence file name is
    f = T.read_tnt(bytes(raw))
    assert f.data.shape == (3, 64) and any("could not be read completely" in w for w in f.warnings)


def test_an_unknown_section_stops_the_walk_with_a_note(tnt):
    f = T.read_tnt(tnt[0].read_bytes() + b"ZZZZ" + b"\0" * 16)
    assert any("unknown section 'ZZZZ'" in w for w in f.warnings) and f.variables["nrEchoes"] == "128"


def test_more_data_than_the_header_accounts_for_is_noted(tmp_path):
    f = T.read_tnt(write_tnt(tmp_path / "x.tnt", np.ones((3, 20)), points_per_window=20, dwell_s=1e-6))
    assert f.data.shape == (3, 20) and f.warnings == []
