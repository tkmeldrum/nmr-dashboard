"""T2-D (sequence SSE-T2): the existing 2D ILT, units, loading, peaks, the pipeline, the script and provenance."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nmrdash import ilt2d, params as P, runner, t2d
from nmrdash.kea_io import read_kea_params
from nmrdash.pipeline import _STATE, boot_start, get_provenance, load, mcmc_start, process, t2d_estimate

DATA = Path(__file__).resolve().parent.parent / "test_data"
M1115, S3115 = DATA / "M1115_rt_30phr" / "1", DATA / "S3115_rt_100phr" / "1"
SMALL = {"t2_steps": 30, "d_steps": 12}                                        # keeps the solves to about a second


def synthetic(noise=0.4, seed=0):
    """Two populations on the sample's own echo / b axes, both inside what those b-values can resolve (about 3.5e-13 .. 7.8e-10 m^2/s):
    (T2 8 ms, D 2e-11) with amplitude 100 and (T2 1 ms, D 2e-10) with amplitude 60."""
    L = t2d.load(M1115, read_kea_params(M1115 / "acqu.par"))
    e, b = L["echo_s"], L["b_vec"]
    data = sum(a * np.exp(-e[:, None] / T2) * np.exp(-b[None, :] * D * 1e9) for a, T2, D in ((100.0, 8e-3, 2e-11), (60.0, 1e-3, 2e-10)))
    return e, b, data + np.random.default_rng(seed).normal(0, noise, data.shape)


# ── loading ──────────────────────────────────────────────────────────────────

def test_loader_reads_axes_data_and_a_measured_noise_channel():
    L = t2d.load(M1115, read_kea_params(M1115 / "acqu.par"))
    assert L["data"].shape == (1000, 25) and L["complex"] and L["noise"].shape == L["data"].shape
    assert L["echo_s"][0] == pytest.approx(44e-6) and L["echo_s"][-1] == pytest.approx(44e-3)
    assert L["b_vec"][0] == pytest.approx(6.40474) and np.all(np.diff(L["b_vec"]) > 0)
    assert t2d.noise_sigma(L["noise"]) == pytest.approx(10.9, rel=0.05)
    assert abs(L["data"].mean()) > 5 * abs(L["noise"].mean())                    # the signal is in the real channel, the noise in the imaginary one


def test_loader_explains_what_is_wrong(tmp_path):
    d = tmp_path / "x"
    shutil.copytree(M1115, d)
    p = read_kea_params(d / "acqu.par")
    (d / "DiffAxis.dat").write_text("1\n2\n3\n")
    with pytest.raises(ValueError, match="3 b-values but acqu.par says 25"):
        t2d.load(d, p)
    (d / "DiffAxis.dat").unlink()
    (d / "data.dat").unlink()
    with pytest.raises(ValueError, match="b-values"):
        t2d.load(d, p)


# ── the 2D inversion ─────────────────────────────────────────────────────────

def test_units_d_in_and_out_in_m2_per_s_and_the_grid_is_log_spaced():
    e, b, data = synthetic()
    r = ilt2d.run(data, e, b, 1e-5, 1.0, 20, 1e-12, 1e-8, 10, 1e9)
    assert r["D"][0] == pytest.approx(1e-12) and r["D"][-1] == pytest.approx(1e-8) and r["T2"][0] == pytest.approx(1e-5) and r["T2"][-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(np.log(r["D"])), np.diff(np.log(r["D"]))[0])
    # the solver itself runs on D * 1e9, exactly as pythonic does
    spec, T2, Ds, *_ = ilt2d.run_2dilt(data, e, b, {"T2minmax": [1e-5, 1.0], "T2_steps": 20, "Dminmax": [1e-3, 10.0], "D_steps": 10, "alpha": 1e9})
    np.testing.assert_allclose(r["spectrum"], spec)
    np.testing.assert_allclose(r["D"], Ds * 1e-9)


def test_known_populations_inside_the_sensitive_window_are_recovered():
    """Restricted to the D range these b-values can resolve, the slow population is recovered well and the fast one roughly:
    the 2D inversion is ill-conditioned (exponentials in b overlap), so D of the fast population can be off by ~2x and its share by ~20%."""
    e, b, data = synthetic()
    lo, hi = ilt2d.sensitive_d_window(b)
    r = ilt2d.run(data, e, b, 1e-5, 1.0, 40, lo, hi, 20, 1e8, sigma=0.4)
    big, small = r["peaks"][0], r["peaks"][1]
    assert big["T2_mean"] == pytest.approx(8e-3, rel=0.3) and big["D_mean"] == pytest.approx(2e-11, rel=0.4)           # slow population: good
    assert 1e-3 / 2 < small["T2_mean"] < 1e-3 * 2 and 2e-10 / 3 < small["D_mean"] < 2e-10 * 3                          # fast population: within a factor
    assert 0.4 < big["fraction"] < 0.8 and big["fraction"] > small["fraction"]                                          # truth is 62.5% / 37.5%
    assert big["constrained"] and small["constrained"] and r["fraction_in_sensitive_window"] == pytest.approx(1.0)
    assert r["rmse_over_amp"] < 0.03 and 0.5 < r["chi2_per_point"] < 2.0
    assert r["projection_T2"].shape == (40,) and r["projection_D"].shape == (20,)


def test_a_range_wider_than_the_data_can_resolve_lets_regularisation_fill_the_rest():
    e, b, data = synthetic()
    wide = ilt2d.run(data, e, b, 1e-5, 1.0, 40, 1e-12, 1e-8, 20, 1e9, sigma=0.4)
    assert wide["fraction_in_sensitive_window"] < 0.85                          # a good part of the "signal" sits where the data cannot see
    assert any(not p["constrained"] for p in wide["peaks"])


def test_the_real_sample_fits_to_the_noise_level():
    L = t2d.load(M1115, read_kea_params(M1115 / "acqu.par"))
    sigma = t2d.noise_sigma(L["noise"])
    r = ilt2d.run(L["data"], L["echo_s"], L["b_vec"], 1e-5, 1.0, 30, 1e-12, 1e-8, 25, 1e9, sigma=sigma)
    assert 0.7 < r["chi2_per_point"] < 1.6 and r["rmse"] == pytest.approx(sigma, rel=0.25)


def test_peak_finder_separates_blobs_and_orders_by_size():
    T2, D = np.geomspace(1e-5, 1, 40), np.geomspace(1e-12, 1e-8, 20)
    spec = np.zeros((20, 40))
    spec[3:6, 5:8] = 4.0
    spec[14:17, 25:29] = 9.0
    peaks = ilt2d.peaks_2d(spec, T2, D)
    assert len(peaks) == 2 and peaks[0]["fraction"] > peaks[1]["fraction"] and sum(p["fraction"] for p in peaks) == pytest.approx(1)
    assert peaks[0]["T2_mean"] == pytest.approx(T2[25:29].prod() ** (1 / 4), rel=1e-6) and peaks[0]["D_mean"] == pytest.approx(D[14:17].prod() ** (1 / 3), rel=1e-6)
    assert ilt2d.peaks_2d(np.zeros((5, 5)), T2[:5], D[:5]) == []


# ── through the pipeline ─────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [M1115, S3115], ids=lambda p: p.parent.name)
def test_both_samples_process_and_are_json_safe(d):
    load(d)
    res = process({"kind": "t2d", "t2d": SMALL})
    json.dumps(res, allow_nan=False)
    r = res["t2d"]
    assert res["mode"] == "t2d" and np.array(r["spectrum"]).shape == (12, 30) and r["peaks"] and r["rmse_over_amp"] < 0.15
    assert len(r["echo_s"]) == len(res["decay"]["t"]) and np.array(r["pred"]).shape == (12 if False else 25, 1000)


def test_exclude_first_drops_echoes_from_the_inversion():
    load(M1115)
    res = process({"kind": "t2d", "exclude_first": 5, "t2d": SMALL})
    assert res["decay"]["skip"] == 5 and len(res["t2d"]["echo_s"]) == 995 and np.array(res["t2d"]["resid"]).shape == (25, 995)
    with pytest.raises(ValueError, match="at least 8"):
        process({"kind": "t2d", "exclude_first": 995, "t2d": SMALL})


def test_settings_are_validated():
    load(M1115)
    with pytest.raises(ValueError, match="minimum of each range"):
        process({"kind": "t2d", "t2d": {"t2_min": 1.0, "t2_max": 1e-5}})
    with pytest.raises(ValueError, match="d_min must be positive"):
        process({"kind": "t2d", "t2d": {"d_min": 0}})


def test_the_time_estimate_grows_with_the_grid_and_is_the_right_order():
    import time
    load(M1115)
    small, big = t2d_estimate({"t2d": {"t2_steps": 20, "d_steps": 10}}), t2d_estimate({"t2d": {"t2_steps": 50, "d_steps": 25}})
    assert big["seconds"] > 10 * small["seconds"] and big["unknowns"] == 1250 and small["n_T2"] == 20
    t0 = time.perf_counter()
    process({"kind": "t2d", "t2d": {"t2_steps": 40, "d_steps": 25}})
    actual = time.perf_counter() - t0
    est = t2d_estimate({"t2d": {"t2_steps": 40, "d_steps": 25}})["seconds"]
    assert 0.3 < est / actual < 3.5, (est, actual)


def test_mcmc_and_bootstrap_say_they_are_not_available_for_t2d():
    load(M1115)
    process({"kind": "t2d", "t2d": SMALL})
    for call in (lambda: mcmc_start({}), lambda: boot_start({})):
        with pytest.raises(ValueError, match="not available for T2-D"):
            call()


def test_provenance_records_every_t2d_input_file():
    load(M1115)
    process({"kind": "t2d", "t2d": SMALL})
    rec = get_provenance({"dataset": "sample: M1115"}, {})["record"]
    names = {f["name"] for f in rec["inputs"]}
    assert {"acqu.par", "data2D_Cp.2d", "data.dat"} <= names and rec["settings"]["kind"] == "t2d"          # these samples have no DiffAxis.dat / T2Axis.dat
    assert rec["acquisition"]["nrPntsTau"] == 25 and rec["acquisition"]["DELTA"] == 5


# ── params file, script and batch ────────────────────────────────────────────

def test_params_t2d_section_round_trips():
    p = P.defaults()
    assert p["t2d"]["d_min"] == 1e-12 and p["t2d"]["d_max"] == 1e-8 and p["t2d"]["t2_min"] == 1e-5 and p["t2d"]["t2_max"] == 1.0 and p["t2d"]["alpha"] == 1e9
    s, o = P.to_settings_t2d(p)
    assert s == {"kind": "t2d", "exclude_first": 0, "t2d": p["t2d"]} and o == {}
    p["t2d"]["t2_steps"] = 60
    assert P.from_settings(P.to_settings_t2d(p)[0])["t2d"]["t2_steps"] == 60
    import tomllib
    assert tomllib.loads(P.dump_toml(p))["t2d"]["t2_steps"] == 60
    with pytest.raises(ValueError, match="minimum of each range"):
        P.validate({"t2d": {"d_min": 1e-6, "d_max": 1e-9}})


def test_run_dataset_writes_the_t2d_files(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(M1115, d)
    p = P.defaults()
    p["t2d"].update(SMALL)
    p["mcmc"]["enabled"] = True
    row = runner.run_dataset(d, p, command="nmrdash run x")
    assert row["status"] == "ok" and row["kind"] == "t2d", row.get("reason")
    out = Path(row["output_dir"])
    names = {x.name for x in out.iterdir()}
    assert {"t2d_spectrum.csv", "t2d_peaks.csv", "t2d_statistics.csv", "t2d_data_fit_residual.csv", "t2d_spectrum.png", "t2d_fit.png", "summary.md", "provenance.json"} <= names
    assert "decay.csv" not in names
    peaks = list(csv.DictReader(open(out / "t2d_peaks.csv")))
    assert peaks and float(peaks[0]["D_mean_m2s"]) > 0 and 0.8 < sum(float(x["fraction"]) for x in peaks) <= 1.0001      # weak signal under the 10% threshold is in no region
    assert {"constrained_by_data"} <= set(peaks[0])
    spec = list(csv.reader(open(out / "t2d_spectrum.csv")))
    assert len(spec) == 1 + 12 and len(spec[0]) == 1 + 30
    assert any("not available for T2-D" in w for w in row["warnings"])
    md = (out / "summary.md").read_text()
    assert "T₂–D regions" in md and "diffusion delays" in md and "?" not in md.split("## T₂–D regions")[0]


def test_a_mixed_batch_of_all_three_sequences(tmp_path):
    root = tmp_path / "all"
    for name, src in (("cpmg", "CPMG_EtGly/1"), ("t1", "T1SAT/1"), ("t2d", "M1115_rt_30phr/1")):
        shutil.copytree(DATA / src, root / name)
    p = P.defaults()
    p["multiexp"]["compare"] = False
    p["ilt"]["max_grid_points"] = 40
    p["t2d"].update(SMALL)
    rows, b = runner.run_batch(root, p, quiet=True)
    table = {r["dataset"]: r for r in csv.DictReader(open(b / "batch_summary.csv"))}
    assert {k: v["kind"] for k, v in table.items()} == {"cpmg": "cpmg", "t1": "t1", "t2d": "t2d"} and all(v["status"] == "ok" for v in table.values())
    assert float(table["t2d"]["t2d_peak1_D_m2s"]) > 0 and 0 < float(table["t2d"]["t2d_peak1_fraction"]) <= 1
    assert table["cpmg"]["t2d_peak1_T2_s"] == "" and table["t2d"]["T2_1_s"] == ""


# ── what the data can and cannot tell apart ──────────────────────────────────

def test_the_sensitive_diffusion_window_follows_the_b_values():
    lo, hi = ilt2d.sensitive_d_window([6.40474, 2884.67])                     # the M1115 axis, in units of 1e9 s/m^2
    assert lo == pytest.approx(1 / 2.88467e12) and hi == pytest.approx(5 / 6.40474e9)
    lo2, hi2 = ilt2d.sensitive_d_window([64.0474, 28846.7])                    # ten times larger b: sees ten times smaller D
    assert lo2 == pytest.approx(lo / 10) and hi2 == pytest.approx(hi / 10)


def test_regions_outside_the_window_are_flagged_and_the_samples_show_why_it_matters():
    for d in (M1115, S3115):
        load(d)
        r = process({"kind": "t2d", "t2d": {"t2_steps": 30, "d_steps": 25}})["t2d"]        # the full 25-point D grid
        lo, hi = r["d_sensitive"]
        assert all(p["constrained"] == (lo <= p["D_mean"] <= hi) for p in r["peaks"])
        assert not r["peaks"][0]["constrained"]                                # the biggest region is regularisation filling D the data cannot see
        assert r["fraction_in_sensitive_window"] < 0.5
        assert all(0 <= p["fraction_in_window"] <= 1.0001 for p in r["peaks"])        # the part of each region inside the window, never more than all of it
        assert sum(p["fraction_in_window"] for p in r["peaks"]) <= 1.0001


def test_summary_warns_when_most_of_the_spectrum_is_unconstrained(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(M1115, d)
    p = P.defaults()
    p["t2d"].update(SMALL)
    out = Path(runner.run_dataset(d, p)["output_dir"])
    md = (out / "summary.md").read_text()
    assert "of the spectrum lies where the data can constrain it" in md and "**no**" in md


def test_auto_d_range_uses_exactly_what_the_b_values_can_resolve():
    load(M1115)
    r = process({"kind": "t2d", "t2d": {**SMALL, "d_min": "auto", "d_max": "auto"}})["t2d"]
    lo, hi = ilt2d.sensitive_d_window(_STATE["t2d"]["b_vec"])
    assert r["d_min"] == pytest.approx(lo, rel=0.06) and r["d_max"] == pytest.approx(hi, rel=0.06)
    assert r["fraction_in_sensitive_window"] > 0.95
    p = P.validate({"t2d": {"d_min": "auto", "d_max": "auto"}})
    assert P.to_settings_t2d(p)[0]["t2d"]["d_min"] == "auto"
    with pytest.raises(ValueError, match="must both be numbers"):
        P.validate({"t2d": {"d_min": "auto"}})
