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

from sample_data import DATA, T2D_CLEAN, T2D_SNR100, t2d_with_a_slow_population, TRUTH
SMALL = {"t2_steps": 30, "d_steps": 12}                                        # keeps the solves to about a second


def synthetic(noise=0.4, seed=0):
    """Two populations on the sample's own echo / b axes, both inside what those b-values can resolve (about 3.5e-13 .. 7.8e-10 m^2/s):
    (T2 8 ms, D 2e-11) with amplitude 100 and (T2 1 ms, D 2e-10) with amplitude 60."""
    L = t2d.load(T2D_SNR100, read_kea_params(T2D_SNR100 / "acqu.par"))
    e, b = L["echo_s"], L["b_vec"]
    data = sum(a * np.exp(-e[:, None] / T2) * np.exp(-b[None, :] * D * 1e9) for a, T2, D in ((100.0, 8e-3, 2e-11), (60.0, 1e-3, 2e-10)))
    return e, b, data + np.random.default_rng(seed).normal(0, noise, data.shape)


# ── loading ──────────────────────────────────────────────────────────────────

def test_loader_reads_axes_data_and_a_measured_noise_channel():
    L = t2d.load(T2D_SNR100, read_kea_params(T2D_SNR100 / "acqu.par"))
    c = TRUTH["t2d"]
    assert L["data"].shape == (c["n_echoes"], c["n_tau"]) and L["complex"] and L["noise"].shape == L["data"].shape
    assert L["echo_s"][0] == pytest.approx(100e-6) and L["echo_s"][-1] == pytest.approx(c["n_echoes"] * 100e-6)
    # calculated, not read: tau = 0.178 ms, Delta = 5 ms, G = 23.67 T/m (the spectrometer's file is for 23.49 T/m: (23.67 / 23.487)^2 = 1.0156 higher)
    tau0 = c["tau_ms"][0] * 1e-3
    assert L["b_vec"][0] == pytest.approx((2 * np.pi * 42.577478e6 * 23.67) ** 2 * tau0**2 * (5e-3 + 2 * tau0 / 3) * 1e-9, rel=1e-3) and np.all(np.diff(L["b_vec"]) > 0)
    assert t2d.noise_sigma(L["noise"]) == pytest.approx(L["data"][0, 0] / 100, rel=0.1)          # SNR 100 at the first echo
    assert abs(L["data"].mean()) > 5 * abs(L["noise"].mean())                    # the signal is in the real channel, the noise in the imaginary one


def test_loader_explains_what_is_wrong(tmp_path):
    d = tmp_path / "x"
    shutil.copytree(T2D_SNR100, d)
    p = read_kea_params(d / "acqu.par")
    (d / "DiffAxis.dat").write_text("1\n2\n3\n")
    L = t2d.load(d, p)                                       # the spectrometer's axis is only a cross-check: a wrong one is reported, not used
    assert not L["b_check"]["agrees"] and any(c["file"] == "DiffAxis.dat" and c["value"] is None for c in L["b_check"]["compared"])
    assert L["b_vec"].shape == (25,)
    (d / "DiffAxis.dat").unlink()
    (d / "data.dat").unlink()
    L = t2d.load(d, p)                                       # no spectrometer files at all: everything is calculated from acqu.par
    assert L["b_check"]["compared"] == [] and L["b_vec"].shape == (25,)
    bad = dict(p)
    del bad["DELTA"]
    with pytest.raises(ValueError, match="DELTA"):
        t2d.load(d, bad)


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
    assert big["constrained"] and small["constrained"] and r["fraction_in_sensitive_window"] > 0.97
    assert r["rmse_over_amp"] < 0.03 and 0.5 < r["chi2_per_point"] < 2.0
    assert r["projection_T2"].shape == (40,) and r["projection_D"].shape == (20,)


def test_a_range_wider_than_the_data_can_resolve_lets_regularisation_fill_the_rest():
    e, b, data = synthetic()
    wide = ilt2d.run(data, e, b, 1e-5, 1.0, 40, 1e-12, 1e-8, 20, 1e9, sigma=0.4)
    assert wide["fraction_in_sensitive_window"] < 0.85                          # a good part of the "signal" sits where the data cannot see
    assert any(not p["constrained"] for p in wide["peaks"])


def test_the_noisy_sample_fits_to_the_noise_level():
    L = t2d.load(T2D_SNR100, read_kea_params(T2D_SNR100 / "acqu.par"))
    sigma = t2d.noise_sigma(L["noise"])
    r = ilt2d.run(L["data"], L["echo_s"], L["b_vec"], 1e-5, 1.0, 30, 1e-12, 1e-8, 25, 1e9, sigma=sigma)
    assert 0.7 < r["chi2_per_point"] < 2.5 and r["rmse"] == pytest.approx(sigma, rel=0.5)


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

@pytest.mark.parametrize("d", [T2D_SNR100, T2D_CLEAN], ids=lambda p: p.name)
def test_both_samples_process_and_are_json_safe(d):
    load(d)
    res = process({"kind": "t2d", "t2d": SMALL})
    json.dumps(res, allow_nan=False)
    r = res["t2d"]
    assert res["mode"] == "t2d" and np.array(r["spectrum"]).shape == (12, 30) and r["peaks"] and r["rmse_over_amp"] < 0.15
    assert len(r["echo_s"]) == len(res["decay"]["t"]) and np.array(r["pred"]).shape == (25, 600)


def test_exclude_first_drops_echoes_from_the_inversion():
    load(T2D_SNR100)
    res = process({"kind": "t2d", "exclude_first": 5, "t2d": SMALL})
    assert res["decay"]["skip"] == 5 and len(res["t2d"]["echo_s"]) == 595 and np.array(res["t2d"]["resid"]).shape == (25, 595)
    with pytest.raises(ValueError, match="at least 8"):
        process({"kind": "t2d", "exclude_first": 595, "t2d": SMALL})


def test_settings_are_validated():
    load(T2D_SNR100)
    with pytest.raises(ValueError, match="minimum of each range"):
        process({"kind": "t2d", "t2d": {"t2_min": 1.0, "t2_max": 1e-5}})
    with pytest.raises(ValueError, match="d_min must be positive"):
        process({"kind": "t2d", "t2d": {"d_min": 0}})


def test_the_time_estimate_grows_with_the_grid_and_is_the_right_order():
    import time
    load(T2D_SNR100)
    small, big = t2d_estimate({"t2d": {"t2_steps": 20, "d_steps": 10}}), t2d_estimate({"t2d": {"t2_steps": 50, "d_steps": 25}})
    assert big["seconds"] > 10 * small["seconds"] and big["unknowns"] == 1250 and small["n_T2"] == 20
    # wall-clock times depend on what else the machine is doing (a shared CI runner running tests in parallel can be 5x off
    # between the trial and the run), so the estimate only has to be in the right ballpark on one of a few attempts
    ratios = []
    for _ in range(3):
        est = t2d_estimate({"t2d": {"t2_steps": 40, "d_steps": 25}})["seconds"]
        t0 = time.perf_counter()
        process({"kind": "t2d", "t2d": {"t2_steps": 40, "d_steps": 25}})
        ratios.append(est / (time.perf_counter() - t0))
        if 0.2 < ratios[-1] < 5:
            break
    assert 0.2 < ratios[-1] < 5, ratios


def test_mcmc_and_bootstrap_need_the_few_population_model_for_t2d():
    """The regularised map has no intervals; they sample the few-population model (see test_t2d_uncertainty.py for the runs themselves)."""
    load(T2D_SNR100)
    process({"kind": "t2d", "t2d": SMALL})
    for call in (lambda: mcmc_start({}), lambda: boot_start({})):
        with pytest.raises(ValueError, match="few-population model"):
            call()


def test_provenance_records_every_t2d_input_file():
    load(T2D_SNR100)
    process({"kind": "t2d", "t2d": SMALL})
    rec = get_provenance({"dataset": "sample: kea_t2d_snr100"}, {})["record"]
    names = {f["name"] for f in rec["inputs"]}
    assert {"acqu.par", "data2D_Cp.2d", "data.dat"} <= names and rec["settings"]["kind"] == "t2d"          # these samples have no DiffAxis.dat / T2Axis.dat
    assert rec["acquisition"]["nrPntsTau"] == 25 and rec["acquisition"]["DELTA"] == 5


# ── params file, script and batch ────────────────────────────────────────────

def test_params_t2d_section_round_trips():
    p = P.defaults()
    assert p["t2d"]["d_min"] == 1e-12 and p["t2d"]["d_max"] == 2.2e-9 and p["t2d"]["t2_min"] == 1e-5 and p["t2d"]["t2_max"] == 1.0 and p["t2d"]["alpha"] == 1e9
    s, o = P.to_settings_t2d(p)
    assert s == {"kind": "t2d", "exclude_first": 0, "t2d": {**p["t2d"], "gradient": 23.67}, "t2d_model": {"enabled": True, "k": None, "d_window": "resolvable", "seed": 0}}
    assert o["mcmc"]["enabled"] is False and o["bootstrap"]["enabled"] is False and o["mcmc"]["target"] == "bulk"
    p["t2d"]["t2_steps"] = 60
    assert P.from_settings(P.to_settings_t2d(p)[0])["t2d"]["t2_steps"] == 60
    import tomllib
    assert tomllib.loads(P.dump_toml(p))["t2d"]["t2_steps"] == 60
    with pytest.raises(ValueError, match="minimum of each range"):
        P.validate({"t2d": {"d_min": 1e-6, "d_max": 1e-9}})


def test_run_dataset_writes_the_t2d_files(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(T2D_SNR100, d)
    p = P.defaults()
    p["t2d"].update(SMALL)
    p["mcmc"].update({"enabled": True, "walkers": 16, "steps": 400, "burn": 100})
    row = runner.run_dataset(d, p, command="nmrdash run x")
    assert row["status"] == "ok" and row["kind"] == "t2d", row.get("reason")
    out = Path(row["output_dir"])
    names = {x.name for x in out.iterdir()}
    assert {"t2d_spectrum.csv", "t2d_peaks.csv", "t2d_statistics.csv", "t2d_data_fit_residual.csv", "t2d_spectrum.png", "t2d_fit.png", "summary.md", "provenance.json"} <= names
    assert "decay.csv" not in names
    peaks = list(csv.DictReader(open(out / "t2d_peaks.csv")))
    assert peaks and float(peaks[0]["D_mean_m2s"]) > 0 and 0.3 < sum(float(x["fraction"]) for x in peaks) <= 1.0001      # weak signal under the 10% threshold is in no region; how much that is depends on the ill-conditioned solve (54% on one platform, >80% on another)
    assert {"constrained_by_data"} <= set(peaks[0])
    spec = list(csv.reader(open(out / "t2d_spectrum.csv")))
    assert len(spec) == 1 + 12 and len(spec[0]) == 1 + 30
    assert "mcmc_summary.csv" in names and not row["warnings"]                    # MCMC now runs on the few-population model
    md = (out / "summary.md").read_text()
    assert "T₂–D regions" in md and "diffusion delays" in md and "?" not in md.split("## T₂–D regions")[0]


def test_a_mixed_batch_of_all_three_sequences(tmp_path):
    root = tmp_path / "all"
    for name, src in (("cpmg", "kea_cpmg_snr100"), ("t1", "kea_t1_snr100"), ("t2d", "kea_t2d_snr100")):
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
    lo, hi = ilt2d.sensitive_d_window([6.40474, 2884.67])                     # the T2D_SNR100 axis, in units of 1e9 s/m^2
    assert lo == pytest.approx(1 / 2.88467e12) and hi == pytest.approx(5 / 6.40474e9)
    lo2, hi2 = ilt2d.sensitive_d_window([64.0474, 28846.7])                    # ten times larger b: sees ten times smaller D
    assert lo2 == pytest.approx(lo / 10) and hi2 == pytest.approx(hi / 10)


def test_regions_outside_the_window_are_flagged_and_a_slow_population_shows_why_it_matters(tmp_path):
    slow = t2d_with_a_slow_population(tmp_path / "slow")
    for d in (T2D_SNR100, slow):
        load(d)
        r = process({"kind": "t2d", "t2d": {"t2_steps": 30, "d_steps": 25}})["t2d"]        # the full 25-point D grid
        lo, hi = r["d_sensitive"]
        assert all(p["constrained"] == (lo <= p["D_mean"] <= hi) for p in r["peaks"])
        if d == slow:
            assert r["fraction_in_sensitive_window"] < 0.95                    # about 82% (a loose bound: platforms differ); the population below the window is regularisation filling D the data cannot see
            assert any(not p["constrained"] for p in r["peaks"])
        assert all(0 <= p["fraction_in_window"] <= 1.0001 for p in r["peaks"])        # the part of each region inside the window, never more than all of it
        assert sum(p["fraction_in_window"] for p in r["peaks"]) <= 1.0001


def test_summary_warns_when_much_of_the_spectrum_is_unconstrained(tmp_path):
    d = t2d_with_a_slow_population(tmp_path / "ds")
    p = P.defaults()
    p["t2d"].update(SMALL)
    out = Path(runner.run_dataset(d, p)["output_dir"])
    md = (out / "summary.md").read_text()
    assert "of the spectrum lies where the data can constrain it" in md and "**no**" in md


def test_auto_d_range_uses_exactly_what_the_b_values_can_resolve():
    load(T2D_SNR100)
    r = process({"kind": "t2d", "t2d": {**SMALL, "d_min": "auto", "d_max": "auto"}})["t2d"]
    lo, hi = ilt2d.sensitive_d_window(_STATE["t2d"]["b_vec"])
    assert r["d_min"] == pytest.approx(lo, rel=0.06) and r["d_max"] == pytest.approx(hi, rel=0.06)
    assert r["fraction_in_sensitive_window"] > 0.85          # not 1: signal at the edge bins is on the limit, which the rounding of the grid ends decides
    p = P.validate({"t2d": {"d_min": "auto", "d_max": "auto"}})
    assert P.to_settings_t2d(p)[0]["t2d"]["d_min"] == "auto"
    with pytest.raises(ValueError, match="must both be numbers"):
        P.validate({"t2d": {"d_min": "auto"}})
