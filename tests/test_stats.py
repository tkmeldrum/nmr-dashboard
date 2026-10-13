"""Fit-quality statistics: they must not depend on absolute amplitude or on the number of echoes."""
from pathlib import Path

import numpy as np
import pytest

from nmrdash import echoes, ilt_tools, multiexp
from nmrdash.pipeline import _STATE, load, process

DATA = Path(__file__).resolve().parent.parent / "test_data"
SIGMA = 4.0


def decay(n=256, scale=1.0, comps=((1000, 0.004), (500, 0.04)), seed=0, dt=2e-4):
    t = dt * np.arange(1, n + 1)
    y = sum(a * np.exp(-t / T) for a, T in comps) * scale
    return t, y + np.random.default_rng(seed).normal(0, SIGMA * scale, n)


def test_noise_sigma_recovers_known_sigma_from_imaginary_channel():
    rng = np.random.default_rng(1)
    im = rng.normal(0, 7.0, 512)
    assert echoes.noise_sigma(im) == pytest.approx(7.0, rel=0.1)
    assert echoes.noise_sigma(np.zeros(100)) is None and echoes.noise_sigma([1.0, 2.0]) is None


def test_noise_sigma_robust_to_a_leaked_signal_spike():
    rng = np.random.default_rng(2)
    im = rng.normal(0, 5.0, 512)
    im[:3] += 500                                   # a few echoes with leaked signal
    assert echoes.noise_sigma(im) == pytest.approx(5.0, rel=0.15)


def test_chi2_red_is_about_one_for_the_right_model_and_large_for_the_wrong_one():
    t, y = decay()
    good = multiexp.fit(t, y, 2, offset=False, sigma=SIGMA)
    bad = multiexp.fit(t, y, 1, offset=False, sigma=SIGMA)
    assert good["chi2_red"] == pytest.approx(1.0, abs=0.25)
    assert bad["chi2_red"] > 3


def test_normalised_measures_ignore_amplitude_but_rss_and_mse_do_not():
    t, y1 = decay(scale=1.0)
    _, y10 = decay(scale=10.0)
    a = multiexp.fit(t, y1, 2, offset=False, sigma=SIGMA)
    b = multiexp.fit(t, y10, 2, offset=False, sigma=10 * SIGMA)
    assert b["rss"] == pytest.approx(100 * a["rss"], rel=1e-3) and b["mse"] == pytest.approx(100 * a["mse"], rel=1e-3)
    assert b["resid_sd_over_amp"] == pytest.approx(a["resid_sd_over_amp"], rel=1e-3)
    assert b["chi2_red"] == pytest.approx(a["chi2_red"], rel=1e-3)


def test_normalised_measures_ignore_number_of_echoes_but_rss_does_not():
    t1, y1 = decay(n=128)
    t2, y2 = decay(n=512, seed=3)
    a = multiexp.fit(t1, y1, 2, offset=False, sigma=SIGMA)
    b = multiexp.fit(t2, y2, 2, offset=False, sigma=SIGMA)
    assert b["rss"] > 3 * a["rss"]                                      # RSS grows with N
    assert b["mse"] == pytest.approx(a["mse"], rel=0.3)                 # MSE does not
    assert b["resid_sd_over_amp"] == pytest.approx(a["resid_sd_over_amp"], rel=0.3)
    assert b["chi2_red"] == pytest.approx(a["chi2_red"], abs=0.3)


def test_delta_criteria_are_zero_for_best_and_invariant_to_amplitude():
    t, y1 = decay()
    _, y10 = decay(scale=10.0)
    r1 = multiexp.fit_with_comparison(t, y1, 2, offset=False, compare_up_to=3)["comparison"]
    r10 = multiexp.fit_with_comparison(t, y10, 2, offset=False, compare_up_to=3)["comparison"]
    assert min(r["d_bic"] for r in r1) == 0 and min(r["d_aic"] for r in r1) == 0
    for a, b in zip(r1, r10):
        assert b["d_bic"] == pytest.approx(a["d_bic"], abs=0.5)         # raw BIC shifts by N*ln(100); the difference does not
        assert b["bic"] != pytest.approx(a["bic"], abs=1)


def test_no_sigma_means_no_chi2_but_everything_else_still_there():
    t, y = decay()
    f = multiexp.fit(t, y, 2, offset=False)
    assert f["chi2_red"] is None and f["noise_sigma"] is None
    assert f["mse"] > 0 and f["resid_sd_over_amp"] > 0


def test_ilt_statistics():
    t, y = decay()
    r = ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8, sigma=SIGMA)
    assert r["rss"] == pytest.approx(r["mse"] * len(t)) and r["signal_range"] == pytest.approx(np.ptp(r["pred"]))
    assert 0 < r["rmse_over_amp"] < 0.05 and r["chi2_per_point"] == pytest.approx(1.0, abs=0.35)
    assert ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8)["chi2_per_point"] is None


def test_noise_from_real_data_gives_a_sensible_chi2():
    load(DATA / "CPMG_EtGly" / "1")
    sigma = echoes.integrate_noise(_STATE["td"])
    assert sigma and sigma > 0
    res = process({"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "compare": False},
                   "ilt": {"enabled": False}})
    chi2 = res["multiexp"]["selected"]["chi2_red"]
    assert res["decay"]["noise_sigma"] == pytest.approx(sigma)
    assert 0.5 < chi2 < 5            # an honest single-exponential fit of a good dataset: close to 1, not wildly off


def test_real_valued_data_has_no_noise_estimate():
    assert echoes.integrate_noise(np.ones((8, 16))) is None
    assert echoes.window_noise(np.ones((8, 16)), np.arange(8.0), 0, 7) is None
    assert np.isnan(echoes.position_noise(np.ones((8, 16)), np.arange(3))).all()


def test_spatial_fits_carry_chi2_per_position():
    load(DATA / "CPMG_EtGly" / "1")
    res = process({"echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4, "z_min": -100, "z_max": 100},
                   "multiexp": {"enabled": True, "n": 1, "offset": True}, "ilt": {"enabled": False}})
    chi2 = [c for c in res["spatial"]["multiexp"]["chi2_red"] if c is not None]
    assert chi2 and all(0.2 < c < 10 for c in chi2)
    assert res["decay"]["noise_sigma"] > 0


def test_normalised_residual_is_comparable_between_model_orders():
    """Same residuals, so (nearly) the same normalised value; the old sum-of-amplitudes denominator drifted with order."""
    t, y = decay(comps=((1000, 0.02),))
    vals = [multiexp.fit(t, y, n, offset=False)["resid_sd_over_amp"] for n in (1, 2, 3)]
    assert max(vals) / min(vals) < 1.05


def test_exp_markers_land_on_the_ilt_peaks_when_the_amounts_agree():
    """Two well separated components: each marker is drawn near the top of its ILT peak, and moves if the amount changes."""
    t, y = decay(comps=((600, 0.004), (400, 0.05)), n=512)
    me = multiexp.fit_with_comparison(t, y, 2, offset=False, compare_up_to=2)["selected"]
    il = ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8)
    mk = ilt_tools.exp_markers(me, il)
    heights = sorted(il["spectrum"][np.argmin(abs(np.array(il["tau"]) - p["T2_peak"]))] for p in il["peaks"])
    got = sorted(m["height"] for m in mk)
    assert got == pytest.approx(heights, rel=0.25)                       # same amounts -> same heights
    assert [m["matched_peak"] is not None for m in mk] == [True, True]
    inflated = {**me, "components": [{**c, "A": 2 * c["A"]} for c in me["components"]]}
    assert [m["height"] for m in ilt_tools.exp_markers(inflated, il)] == pytest.approx([2 * m["height"] for m in mk])   # twice the amount, twice the height


def test_exp_markers_without_an_ilt_peak_have_no_height():
    il = {"tau": [1e-3, 1e-2], "spectrum": [0.0, 0.0], "peaks": []}
    me = {"components": [{"T2": 5e-3, "A": 10.0, "fraction": 1.0}]}
    assert ilt_tools.exp_markers(me, il)[0]["height"] is None


def test_process_attaches_markers_only_when_both_analyses_ran():
    load(DATA / "CPMG_EtGly" / "1")
    both = process({"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "compare": False}, "ilt": {"enabled": True, "t2_min": 1e-4, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}})
    assert len(both["ilt"]["exp_markers"]) == 1
    only = process({"echo": "integrate", "multiexp": {"enabled": False}, "ilt": {"enabled": True, "t2_min": 1e-4, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}})
    assert "exp_markers" not in only["ilt"]
