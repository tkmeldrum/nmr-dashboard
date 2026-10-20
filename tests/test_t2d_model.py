"""The few-population T2-D model: recovery of known populations, honest intervals, model choice, and behaviour on the real samples."""
import json
from pathlib import Path

import numpy as np
import pytest

from nmrdash import ilt2d, t2d_model as M
from nmrdash.pipeline import _STATE, load, process

from sample_data import T2D_NOISY, t2d_with_a_slow_population, TRUTH
SIGMA = 10.0
T2R = (1e-5, 1.0)


@pytest.fixture(scope="module")
def axes():
    load(T2D_NOISY)
    L = _STATE["t2d"]
    return np.asarray(L["echo_s"][:400], float), np.asarray(L["b_vec"], float), ilt2d.sensitive_d_window(L["b_vec"])


def synth(axes, pops, seed=0, sigma=SIGMA):
    t, b, _ = axes
    Y = M.predict(t, b, [p[0] for p in pops], [p[1] for p in pops], [p[2] for p in pops])
    return Y + np.random.default_rng(seed).normal(0, sigma, Y.shape)


def regions_near(pops, f=1.3):
    """What the 2D map would hand over: roughly the right places, not exactly."""
    return [{"T2_mean": T * f, "D_mean": D / f, "fraction": A} for A, T, D in sorted(pops, key=lambda p: -p[0])]


TWO = [(4000.0, 0.004, 3e-11), (2500.0, 0.03, 2e-12)]


def test_two_populations_are_recovered_within_their_intervals(axes):
    out = M.fit(axes[0], axes[1], synth(axes, TWO), regions_near(TWO), T2R, axes[2], sigma=SIGMA, k=2, max_k=2)
    assert out["K"] == 2 and out["chi2_per_point"] == pytest.approx(1.0, abs=0.05)
    for p, (A, T, D) in zip(out["populations"], TWO):
        assert abs(p["T2"] - T) < 3 * p["T2_se"] and abs(p["D"] - D) < 3 * p["D_se"] and abs(p["A"] - A) < 3 * p["A_se"]
        assert p["supported"] and p["D_at_limit"] is None
    assert sum(p["fraction"] for p in out["populations"]) == pytest.approx(1.0)


def test_the_95_percent_intervals_cover_the_truth_about_95_percent_of_the_time(axes):
    """Coverage over many noise draws: the claim behind every interval the page shows."""
    hits = {"T2": 0, "D": 0, "A": 0}
    n = 30
    for seed in range(n):
        out = M.fit(axes[0], axes[1], synth(axes, TWO, seed=seed), regions_near(TWO), T2R, axes[2], sigma=SIGMA, k=2, max_k=2, seed=seed)
        for p, (A, T, D) in zip(out["populations"], TWO):
            hits["T2"] += p["T2_ci"][0] <= T <= p["T2_ci"][1]
            hits["D"] += p["D_ci"][0] <= D <= p["D_ci"][1]
            hits["A"] += abs(p["A"] - A) <= p["t_crit"] * p["A_se"] if "t_crit" in p else abs(p["A"] - A) <= 1.96 * p["A_se"]
    for k, h in hits.items():
        assert h / (2 * n) >= 0.85, (k, h / (2 * n))


@pytest.mark.parametrize("truth", [[(5000.0, 0.01, 1e-11)], TWO, TWO + [(1500.0, 0.0012, 2e-10)]])
def test_the_number_of_populations_is_recovered_by_bic(axes, truth):
    out = M.fit(axes[0], axes[1], synth(axes, truth, seed=3), regions_near(truth), T2R, axes[2], sigma=SIGMA, max_k=4)
    assert out["recommended_k"] == len(truth) and out["K"] == len(truth)
    assert not any("not supported" in w for w in out["warnings"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_population_slower_than_the_window_is_reported_as_a_bound(axes, seed):
    """D below what the largest b-value can tell apart from zero: the fit stops at the limit and says the data give only a bound."""
    truth = [(4000.0, 0.004, 3e-11), (3000.0, 0.02, 1e-13)]
    out = M.fit(axes[0], axes[1], synth(axes, truth, seed), regions_near(truth), T2R, axes[2], sigma=SIGMA, k=2, max_k=2)
    slow = out["populations"][1]
    assert slow["D_at_limit"] == "low" and np.isnan(slow["D_se"]) and slow["D"] == pytest.approx(axes[2][0], rel=0.01)
    assert any("only a bound" in w for w in out["warnings"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_population_too_fast_to_leave_any_signal_is_not_invented(axes, seed):
    """exp(-b D) is already ~e^-25 at the first b-value for D = 4e-9: nothing in the data, so BIC asks for one population, not two."""
    truth = [(4000.0, 0.004, 3e-11), (3000.0, 0.02, 4e-9)]
    out = M.fit(axes[0], axes[1], synth(axes, truth, seed), regions_near(truth), T2R, axes[2], sigma=SIGMA, max_k=3)
    assert out["recommended_k"] == 1 and out["K"] == 1
    p = out["populations"][0]
    assert abs(p["D"] - 3e-11) < 3 * p["D_se"] and abs(p["A"] - 4000) < 3 * p["A_se"]


def test_asking_for_more_populations_than_exist_is_called_out(axes):
    truth = [(5000.0, 0.01, 1e-11)]
    out = M.fit(axes[0], axes[1], synth(axes, truth, seed=5), regions_near(truth), T2R, axes[2], sigma=SIGMA, k=3, max_k=3)
    assert out["recommended_k"] == 1 and any("BIC prefers 1 population" in w for w in out["warnings"])
    assert any(not p["supported"] or p["D_at_limit"] or p["T2_at_limit"] for p in out["populations"])


def test_populations_come_back_ordered_by_t2(axes):
    out = M.fit(axes[0], axes[1], synth(axes, TWO), list(reversed(regions_near(TWO))), T2R, axes[2], sigma=SIGMA, k=2, max_k=2)
    assert [p["T2"] for p in out["populations"]] == sorted(p["T2"] for p in out["populations"])


def test_the_noisy_sample_fits_to_the_noise_level_and_a_slow_population_is_bound(tmp_path):
    truth = TRUTH["t2d"]
    for sample, k_expected in ((T2D_NOISY, 2), (t2d_with_a_slow_population(tmp_path / "slow"), 3)):
        load(sample)
        res = process({"kind": "t2d", "t2d": {"t2_steps": 30, "d_steps": 15}, "t2d_model": {"enabled": True}})["t2d"]
        m = res["model"]
        json.dumps(res, allow_nan=False)                                                  # strict JSON, including the "no interval" values
        assert m["K"] == k_expected and len(m["comparison"]) >= 3 and m["d_window"] == pytest.approx(res["d_sensitive"])
        if k_expected == 2:                                                               # both populations in the window: recovered, and the fit is at the noise level
            assert m["chi2_per_point"] < 1.1 and not any(p["D_at_limit"] for p in m["populations"])
            for p, T2, D, f in zip(m["populations"], truth["T2_s"], truth["D_m2s"], truth["fraction"]):
                assert p["T2"] == pytest.approx(T2, rel=0.05) and p["D"] == pytest.approx(D, rel=0.05) and p["fraction"] == pytest.approx(f, abs=0.03)
        else:                                                                             # the slow one cannot be resolved: it is reported as bound at the low limit
            assert any(p["D_at_limit"] == "low" for p in m["populations"])
    assert _STATE["processed"]["t2d_model"]["fit"]["K"] == 3                                  # kept for MCMC and the bootstrap


def test_the_model_is_off_unless_asked_for_and_k_is_checked():
    load(T2D_NOISY)
    assert "model" not in process({"kind": "t2d", "t2d": {"t2_steps": 20, "d_steps": 10}})["t2d"]
    for bad in (0, 9, 2.5, "2"):
        with pytest.raises(ValueError, match="t2d_model.k"):
            process({"kind": "t2d", "t2d": {"t2_steps": 20, "d_steps": 10}, "t2d_model": {"enabled": True, "k": bad}})
