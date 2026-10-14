"""MCMC and bootstrap for the T2-D few-population model."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("emcee")

from nmrdash import ilt2d, t2d_model as M, t2d_uncertainty as U  # noqa: E402
from nmrdash.pipeline import (_STATE, boot_finish, boot_start, boot_step, load, mcmc_finish, mcmc_start, mcmc_step, process)  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "test_data"
M1115, S3115 = DATA / "M1115_rt_30phr" / "1", DATA / "S3115_rt_100phr" / "1"
SIGMA, T2R = 10.0, (1e-5, 1.0)
TWO = [(4000.0, 0.004, 3e-11), (2500.0, 0.03, 2e-12)]
THEME = {"fg": "#222", "grid": "#aaa"}


@pytest.fixture(scope="module")
def axes():
    load(M1115)
    L = _STATE["t2d"]
    return np.asarray(L["echo_s"][:300], float), np.asarray(L["b_vec"], float), ilt2d.sensitive_d_window(L["b_vec"])


def synth(axes, pops, seed=0):
    t, b, _ = axes
    Y = M.predict(t, b, [p[0] for p in pops], [p[1] for p in pops], [p[2] for p in pops])
    return Y + np.random.default_rng(seed).normal(0, SIGMA, Y.shape)


def fitted(axes, pops, seed=0, k=None):
    regions = [{"T2_mean": T * 1.3, "D_mean": D / 1.3} for A, T, D in pops]
    Y = synth(axes, pops, seed)
    return Y, M.fit(axes[0], axes[1], Y, regions, T2R, axes[2], sigma=SIGMA, k=k or len(pops), max_k=len(pops))


def mcmc(axes, pops, seed=0, steps=1500, **kw):
    Y, f = fitted(axes, pops, seed)
    m = U.Mcmc2D(axes[0], axes[1], Y, f, T2R, axes[2], sigma=SIGMA, nwalkers=kw.pop("nwalkers", 20), steps=steps, burn=steps // 3, seed=seed, **kw)
    return m, f


def test_mcmc_agrees_with_least_squares_where_the_problem_is_well_determined(axes):
    m, f = mcmc(axes, TWO, steps=2500)
    m.step(2500)
    r = m.result(THEME)
    truth = [v for A, T, D in TWO for v in (A, T, D)]
    for p, x in zip(r["params"], truth):
        assert abs(p["median"] - x) < 3.5 * p["sd"], p["name"]
        assert p["sd"] == pytest.approx(p["ls_se"], rel=0.4), p["name"]                 # posterior width ~ covariance width
    d = {x["name"]: x for x in r["derived"]}
    assert d["fraction 1"]["median"] + d["fraction 2"]["median"] == pytest.approx(1.0, abs=0.02)
    assert r["diagnostics"]["acceptance"] > 0.15 and not any("edge" in w for w in r["warnings"])


def test_the_95_percent_credible_intervals_cover_the_truth_about_95_percent_of_the_time(axes):
    hits, total = 0, 0
    for seed in range(14):
        m, _ = mcmc(axes, TWO, seed=seed, steps=1500)
        m.step(1500)
        r = m.result(THEME)
        for p, x in zip(r["params"], [v for A, T, D in TWO for v in (A, T, D)]):
            hits += p["lo"] <= x <= p["hi"]
            total += 1
    assert hits / total >= 0.85, hits / total


def test_a_population_the_data_cannot_place_gets_a_warning_and_a_one_sided_posterior(axes):
    truth = [(4000.0, 0.004, 3e-11), (3000.0, 0.02, 1e-13)]                             # the second diffuses more slowly than the b-values can tell from zero
    m, f = mcmc(axes, truth, steps=1500)
    m.step(1500)
    r = m.result(THEME)
    assert any("D of population 2" in w and "only a bound" in w for w in r["warnings"])
    d2 = next(p for p in r["params"] if p["name"] == "D2")
    assert d2["lo"] <= 1.5 * axes[2][0]                                                  # the posterior sits against the lower edge of the allowed range


def test_the_noise_level_can_be_sampled_and_is_found(axes):
    Y, f = fitted(axes, TWO)
    m = U.Mcmc2D(axes[0], axes[1], Y, f, T2R, axes[2], sigma=None, nwalkers=24, steps=1500, burn=500)
    m.step(1500)
    s = next(p for p in m.result(THEME)["params"] if p["name"] == "sigma")
    assert s["median"] == pytest.approx(SIGMA, rel=0.05)


def test_runs_are_reproducible_and_stopping_early_is_reported(axes):
    a, _ = mcmc(axes, TWO, seed=4, steps=600)
    b, _ = mcmc(axes, TWO, seed=4, steps=600)
    a.step(600)
    b.step(250), b.step(350)                                                            # chunked: the same chain
    assert [p["median"] for p in a.result(THEME)["params"]] == [p["median"] for p in b.result(THEME)["params"]]
    c, _ = mcmc(axes, TWO, seed=4, steps=600)
    c.step(200)
    assert any("Stopped early" in w for w in c.result(THEME)["warnings"])


def test_the_time_estimate_is_the_right_order(axes):
    m, _ = mcmc(axes, TWO, steps=800)
    t0 = time.perf_counter()
    m.step(800)
    assert 0.2 < m.est_seconds / (time.perf_counter() - t0) < 5


@pytest.mark.parametrize("method", ["noise", "residual", "parametric"])
def test_the_bootstrap_widths_match_least_squares_for_every_noise_source(axes, method):
    Y, f = fitted(axes, TWO)
    noise = np.random.default_rng(99).normal(0, SIGMA, Y.shape)                         # stands in for the imaginary channel
    b = U.Bootstrap2D(axes[0], axes[1], Y, f, T2R, axes[2], noise=noise, sigma=SIGMA, method=method, replicates=60, seed=2)
    b.step(60)
    r = b.result()
    assert r["method"] == method and r["replicates"] == 60 and r["failures"] == 0
    for p in r["params"]:
        if p["name"] in ("T2,1", "D1", "T2,2", "D2"):
            assert p["sd"] == pytest.approx(p["ls_se"], rel=0.5), p["name"]
            assert p["lo"] <= p["ls"] <= p["hi"]


def test_the_bootstrap_checks_its_inputs(axes):
    Y, f = fitted(axes, TWO)
    with pytest.raises(ValueError, match="imaginary channel"):
        U.Bootstrap2D(axes[0], axes[1], Y, f, T2R, axes[2], noise=None, sigma=SIGMA, method="noise", replicates=5)
    with pytest.raises(ValueError, match="Unknown"):
        U.Bootstrap2D(axes[0], axes[1], Y, f, T2R, axes[2], noise=None, sigma=SIGMA, method="nope", replicates=5)


def test_through_the_pipeline_on_a_real_sample_and_strict_json():
    load(S3115)
    process({"kind": "t2d", "t2d": {"t2_steps": 24, "d_steps": 12}, "t2d_model": {"enabled": True}})
    info = mcmc_start({"steps": 800, "burn": 300, "nwalkers": 24, "ci_level": 0.95})
    assert info["target"].startswith("3-population") and info["ndim"] == 9
    mcmc_step(800)
    r = mcmc_finish(THEME, 50)
    json.dumps(r, allow_nan=False)
    assert r["kind"] == "mcmc_t2d" and len(r["params"]) == 9 and len(r["scatter"]["T2"]) == 3 and r["corner_png_b64"]
    boot_start({"replicates": 12})
    boot_step(12)
    b = boot_finish()
    json.dumps(b, allow_nan=False)
    assert b["kind"] == "bootstrap_t2d" and b["method"] == "noise" and b["replicates"] == 12


def test_without_the_model_the_message_says_what_to_do():
    load(M1115)
    process({"kind": "t2d", "t2d": {"t2_steps": 20, "d_steps": 10}})
    for start in (mcmc_start, boot_start):
        with pytest.raises(ValueError, match="few-population model"):
            start({})
