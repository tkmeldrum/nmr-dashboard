import base64
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("emcee")

from nmrdash import multiexp, uncertainty as u  # noqa: E402
from nmrdash.pipeline import (boot_finish, boot_start, boot_step, load, mcmc_finish, mcmc_start, mcmc_step,  # noqa: E402
                              process)

DATA = Path(__file__).resolve().parent.parent / "test_data"
SIGMA = 4.0


def decay(comps=((1000, 0.004), (500, 0.04)), n=512, seed=0, sigma=SIGMA):
    t = 2e-4 * np.arange(1, n + 1)
    y = sum(a * np.exp(-t / T) for a, T in comps)
    rng = np.random.default_rng(seed)
    return t, y + rng.normal(0, sigma, n), rng.normal(0, sigma, n)          # (t, y, an independent noise-only series)


def mcmc(t, y, n=2, offset=False, sigma=SIGMA, **kw):
    kw = {"steps": 1500, "burn": 500, **kw}
    f = multiexp.fit(t, y, n, offset, sigma=sigma)
    return u.Mcmc(t, y, n, offset, f, sigma=sigma, **kw), f


def test_mcmc_matches_least_squares_on_a_well_conditioned_problem():
    t, y, _ = decay()
    m, f = mcmc(t, y)
    m.step(1500)
    r = m.result({"fg": "#222", "grid": "#aaa"})
    for p in r["params"]:
        assert p["median"] == pytest.approx(p["ls"], abs=1.5 * p["ls_se"])
        assert p["sd"] == pytest.approx(p["ls_se"], rel=0.3)               # posterior width ~ covariance width
    d = {x["name"]: x for x in r["derived"]}
    assert d["fraction 1"]["lo"] < f["components"][0]["fraction"] < d["fraction 1"]["hi"]
    assert 0.2 < r["diagnostics"]["acceptance"] < 0.9 and r["diagnostics"]["tau_max"] > 1
    assert len(r["predictive"]["lo"]) == len(r["predictive"]["hi"]) == len(r["predictive"]["t"])
    assert base64.b64decode(r["corner_png_b64"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_chunked_run_is_identical_to_a_single_run():
    t, y, _ = decay()
    a, _ = mcmc(t, y, seed=7)
    b, _ = mcmc(t, y, seed=7)
    a.step(1500)
    for _ in range(3):
        b.step(500)
    np.testing.assert_array_equal(a.sampler.get_chain(), b.sampler.get_chain())


def test_stopping_early_still_gives_a_summary_with_a_warning():
    t, y, _ = decay()
    m, _ = mcmc(t, y, steps=3000, burn=1000)
    m.step(300)
    r = m.result()
    assert r["steps_run"] == 300 and r["burn"] == 150                     # burn-in cut to half of what ran
    assert any("Stopped early" in w for w in r["warnings"]) and any("Burn-in was cut" in w for w in r["warnings"])
    with pytest.raises(RuntimeError):
        mcmc(t, y)[0].result()                                            # nothing has run yet


def test_sampled_sigma_recovers_the_true_noise():
    t, y, _ = decay(sigma=6.0)
    f = multiexp.fit(t, y, 2, False)
    m = u.Mcmc(t, y, 2, False, f, sigma=None, steps=1500, burn=500)
    m.step(1500)
    sig = next(p for p in m.result()["params"] if p["name"] == "sigma")
    assert sig["lo"] < 6.0 < sig["hi"] * 1.05 and sig["median"] == pytest.approx(6.0, rel=0.1)


def test_credible_intervals_have_sensible_coverage():
    hits, total = 0, 0
    for seed in range(20):
        t, y, _ = decay(comps=((1000, 0.02),), n=256, seed=seed)
        m, _ = mcmc(t, y, n=1, ci_level=0.68, steps=700, burn=250)
        m.step(700)
        for p, truth in zip(m.result()["params"], (1000, 0.02)):
            hits += p["lo"] <= truth <= p["hi"]
            total += 1
    assert 0.5 <= hits / total <= 0.88                                    # nominal 0.68, 40 intervals


def test_time_estimate_is_the_right_order_of_magnitude():
    t, y, _ = decay()
    m, _ = mcmc(t, y, steps=1500)
    t0 = time.perf_counter()
    m.step(1500)
    actual = time.perf_counter() - t0
    assert 0.25 < m.est_seconds / actual < 4


def test_overparameterised_model_does_not_crash_and_reports_it():
    t, y, _ = decay(comps=((1000, 0.02),))
    f = multiexp.fit(t, y, 3, False, sigma=SIGMA)
    m = u.Mcmc(t, y, 3, False, f, sigma=SIGMA, steps=800, burn=300)
    m.step(800)
    r = m.result()
    assert r["warnings"] or r["diagnostics"]["steps_per_tau"] < 100
    json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o), allow_nan=True)


# ── bootstrap ────────────────────────────────────────────────────────────────

def boot(t, y, noise, method="noise", me_n=1, replicates=150, ilt=None, offset=False, **kw):
    f = multiexp.fit(t, y, me_n, offset, sigma=SIGMA) if me_n else None
    return u.Bootstrap(t, y, noise, SIGMA, me={"n": me_n, "offset": offset} if me_n else None, ls_fit=f, ilt=ilt,
                       method=method, replicates=replicates, **kw), f


@pytest.mark.parametrize("method", ["noise", "residual", "parametric"])
def test_bootstrap_se_matches_the_covariance_se(method):
    t, y, noise = decay(comps=((1000, 0.02),), n=256)
    b, f = boot(t, y, noise, method)
    b.step(150)
    r = b.result()["multiexp"]
    T = next(p for p in r["params"] if p["name"] == "T2")
    assert T["sd"] == pytest.approx(f["components"][0]["T2_se"], rel=0.4)
    assert T["lo"] < 0.02 < T["hi"] and r["n_ok"] >= 140


def test_noise_bootstrap_needs_an_imaginary_channel_and_residual_does_not():
    t, y, _ = decay(comps=((1000, 0.02),), n=256)
    f = multiexp.fit(t, y, 1, False)
    with pytest.raises(ValueError):
        u.Bootstrap(t, y, None, None, me={"n": 1, "offset": False}, ls_fit=f, method="noise")
    with pytest.raises(ValueError):
        u.Bootstrap(t, y, None, None, me={"n": 1, "offset": False}, ls_fit=f, method="parametric")
    b = u.Bootstrap(t, y, None, None, me={"n": 1, "offset": False}, ls_fit=f, replicates=20)
    assert b.method == "residual"                                         # sensible default without noise information


def test_bootstrap_is_reproducible_and_chunking_does_not_matter():
    t, y, noise = decay(comps=((1000, 0.02),), n=256)
    a, _ = boot(t, y, noise, replicates=60, seed=3)
    b, _ = boot(t, y, noise, replicates=60, seed=3)
    a.step(60)
    for _ in range(4):
        b.step(20)
    np.testing.assert_allclose(np.array(a.me_rows), np.array(b.me_rows))


def test_ilt_bootstrap_band_and_peak_intervals():
    t, y, noise = decay(comps=((600, 0.004), (400, 0.05)))
    ilt = {"t2_min": 1e-4, "t2_max": 1.0, "n_steps": 80, "alpha": 1e8}
    b, _ = boot(t, y, noise, me_n=0, replicates=40, ilt=ilt)
    b.step(40)
    r = b.result()["ilt"]
    lo, med, hi = map(np.array, (r["lo"], r["median"], r["hi"]))
    assert np.all(lo <= med + 1e-9) and np.all(med <= hi + 1e-9) and np.all(lo >= 0)
    assert len(r["bands"]) == 2
    fast, slow = sorted(r["bands"], key=lambda x: x["t2_peak"])
    assert fast["T2"]["lo"] < 0.004 * 1.6 and slow["T2"]["hi"] > 0.05 * 0.6
    assert fast["fraction"]["median"] == pytest.approx(0.6, abs=0.1)


def test_warm_started_fit_agrees_with_the_default_fit():
    t, y, _ = decay()
    f = multiexp.fit(t, y, 2, False)
    g = multiexp.fit(t, y, 2, False, p0=u.params_from_fit(f))
    assert g["rss"] == pytest.approx(f["rss"], rel=1e-6)


# ── through the pipeline (what the page calls) ───────────────────────────────

SETTINGS = {"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False},
            "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}}


def test_pipeline_mcmc_and_bootstrap_on_real_data_are_json_safe():
    load(DATA / "CPMG_EtGly" / "1")
    process(SETTINGS)
    info = mcmc_start({"steps": 600, "burn": 200, "nwalkers": 16})
    assert info["est_seconds"] > 0 and info["ndim"] == 3 and not info["sigma_sampled"]
    mcmc_step(300)
    assert mcmc_step(300) == {"done": 600, "total": 600}
    r = mcmc_finish({"fg": "#222", "grid": "#aaa"})
    json.dumps(r, allow_nan=False)
    assert r["params"][1]["name"] == "T2" and r["params"][1]["lo"] < r["params"][1]["median"] < r["params"][1]["hi"]

    b = boot_start({"replicates": 30, "do_ilt": True})
    assert b["method"] == "noise" and b["done"] == 3
    boot_step(100)
    out = boot_finish()
    json.dumps(out, allow_nan=False)
    assert out["replicates"] == 30 and out["multiexp"] and out["ilt"]


def test_pipeline_can_target_a_position_after_ft():
    load(DATA / "CPMG_EtGly" / "1")
    process({"echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4, "z_min": -100, "z_max": 100},
             "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95}, "ilt": {"enabled": False}})
    info = mcmc_start({"target": "position", "z_um": 0, "steps": 300, "burn": 100, "nwalkers": 16})
    assert info["target"].startswith("position z =")
    mcmc_step(300)
    assert mcmc_finish()["steps_run"] == 300
    assert boot_start({"target": "position", "z_um": 20, "replicates": 10, "do_ilt": False})["target"].startswith("position z =")


def test_pipeline_errors_are_clear():
    load(DATA / "CPMG_EtGly" / "1")
    with pytest.raises(RuntimeError, match="Run the processing first"):
        mcmc_start({})
    process({**SETTINGS, "multiexp": {"enabled": False}, "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}})
    with pytest.raises(ValueError, match="multi-exponential"):
        mcmc_start({})
    with pytest.raises(ValueError, match="after an FT run"):
        boot_start({"target": "position"})
