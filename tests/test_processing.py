import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nmrdash import echoes, ilt_tools, multiexp
from nmrdash.kea_io import prepare_t2_data, read_kea_params
from nmrdash.pipeline import load, process

DATA = Path(__file__).resolve().parent.parent / "test_data"
RNG = np.random.default_rng(1)


def synth(components, c=0.0, noise=0.0, n=256, dt=2e-4):
    t = dt * np.arange(1, n + 1)
    y = sum(a * np.exp(-t / T) for a, T in components) + c
    return t, y + noise * RNG.standard_normal(n)


def test_monoexp_recovered_within_ci():
    t, y = synth([(1000, 0.02)], c=5, noise=3)
    f = multiexp.fit(t, y, 1)
    comp = f["components"][0]
    lo, hi = comp["T2_ci"]
    assert lo < 0.02 < hi and comp["T2_se"] / 0.02 < 0.05
    assert comp["A_ci"][0] < 1000 < comp["A_ci"][1]


def test_biexp_recovered_and_sorted():
    t, y = synth([(600, 0.004), (400, 0.04)], noise=2)
    f = multiexp.fit(t, y, 2, offset=False)
    T = [c["T2"] for c in f["components"]]
    assert T == sorted(T)
    assert T[0] == pytest.approx(0.004, rel=0.1) and T[1] == pytest.approx(0.04, rel=0.1)
    assert sum(c["fraction"] for c in f["components"]) == pytest.approx(1)
    assert f["components"][0]["fraction"] == pytest.approx(0.6, abs=0.05)


def test_bic_picks_true_number_of_components():
    for truth, comps in {1: [(1000, 0.02)], 2: [(600, 0.004), (400, 0.04)]}.items():
        t, y = synth(comps, noise=2)
        r = multiexp.fit_with_comparison(t, y, 1, offset=False, compare_up_to=3)
        assert r["recommended_n"] == truth


def test_overparameterised_fit_warns_instead_of_lying():
    t, y = synth([(1000, 0.02)], noise=3)
    f = multiexp.fit(t, y, 3, offset=False)
    assert f["warnings"]


def test_too_few_echoes_raises():
    with pytest.raises(ValueError):
        multiexp.fit(np.arange(1, 6) * 1e-3, np.ones(5), 3)


def test_ilt_peak_at_true_t2_and_fractions():
    t, y = synth([(600, 0.004), (400, 0.05)], noise=1)
    r = ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8)
    peaks = r["peaks"]
    assert len(peaks) == 2
    assert peaks[0]["T2_mean"] == pytest.approx(0.004, rel=0.35)
    assert peaks[1]["T2_mean"] == pytest.approx(0.05, rel=0.35)
    assert sum(p["fraction"] for p in peaks) == pytest.approx(1)
    assert peaks[0]["fraction"] == pytest.approx(0.6, abs=0.1)


def test_alpha_sweep_shape():
    t, y = synth([(1000, 0.02)], noise=2)
    r = ilt_tools.run(t, y, 1e-4, 1.0, 60, 1e8, sweep=True)
    assert len(r["sweep"]) == 9
    rm = [s["rmse"] for s in r["sweep"]]
    assert rm[-1] <= rm[0]          # less smoothing (bigger alpha) never fits worse


def test_integrate_matches_existing_prepare_t2_data():
    d = DATA / "CPMG_EtGly" / "1"
    params = read_kea_params(d / "acqu.par")
    _, expected = prepare_t2_data(d / "data.2d", params)
    load(d)
    from nmrdash.pipeline import _STATE
    got, phase = echoes.integrate(_STATE["td"])
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert -180 <= phase <= 180


CPMG = sorted(p.parent for p in DATA.rglob("data.2d"))


@pytest.mark.parametrize("d", CPMG, ids=lambda p: str(p.relative_to(DATA)))
def test_process_all_samples_json_safe(d):
    r = load(d)
    t = r["raw"]["echo_time_s"]
    res = process({
        "echo": "integrate",
        "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": True},
        "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 10 * t[-1], "n_steps": min(100, len(t)),
                "alpha": 1e8, "sweep": False},
    })
    json.dumps(res, allow_nan=False)          # strict JSON: no NaN/inf leaks
    assert res["multiexp"]["selected"]["n"] == 2
    assert res["ilt"]["peaks"]


def test_process_requires_loaded_dataset():
    load(DATA / "T1SAT" / "1")
    load(DATA / "CPMG_EtGly" / "1")           # loading anything replaces what was loaded before
    from nmrdash.pipeline import _STATE
    _STATE.clear()                            # nothing loaded (what an unsupported folder leaves behind)
    with pytest.raises(RuntimeError, match="No supported dataset"):
        process({"echo": "integrate"})


def test_ilt_max_grid_points_user_set_and_capped_by_echoes():
    t, y = synth([(1000, 0.02)], noise=2, n=512)
    assert ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8)["n_steps"] == 100
    assert ilt_tools.run(t, y, 1e-4, 1.0, 300, 1e8)["n_steps"] == 300          # user raised the max
    assert ilt_tools.run(t, y, 1e-4, 1.0, 50000, 1e8)["n_steps"] == 512        # safety limit, then echoes
    t, y = synth([(1000, 0.02)], noise=2, n=64)
    assert ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8)["n_steps"] == 64           # capped by echo count


def test_ilt_peak_threshold_and_sweep_range():
    t, y = synth([(900, 0.004), (100, 0.05)], noise=1)
    loose = ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8, peak_min_fraction=0.01)["peaks"]
    strict = ilt_tools.run(t, y, 1e-4, 1.0, 100, 1e8, peak_min_fraction=0.9)["peaks"]
    assert len(strict) == 1 and len(loose) >= len(strict)
    s = ilt_tools.run(t, y, 1e-4, 1.0, 60, 1e8, sweep=True, sweep_range=(6, 10))["sweep"]
    assert s[0]["alpha"] == pytest.approx(1e6) and s[-1]["alpha"] == pytest.approx(1e10)


def test_poorly_determined_t2_is_flagged():
    t, y = synth([(1000, 0.02)], noise=20)
    f = multiexp.fit(t, y, 2, offset=True)
    assert any("poorly determined" in w or "within 10%" in w or "unresolved" in w for w in f["warnings"])
