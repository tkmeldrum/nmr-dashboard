import json
from pathlib import Path

import numpy as np
import pytest

from nmrdash import echoes, spatial
from nmrdash.pipeline import load, process

DATA = Path(__file__).resolve().parent.parent / "test_data"
G, GAMMA, DWELL = 23.7, 42.577, 1.0


def two_zone_dataset(noise=0.0, n_echo=128, dt=2e-4, n_pts=64, center=33):
    """z in [-100,0] has T2 = 10 ms, z in [0,100] has T2 = 40 ms; unit density."""
    t_echo = dt * np.arange(1, n_echo + 1)
    t = (np.arange(n_pts) - center) * DWELL * 1e-6
    z = np.linspace(-100, 100, 400)
    T2 = np.where(z < 0, 0.010, 0.040)
    f = z * GAMMA * G
    carrier = np.exp(2j * np.pi * np.outer(t, f))                      # (pts, z)
    decay = np.exp(-np.outer(z * 0 + 1, t_echo) / T2[:, None])         # (z, echoes)
    td = carrier @ decay
    rng = np.random.default_rng(3)
    td = td + noise * (rng.standard_normal(td.shape) + 1j * rng.standard_normal(td.shape))
    return t_echo, td


def test_spatially_resolved_t2_recovers_two_zones():
    t_echo, td = two_zone_dataset(noise=0.5)
    r = echoes.ft(td, DWELL, GAMMA, G)
    lo, hi = echoes.auto_window(r["z_um"], echoes.profile(r["S"]))
    idx = echoes.position_indices(r["z_um"], lo, hi, 15.0)
    Y = echoes.position_decays(r["S"], idx)
    out = spatial.fit_positions(t_echo, Y, r["z_um"][idx], {"enabled": True, "n": 1, "offset": False, "ci_level": 0.95}, None)
    z, T2 = np.array(out["z_um"]), out["multiexp"]["T2"][0]
    assert T2[(z > -70) & (z < -25)] == pytest.approx(0.010, rel=0.2)
    assert T2[(z > 25) & (z < 70)] == pytest.approx(0.040, rel=0.2)
    assert out["multiexp"]["ok"].all()


def test_ilt_map_has_two_regimes():
    t_echo, td = two_zone_dataset(noise=0.5)
    r = echoes.ft(td, DWELL, GAMMA, G)
    idx = echoes.position_indices(r["z_um"], -60, 60, 20.0)
    Y = echoes.position_decays(r["S"], idx)
    il = {"enabled": True, "t2_min": 1e-3, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}
    out = spatial.fit_positions(t_echo, Y, r["z_um"][idx], None, il)["ilt"]
    z, gm = r["z_um"][idx], out["mean_T2"]
    assert np.nanmean(gm[z < -20]) == pytest.approx(0.010, rel=0.4)
    assert np.nanmean(gm[z > 20]) == pytest.approx(0.040, rel=0.4)
    assert len(out["spectra"]) == len(z) and len(out["tau"]) == 60


def test_position_indices_spacing_and_limits():
    z = np.linspace(-500, 500, 1024)
    idx = echoes.position_indices(z, -100, 100, 20)
    assert len(idx) == 11 and np.all(np.diff(z[idx]) == pytest.approx(20, abs=1.5))
    with pytest.raises(ValueError):
        echoes.position_indices(z, -500, 500, 1)          # 1001 positions
    with pytest.raises(ValueError):
        echoes.position_indices(z, -1, 1, 0)


def test_native_resolution_value():
    assert echoes.native_resolution_um(64, 1.0, GAMMA, G) == pytest.approx(15.5, abs=0.1)


def test_progress_callback_reports_every_position():
    t_echo, td = two_zone_dataset(noise=0.5)
    r = echoes.ft(td, DWELL, GAMMA, G)
    idx = echoes.position_indices(r["z_um"], -60, 60, 30.0)
    seen = []
    spatial.fit_positions(t_echo, echoes.position_decays(r["S"], idx), r["z_um"][idx],
                          {"enabled": True, "n": 1, "offset": True}, None, progress=lambda i, n: seen.append((i, n)))
    assert seen[-1] == (len(idx), len(idx)) and len(seen) == len(idx)


def test_process_ft_adds_spatial_results_on_real_data():
    load(DATA / "CPMG_EtGly" / "1")
    res = process({
        "echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4},
        "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": False},
        "ilt": {"enabled": True, "t2_min": 1e-4, "t2_max": 1.0, "n_steps": 100, "alpha": 1e8},
    })
    json.dumps(res, allow_nan=False)
    sp, ft = res["spatial"], res["ft"]
    assert len(sp["z_um"]) == ft["n_positions"] > 5 and ft["spacing_auto"] and ft["spacing_um"] == pytest.approx(15, abs=1)
    assert len(sp["multiexp"]["T2"]) == 2 and len(sp["multiexp"]["T2"][0]) == ft["n_positions"]
    assert len(sp["ilt"]["spectra"]) == ft["n_positions"]


def test_integrate_mode_has_no_spatial():
    load(DATA / "CPMG_EtGly" / "1")
    res = process({"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True}, "ilt": {"enabled": False}})
    assert res["spatial"] is None


CPMG = sorted(p.parent for p in DATA.rglob("data.2d"))


@pytest.mark.parametrize("d", CPMG, ids=lambda p: str(p.relative_to(DATA)))
def test_spatial_runs_on_every_sample(d):
    load(d)
    res = process({
        "echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4},
        "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95},
        "ilt": {"enabled": True, "t2_min": 1e-5, "t2_max": 1.0, "n_steps": 100, "alpha": 1e8},
    })
    json.dumps(res, allow_nan=False)
    assert res["spatial"]["z_um"]
