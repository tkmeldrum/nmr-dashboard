import json
from pathlib import Path

import numpy as np
import pytest

from nmrdash import echoes, spatial
from nmrdash.pipeline import load, process

from sample_data import CPMG_SNR100, DATA
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
    out = spatial.fit_positions(t_echo, (r["z_um"][idx], Y), None, {"enabled": True, "n": 1, "offset": False, "ci_level": 0.95}, None)
    z, T2 = np.array(out["multiexp"]["z_um"]), out["multiexp"]["T2"][0]
    assert T2[(z > -70) & (z < -25)] == pytest.approx(0.010, rel=0.2)
    assert T2[(z > 25) & (z < 70)] == pytest.approx(0.040, rel=0.2)
    assert out["multiexp"]["ok"].all()


def test_ilt_map_has_two_regimes():
    t_echo, td = two_zone_dataset(noise=0.5)
    r = echoes.ft(td, DWELL, GAMMA, G)
    idx = echoes.position_indices(r["z_um"], -60, 60, 20.0)
    Y = echoes.position_decays(r["S"], idx)
    il = {"enabled": True, "t2_min": 1e-3, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}
    out = spatial.fit_positions(t_echo, None, (r["z_um"][idx], Y), None, il)["ilt"]
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
    pos = (r["z_um"][idx], echoes.position_decays(r["S"], idx))
    il = {"enabled": True, "t2_min": 1e-3, "t2_max": 1.0, "n_steps": 40, "alpha": 1e8}
    spatial.fit_positions(t_echo, pos, pos, {"enabled": True, "n": 1, "offset": True}, il, progress=lambda i, n: seen.append((i, n)))
    assert len(seen) == 2 * len(idx) and seen[-1] == (2 * len(idx), 2 * len(idx))      # both analyses report, one total


def test_process_ft_adds_spatial_results_on_real_data():
    load(CPMG_SNR100)
    res = process({
        "echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4},
        "multiexp": {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": False},
        "ilt": {"enabled": True, "t2_min": 1e-4, "t2_max": 1.0, "n_steps": 100, "alpha": 1e8},
    })
    json.dumps(res, allow_nan=False)
    sp, ft = res["spatial"], res["ft"]
    assert len(sp["multiexp"]["z_um"]) == ft["n_positions"] > 5 and ft["spacing_auto"] and ft["spacing_um"] == pytest.approx(15, abs=1)
    assert len(sp["multiexp"]["T2"]) == 2 and len(sp["multiexp"]["T2"][0]) == ft["n_positions"]
    # the ILT map sits on the zero-filled grid, not the natural-resolution grid
    assert len(sp["ilt"]["z_um"]) == len(sp["ilt"]["spectra"]) == ft["n_ilt_positions"] > 5 * ft["n_positions"]
    assert ft["zf_step_um"] < 1.5


def test_integrate_mode_has_no_spatial():
    load(CPMG_SNR100)
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
    assert res["spatial"]["ilt"]["z_um"]


def test_heat_map_uses_full_zero_filled_grid():
    load(CPMG_SNR100)
    res = process({"echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4},
                   "multiexp": {"enabled": False}, "ilt": {"enabled": False}})
    ft = res["ft"]
    assert len(ft["heat"]["z_um"]) == ft["nfft"] == 1024                 # not thinned
    assert len(ft["heat"]["mag"]) == 1024


def test_ilt_bands_track_the_two_zones():
    t_echo, td = two_zone_dataset(noise=0.5)
    r = echoes.ft(td, DWELL, GAMMA, G)
    idx = echoes.grid_indices(r["z_um"], -80, 80)
    Y = echoes.position_decays(r["S"], idx)
    il = {"enabled": True, "t2_min": 1e-3, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}
    out = spatial.fit_positions(t_echo, None, (r["z_um"][idx], Y), None, il)["ilt"]
    assert len(out["bands"]) >= 2
    z = np.array(out["z_um"])
    fast = out["bands"].index(min(out["bands"], key=lambda b: b["t2_peak"]))
    slow = out["bands"].index(max(out["bands"], key=lambda b: b["t2_peak"]))
    assert np.nanmean(out["band_fraction"][fast][z < -30]) > 0.6          # fast band dominates the T2=10 ms side
    assert np.nanmean(out["band_fraction"][slow][z > 30]) > 0.6           # slow band dominates the T2=40 ms side
    assert np.nanmean(out["band_mean_T2"][slow][z > 30]) == pytest.approx(0.040, rel=0.4)
    for b in range(len(out["bands"])):
        assert len(out["band_mean_T2"][b]) == len(out["band_fraction"][b]) == len(z)


def test_grid_indices_and_99pct_ci():
    z = np.linspace(-500, 500, 1024)
    assert len(echoes.grid_indices(z, -200, 200)) == pytest.approx(410, abs=3)
    with pytest.raises(ValueError):
        echoes.grid_indices(z, 900, 950)
    from nmrdash import multiexp
    t = 2e-4 * np.arange(1, 257)
    y = 1000 * np.exp(-t / 0.02) + np.random.default_rng(0).normal(0, 3, 256)
    w95 = np.diff(multiexp.fit(t, y, 1, ci_level=0.95)["components"][0]["T2_ci"])[0]
    w99 = np.diff(multiexp.fit(t, y, 1, ci_level=0.99)["components"][0]["T2_ci"])[0]
    assert w99 > w95


def test_default_window_is_plus_minus_200_and_exposed_to_ui():
    r = load(CPMG_SNR100)
    assert r["defaults"]["z_window_um"] == [-200.0, 200.0]


def test_noise_only_positions_are_skipped_and_progress_counts_only_fits():
    t_echo, td = two_zone_dataset(noise=0.5)
    r = echoes.ft(td, DWELL, GAMMA, G)
    idx = echoes.position_indices(r["z_um"], -300, 300, 30.0)           # sample only occupies +-100 um
    prof = echoes.profile(r["S"])
    keep = (prof / prof.max())[idx] >= 0.05
    assert 0 < keep.sum() < len(idx)
    seen = []
    out = spatial.fit_positions(t_echo, (r["z_um"][idx], echoes.position_decays(r["S"], idx), keep), None,
                                {"enabled": True, "n": 1, "offset": False}, None, progress=lambda i, n: seen.append((i, n)))
    T2 = out["multiexp"]["T2"][0]
    assert np.isnan(T2[~keep]).all() and np.isfinite(T2[keep]).all()
    assert len(seen) == keep.sum() and seen[-1] == (keep.sum(), keep.sum())


def test_pipeline_threshold_and_error_when_nothing_has_signal():
    load(CPMG_SNR100)
    base = {"echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4, "z_min": -200, "z_max": 200},
            "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95}, "ilt": {"enabled": False}}
    res = process(base)
    ft = res["ft"]
    assert ft["min_signal_pct"] == 5 and 0 < ft["n_positions_kept"] < ft["n_positions"]
    assert sum(v is None for v in res["spatial"]["multiexp"]["T2"][0]) == ft["n_positions"] - ft["n_positions_kept"]
    base["ft"]["min_signal_pct"] = 0
    assert process(base)["ft"]["n_positions_kept"] == ft["n_positions"]            # 0% fits everything
    base["ft"]["min_signal_pct"] = 150
    with pytest.raises(ValueError):
        process(base)
