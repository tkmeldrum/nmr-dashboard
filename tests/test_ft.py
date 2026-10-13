import json
from pathlib import Path

import numpy as np
import pytest

from nmrdash import constants, echoes
from nmrdash.pipeline import _STATE, load, process

DATA = Path(__file__).resolve().parent.parent / "test_data"
G, GAMMA = 23.7, 42.577          # T/m, MHz/T
DWELL_US = 1.0


def box_echoes(z_lo_um, z_hi_um, n_pts=64, n_echo=3, center=33, phase=0.7):
    """Time-domain echoes of a sample occupying [z_lo, z_hi] µm (centre of echo at point `center`)."""
    t = (np.arange(n_pts) - center) * DWELL_US * 1e-6
    z = np.linspace(z_lo_um, z_hi_um, 400)
    f = z * GAMMA * G                      # Hz
    sig = np.exp(2j * np.pi * np.outer(t, f)).sum(axis=1) * np.exp(1j * phase)
    return np.repeat(sig[:, None], n_echo, axis=1)


@pytest.mark.parametrize("lo,hi", [(-90, 90), (20, 80)])
def test_box_profile_has_right_position_and_width(lo, hi):
    td = box_echoes(lo, hi)
    r = echoes.ft(td, DWELL_US, GAMMA, G, zero_fill=4)
    assert r["center"] == 33
    prof = echoes.profile(r["S"])
    z = r["z_um"]
    above = z[prof >= 0.5 * prof.max()]
    assert above.min() == pytest.approx(lo, abs=0.2 * (hi - lo) / 2 + 12)   # within ~ the resolution (≈ 1/(N·dt·γG) ≈ 15 µm)
    assert above.max() == pytest.approx(hi, abs=0.2 * (hi - lo) / 2 + 12)
    assert (above.max() - above.min()) == pytest.approx(hi - lo, rel=0.2)


def test_autophase_puts_signal_in_real_channel():
    r = echoes.ft(box_echoes(-50, 50, phase=2.0), DWELL_US, GAMMA, G)
    S = r["S"]
    assert abs(S.sum().imag) < 1e-6 * abs(S.sum())


def test_z_axis_scales_with_gradient_and_gamma():
    td = box_echoes(-50, 50)
    z1 = echoes.ft(td, DWELL_US, GAMMA, G)["z_um"]
    z2 = echoes.ft(td, DWELL_US, GAMMA, 2 * G)["z_um"]
    np.testing.assert_allclose(z2, z1 / 2)
    zF = echoes.ft(td, DWELL_US, constants.GAMMA_MHZ_PER_T["19F"], G)["z_um"]
    np.testing.assert_allclose(zF, z1 * GAMMA / constants.GAMMA_MHZ_PER_T["19F"])


def test_window_decay_and_auto_window():
    td = box_echoes(-60, 60)
    r = echoes.ft(td, DWELL_US, GAMMA, G)
    lo, hi = echoes.auto_window(r["z_um"], echoes.profile(r["S"]))
    assert lo < -40 and hi > 40
    with pytest.raises(ValueError):
        echoes.window_decay(r["S"], r["z_um"], 1e6, 2e6)


def test_nucleus_from_b1freq():
    assert constants.guess_nucleus(19.44)[0] == "1H"
    assert constants.guess_nucleus(18.30)[0] == "19F"
    assert constants.guess_nucleus(None)[0] == "1H"
    nuc, why = constants.guess_nucleus(25.0)
    assert nuc == "1H" and "neither" in why


def test_load_reports_default_nucleus():
    r = load(DATA / "CPMG_EtGly" / "1")
    assert r["defaults"]["nucleus"] == "1H" and "19.44" in r["defaults"]["nucleus_reason"]


CPMG = sorted(p.parent for p in DATA.rglob("data.2d"))


@pytest.mark.parametrize("d", CPMG, ids=lambda p: str(p.relative_to(DATA)))
def test_ft_runs_on_every_sample_including_two_point_data(d):
    r = load(d)
    n = len(r["raw"]["echo_time_s"])
    res = process({
        "echo": "ft", "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4, "z_min": None, "z_max": None},
        "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False},
        "ilt": {"enabled": False},
    })
    json.dumps(res, allow_nan=False)
    ft = res["ft"]
    assert len(ft["z_um"]) == len(ft["profile"]) and ft["window"][0] < ft["window"][1] and ft["window_auto"]
    assert len(res["decay"]["y"]) == n
    assert len(ft["heat"]["mag"]) == len(ft["heat"]["z_um"]) == ft["nfft"]      # full zero-filled grid, not thinned


def test_ft_decay_tracks_integrated_decay_on_good_data():
    load(DATA / "CPMG_EtGly" / "1")
    td = _STATE["td"]
    integ, _ = echoes.integrate(td)
    r = echoes.ft(td, 1.0, GAMMA, G)
    y = echoes.window_decay(r["S"], r["z_um"], *echoes.auto_window(r["z_um"], echoes.profile(r["S"])))
    assert np.corrcoef(integ, y)[0, 1] > 0.9


def test_user_window_and_bad_gradient():
    load(DATA / "CPMG_EtGly" / "1")
    ok = process({"echo": "ft", "ft": {"nucleus": "19F", "G": 20, "z_min": -100, "z_max": 100},
                  "multiexp": {"enabled": False}, "ilt": {"enabled": False}})
    assert ok["ft"]["window"] == [-100.0, 100.0] and not ok["ft"]["window_auto"] and ok["ft"]["nucleus"] == "19F"
    with pytest.raises(ValueError):
        process({"echo": "ft", "ft": {"G": 0}, "multiexp": {"enabled": False}, "ilt": {"enabled": False}})
