"""The physical relations in nmrdash/physics.py, checked against hand calculations and (for b-values) against the spectrometer itself."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from nmrdash import physics as ph, t2d
from nmrdash.kea_io import read_kea_params
from nmrdash.pipeline import _STATE, load, process, t2d_axes

DATA = Path(__file__).resolve().parent.parent / "test_data"
M1115, S3115 = DATA / "M1115_rt_30phr" / "1", DATA / "S3115_rt_100phr" / "1"


def test_the_stimulated_echo_b_value_by_hand():
    """tau = 1 ms, Delta = 5 ms, G = 23.67 T/m, 1H: gamma = 2 pi 42.577e6 = 2.6752e8 rad/s/T; (gamma G)^2 = 4.0097e19; tau^2 (Delta + 2 tau / 3) = 1e-6 * 5.6667e-3."""
    gamma = 2 * np.pi * 42.577e6
    expected = (gamma * 23.67) ** 2 * (1e-3) ** 2 * (5e-3 + 2e-3 / 3)
    assert ph.b_value_stimulated_echo(1e-3, 5e-3, 23.67) == pytest.approx(expected, rel=1e-12)
    assert ph.b_value_stimulated_echo(1e-3, 5e-3, 23.67) == pytest.approx(2.2722e11, rel=1e-3)          # the same number worked out separately


def test_b_scales_as_the_square_of_the_gradient_and_of_tau_and_grows_with_delta():
    b = lambda tau, delta, G: ph.b_value_stimulated_echo(tau, delta, G)
    assert b(1e-3, 5e-3, 2 * 23.67) == pytest.approx(4 * b(1e-3, 5e-3, 23.67))
    assert b(2e-3, 0.0, 23.67) / b(1e-3, 0.0, 23.67) == pytest.approx(8.0)                                      # Delta = 0: b ~ (2/3) tau^3
    assert b(1e-3, 10e-3, 23.67) > b(1e-3, 5e-3, 23.67) > b(1e-3, 0.0, 23.67)
    assert ph.b_value_stimulated_echo(1e-3, 5e-3, 23.67, "19F") == pytest.approx(b(1e-3, 5e-3, 23.67) * (40.078 / 42.577) ** 2)


def test_the_tau_grid_is_geometric_and_hits_both_ends():
    g = ph.geometric_grid(0.178e-3, 3.2e-3, 25)
    assert g[0] == pytest.approx(0.178e-3) and g[-1] == pytest.approx(3.2e-3) and len(g) == 25
    assert np.allclose(g[1:] / g[:-1], g[1] / g[0])


@pytest.mark.parametrize("sample", [M1115, S3115])
def test_the_calculated_b_values_have_the_spectrometers_shape_and_scale(sample):
    """The spectrometer's own b-values follow exactly tau^2 (Delta + 2 tau / 3) (shape deviation ~1e-6) and correspond to the gradient implied by its `gradient` parameter."""
    params = read_kea_params(sample / "acqu.par")
    L = t2d.load(sample, params)
    chk = t2d.check_axes(L, ph.DEFAULT_GRADIENT_T_PER_M)
    shape = next(c for c in chk["compared"] if c["kind"] == "shape")
    assert chk["agrees"] and shape["value"] < 1e-5
    kea = ph.gradient_implied_by_kea(params["gradient"])
    assert kea == pytest.approx(23.487, abs=0.001)
    assert shape["implied_gradient_t_per_m"] == pytest.approx(kea, rel=5e-4)                      # 0.03%: Kea's rounding of gamma
    tau = next(c for c in chk["compared"] if c["what"] == "delays tau")
    assert tau["value"] < 1e-5 and tau["agrees"]


def test_the_default_gradient_is_one_constant_used_for_the_profile_and_the_diffusion_weighting():
    from nmrdash import constants, params as P
    assert ph.DEFAULT_GRADIENT_T_PER_M == 23.67 == constants.DEFAULT_G_T_PER_M == constants.T2D_DEFAULTS["gradient"]
    assert P.defaults()["ft"]["G"] == 23.67 and P.defaults()["t2d"]["gradient"] == 23.67


def test_the_gradient_setting_rescales_b_and_the_resolvable_range_in_processing():
    load(M1115)
    a, b = t2d_axes({"t2d": {"gradient": 23.67}}), t2d_axes({"t2d": {"gradient": 2 * 23.67}})
    assert np.allclose(np.array(b["b_vec"]) / np.array(a["b_vec"]), 4.0)
    assert b["d_window"][1] == pytest.approx(a["d_window"][1] / 4)                                 # 5 / b_min
    res = process({"kind": "t2d", "t2d": {"t2_steps": 20, "d_steps": 10, "gradient": 30.0}})["t2d"]
    assert res["b_info"]["gradient_t_per_m"] == 30.0
    assert res["b_vec"] == pytest.approx(list(t2d.b_values(_STATE["t2d"], 30.0)))
    # the spectrometer's scale is reported as a gradient, whatever is used here
    assert next(c for c in res["b_check"]["compared"] if c["kind"] == "shape")["implied_gradient_t_per_m"] == pytest.approx(23.4908, abs=0.002)
    with pytest.raises(ValueError, match="gradient"):
        process({"kind": "t2d", "t2d": {"gradient": 0}})


def test_missing_acquisition_parameters_are_named(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(M1115, d)
    text = (d / "acqu.par").read_text().replace("DELTA = 5", "")
    (d / "acqu.par").write_text(text)
    info = load(d)
    assert not info["supported"] and "DELTA" in info["reason"]


def test_a_spectrometer_file_with_the_wrong_shape_is_flagged(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(M1115, d)
    a = np.loadtxt(d / "data.dat", ndmin=2)
    a[:, 1] *= np.linspace(1.0, 1.5, len(a))                                                       # not tau^2 (Delta + 2 tau / 3) any more
    np.savetxt(d / "data.dat", a)
    L = t2d.load(d, read_kea_params(d / "acqu.par"))
    assert not t2d.check_axes(L, ph.DEFAULT_GRADIENT_T_PER_M)["agrees"]
