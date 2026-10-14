"""The estimation formulas in nmrdash/inference.py, against hand calculations and known statistical facts."""
import numpy as np
import pytest
from scipy import stats

from nmrdash import inference as inf


def test_noise_from_a_noise_only_channel_recovers_the_standard_deviation():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 7.0, 20000)
    assert inf.noise_sigma_mad(x) == pytest.approx(7.0, rel=0.03)
    x[:50] = 1e6                                                                           # a few wild outliers hardly matter (the SD would be ruined)
    assert inf.noise_sigma_mad(x) == pytest.approx(7.0, rel=0.03) and np.std(x) > 1000
    assert inf.noise_sigma_mad([1.0, 2.0]) is None and inf.noise_sigma_mad(np.ones(20)) is None


def test_phasing_puts_the_signal_in_the_real_channel_and_integrates():
    n_pts, n_echo = 16, 5
    shape = np.exp(-0.5 * ((np.arange(n_pts) - 8) / 2.0) ** 2)
    amp = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    td = np.outer(shape, amp) * np.exp(1j * 0.7)                                           # a pure phase of 0.7 rad
    decay, deg = inf.phase_and_integrate(td)
    assert deg == pytest.approx(-np.degrees(0.7)) and decay == pytest.approx(shape.sum() * amp)
    real, deg0 = inf.phase_and_integrate(np.outer(shape, amp))
    assert deg0 == 0.0 and real == pytest.approx(shape.sum() * amp)


def test_fit_measures_by_hand():
    y, m = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.0])
    assert inf.sum_squared_residuals(y, m) == pytest.approx(0.25 + 1.0)
    assert inf.reduced_chi2(1.25, 0.5, 5) == pytest.approx(1.25 / (0.25 * 5))
    assert inf.reduced_chi2(1.25, None, 5) is None
    assert inf.chi2_per_point(1.25, 0.5, 10) == pytest.approx(0.5) and inf.chi2_per_point(1.25, None, 10) is None


def test_information_criteria_by_hand():
    rss, n, k = 4.0, 100, 5
    aic, bic = inf.information_criteria(rss, n, k)
    assert aic == pytest.approx(100 * np.log(0.04) + 10) and bic == pytest.approx(100 * np.log(0.04) + 5 * np.log(100))
    a2, b2 = inf.information_criteria(rss, n, k + 2)                                       # more parameters at the same RSS: both get worse, BIC more (n > e^2)
    assert a2 - aic == pytest.approx(4) and b2 - bic == pytest.approx(2 * np.log(100)) and b2 - bic > a2 - aic


def test_the_t_multiplier_is_the_normal_one_for_many_dof_and_larger_for_few():
    assert inf.student_t_critical(0.95, 1_000_000) == pytest.approx(1.95996, abs=1e-4)
    assert inf.student_t_critical(0.68, 1_000_000) == pytest.approx(0.99445, abs=1e-3)
    assert inf.student_t_critical(0.95, 3) == pytest.approx(3.1824, abs=1e-3)              # from the t table
    assert inf.student_t_critical(0.95, 3) == pytest.approx(stats.t.ppf(0.975, 3))


def test_the_curvature_penalty_ignores_straight_lines_and_sees_bends():
    D2 = inf.second_derivative_matrix(8)
    assert D2.shape == (6, 8)
    x = np.arange(8.0)
    assert D2 @ x == pytest.approx(np.zeros(6))                                            # a straight line has no curvature
    assert D2 @ (x ** 2) == pytest.approx(np.full(6, 2.0))                                 # the second difference of x^2 is 2
    assert D2[0, :3].tolist() == [1, -2, 1]


def test_the_regularisation_weight_by_hand_and_in_the_right_direction():
    assert inf.tikhonov_weighting(1e8, 0.1, 0.01) == pytest.approx(1 / np.sqrt(0.01 ** 2 * 0.1 ** 3 * 1e8))
    assert inf.tikhonov_weighting(1e9, 0.1, 0.01) < inf.tikhonov_weighting(1e8, 0.1, 0.01)      # larger alpha: smaller weight: less smoothing
