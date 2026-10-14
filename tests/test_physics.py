"""The physical relations in nmrdash/physics.py, checked against hand calculations and (for b-values) against the spectrometer itself."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from nmrdash import physics as ph, t2d
from nmrdash.kea_io import read_kea_params
from nmrdash.pipeline import _STATE, load, process, t2d_axes

DATA = Path(__file__).resolve().parent.parent / "test_data"
ROOT_DIR = Path(__file__).resolve().parent.parent
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


# ── the signal models, each against an independent calculation ───────────────

def test_exponential_decay_at_known_points():
    T2 = 0.03
    assert ph.exp_decay(0.0, T2) == 1.0
    assert ph.exp_decay(T2, T2) == pytest.approx(1 / np.e)                                # 1/e at t = T2
    assert ph.exp_decay(T2 * np.log(2), T2) == pytest.approx(0.5)                          # half-life = T2 ln 2
    t = np.linspace(0, 0.2, 50)
    assert np.all(np.diff(ph.exp_decay(t, T2)) < 0)
    assert ph.exp_decay(t, T2) == pytest.approx(np.exp(-t / T2))                           # and the vectorised form agrees


def test_multi_exponential_is_the_sum_of_single_decays_plus_the_offset():
    t = np.linspace(0, 0.5, 40)
    A, T2, c = [3.0, 1.5, 0.5], [0.004, 0.03, 0.2], 0.7
    expected = c + sum(a * np.exp(-t / T) for a, T in zip(A, T2))
    assert ph.multi_exp(t, A, T2, c) == pytest.approx(expected)
    assert ph.multi_exp(t, A, T2) == pytest.approx(expected - c)
    assert ph.multi_exp(0.0, A, T2, c) == pytest.approx(sum(A) + c)                        # the amplitudes add up at t = 0
    assert ph.multi_exp(t, [2.0], [0.01]) == pytest.approx(2.0 * ph.exp_decay(t, 0.01))


def test_saturation_recovery_starts_at_zero_and_reaches_the_plateau():
    T1 = 0.4
    assert ph.saturation_recovery(0.0, T1) == 0.0
    assert ph.saturation_recovery(T1, T1) == pytest.approx(1 - 1 / np.e)                   # 63% recovered after T1
    assert ph.saturation_recovery(40 * T1, T1) == pytest.approx(1.0)
    t = np.linspace(0, 3, 30)
    assert ph.multi_saturation(t, [2.0, 1.0], [0.1, 0.8], 0.05) == pytest.approx(0.05 + 2 * (1 - np.exp(-t / 0.1)) + (1 - np.exp(-t / 0.8)))
    assert np.all(ph.saturation_recovery(t, T1) + ph.exp_decay(t, T1) == pytest.approx(1.0))     # recovery and decay are complements


def test_diffusion_attenuation():
    assert ph.diffusion_attenuation(0.0, 1e-9) == 1.0
    assert ph.diffusion_attenuation(1e9, 1e-9) == pytest.approx(1 / np.e)                  # bD = 1
    assert ph.diffusion_attenuation(2e9, 3e-10) == pytest.approx(np.exp(-0.6))


def test_the_t2d_signal_factorises_for_one_population_and_adds_for_several():
    t, b = np.linspace(1e-3, 0.1, 30), np.array([1e9, 1e10, 1e11])
    one = ph.t2d_signal(t, b, [2.0], [0.02], [2e-11])
    assert one.shape == (30, 3)
    assert one == pytest.approx(2.0 * np.outer(np.exp(-t / 0.02), np.exp(-b * 2e-11)))     # the product of the T2 decay and the diffusion factor
    two = ph.t2d_signal(t, b, [2.0, 1.0], [0.02, 0.005], [2e-11, 5e-10])
    assert two == pytest.approx(one + ph.t2d_signal(t, b, [1.0], [0.005], [5e-10]))
    E, G = ph.t2d_factors(t, b, [0.02, 0.005], [2e-11, 5e-10])
    assert E.shape == (30, 2) and G.shape == (3, 2) and E[:, 0] == pytest.approx(np.exp(-t / 0.02))


def test_the_kernels_reproduce_the_models_on_a_grid():
    t = np.linspace(1e-3, 0.3, 25)
    grid = np.geomspace(1e-3, 1.0, 12)
    K = ph.t2_kernel(t, grid)
    assert K.shape == (25, 12) and K[3, 5] == pytest.approx(np.exp(-t[3] / grid[5]))
    f = np.random.default_rng(0).uniform(0, 1, 12)
    assert K @ f == pytest.approx(ph.multi_exp(t, f, grid))                                 # a spectrum f through the kernel is the multi-exponential sum
    b, Dg = np.array([1e9, 4e9, 1e10]), np.geomspace(1e-12, 1e-9, 7)
    Kd = ph.diffusion_kernel(b, Dg)
    assert Kd.shape == (3, 7) and Kd[1, 2] == pytest.approx(np.exp(-b[1] * Dg[2]))
    F = np.random.default_rng(1).uniform(0, 1, (12, 7))                                     # a T2 x D distribution
    s2d = K @ F @ Kd.T                                                                      # the separable kernel applied to it
    direct = sum(F[i, j] * np.outer(np.exp(-t / grid[i]), np.exp(-b * Dg[j])) for i in range(12) for j in range(7))
    assert s2d == pytest.approx(direct)
    full = np.kron(K, Kd) @ F.reshape(-1)                                                   # K2D = kron(K1D, KD) on the flattened distribution
    assert full.reshape(25, 3) == pytest.approx(direct)


def test_position_from_frequency_by_hand():
    # 1 kHz in 42.577 MHz/T at 23.67 T/m: 1e3 / (42.577 * 23.67) = 0.99226 micrometres
    assert ph.z_um_from_frequency(1e3, 42.577, 23.67) == pytest.approx(1e3 / (42.577 * 23.67))
    assert ph.z_um_from_frequency(1e3, 42.577, 23.67) == pytest.approx(0.99226, rel=1e-4)
    assert ph.z_um_from_frequency(-1e3, 42.577, 23.67) == pytest.approx(-0.99226, rel=1e-4)       # negative frequency, negative z
    assert ph.native_resolution_um(64, 1.0, 42.577, 23.67) == pytest.approx(1e6 / 64 / (42.577 * 23.67))
    assert ph.native_resolution_um(64, 1.0, 42.577, 23.67) == pytest.approx(15.5, abs=0.05)
    # the resolution is the spacing of the frequency grid of N points at dwell 1 us, turned into length
    f = np.fft.fftfreq(64, d=1e-6)
    assert np.diff(ph.z_um_from_frequency(np.fft.fftshift(f), 42.577, 23.67)).mean() == pytest.approx(ph.native_resolution_um(64, 1.0, 42.577, 23.67))


def test_the_resolvable_diffusion_window_by_definition():
    lo, hi = ph.resolvable_diffusion_window([6.4, 100.0, 2885.0])                           # b in 1e9 s/m^2
    assert lo == pytest.approx(1 / 2.885e12) and hi == pytest.approx(5 / 6.4e9)


def test_every_function_in_the_physics_and_inference_modules_is_listed_on_the_about_page():
    """Nothing can sit in these files without appearing in the list a reviewer is shown (and so the list cannot silently leave something out)."""
    import inspect
    from nmrdash import inference
    for module, registry in ((ph, ph.MODELS), (inference, inference.METHODS)):
        listed = {f.__name__ for m in registry for f in m["functions"]}
        defined = {n for n, f in inspect.getmembers(module, inspect.isfunction) if f.__module__ == module.__name__ and not n.startswith("_")}
        assert defined == listed, (module.__name__, defined ^ listed)
        for m in registry:
            assert {"id", "name", "equation", "describes", "functions"} <= set(m) and m["functions"] and all(f.__doc__ for f in m["functions"])
        assert len({m["id"] for m in registry}) == len(registry)


def test_the_about_page_links_every_listed_function_to_its_lines(tmp_path):
    import re
    import sys
    sys.path.insert(0, str(ROOT_DIR / "tools"))
    from build_site import build
    from nmrdash import inference
    out = build(tmp_path / "site", "v1", vendor=False)
    about = (out / "physics.html").read_text()
    assert "{{" not in about and "{{" not in (out / "about.html").read_text()
    index = (out / "about.html").read_text()
    for m in ph.MODELS + inference.METHODS:
        assert f'href="physics.html#{m["id"]}"' in index and f'id="{m["id"]}"' in about              # About has a short index; each model has a card with that anchor
    for m in ph.MODELS + inference.METHODS:
        assert m["name"] in about.replace("&amp;", "&") or m["name"].replace("&", "&amp;") in about
        for f in m["functions"]:
            assert re.search(rf'blob/[^"]+/nmrdash/{f.__module__.split(".")[-1]}\.py#L\d+-L\d+"><code>{f.__module__.split(".")[-1]}\.py: {f.__name__}</code>', about), f.__name__
    assert "Diffusion during the CPMG echo train" in about                                  # the footnote on what is not modelled


# ── references: kept in a document to be checked, shown on the page only once verified ──

def test_the_references_document_matches_its_data_and_covers_every_listed_model():
    import json
    import sys
    sys.path.insert(0, str(ROOT_DIR / "tools"))
    import make_references
    from nmrdash import inference
    assert (ROOT_DIR / "docs" / "REFERENCES.md").read_text() == make_references.render()           # regenerate with tools/make_references.py after editing the JSON
    refs = {r["id"]: r for r in json.loads((ROOT_DIR / "docs" / "references.json").read_text())["references"]}
    for m in ph.MODELS + inference.METHODS:
        assert m["id"] in refs and refs[m["id"]]["reference"], m["id"]
    assert "source" not in {k for m in ph.MODELS + inference.METHODS for k in m}


def test_the_about_page_shows_only_verified_references(tmp_path, monkeypatch):
    import json
    import sys
    sys.path.insert(0, str(ROOT_DIR / "tools"))
    import build_site
    refs = json.loads((ROOT_DIR / "docs" / "references.json").read_text())["references"]
    out = build_site.build(tmp_path / "a", "v1", vendor=False)
    about = (out / "about.html").read_text() + (out / "physics.html").read_text()
    for r in refs:
        if not r.get("verified"):
            assert r["reference"].split(";")[0] not in about, r["id"]                                  # nothing unchecked on the page
    assert "Where it comes from" not in about                                                          # none are verified yet
    monkeypatch.setattr(build_site, "verified_references", lambda: {"exp_decay": "A checked reference (2020)"})
    out2 = build_site.build(tmp_path / "b", "v1", vendor=False)
    about2 = (out2 / "physics.html").read_text()
    assert about2.count("Where it comes from") == 1 and "A checked reference (2020)" in about2
    docs = [(ROOT_DIR / "nmrdash" / f).read_text() for f in ("physics.py", "inference.py")]
    assert not any("Hahn (1950)" in d or "Carr and Purcell (1954)" in d for d in docs)                  # the code carries a pointer, not the citations
