"""Excluding the first n echoes (T1: delays) from every fit: one setting, applied once, inherited by every analysis."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nmrdash import echoes, multiexp, params as P, runner
from nmrdash.kea_io import read_kea_binary
from nmrdash.pipeline import _STATE, _skip_count, boot_finish, boot_start, boot_step, load, mcmc_finish, mcmc_start, mcmc_step, process

from sample_data import CPMG_CLEAN, CPMG_NOISY, T1_CLEAN, TRUTH, write_kea_binary
CPMG = {"echo": "integrate", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": False},
        "ilt": {"enabled": True, "t2_min": 1e-4, "t2_max": 1.0, "n_steps": 60, "alpha": 1e8}}
T1 = {"kind": "t1", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": True}}


def test_skip_count_is_validated():
    assert _skip_count({}, 100) == 0 and _skip_count({"exclude_first": None}, 100) == 0 and _skip_count({"exclude_first": "3"}, 100) == 3
    with pytest.raises(ValueError, match="negative"):
        _skip_count({"exclude_first": -1}, 100)
    with pytest.raises(ValueError, match="whole number"):
        _skip_count({"exclude_first": "two"}, 100)
    with pytest.raises(ValueError, match="leaves 5"):
        _skip_count({"exclude_first": 95}, 100)


def test_fits_use_only_the_included_echoes_and_the_full_decay_is_still_returned():
    load(CPMG_NOISY)
    base = process(CPMG)
    res = process({**CPMG, "exclude_first": 3})
    n = len(base["decay"]["t"])
    assert base["decay"]["skip"] == 0 and res["decay"]["skip"] == 3
    assert len(res["decay"]["t"]) == len(res["decay"]["y"]) == n                         # everything is still there to plot
    me, il = res["multiexp"]["selected"], res["ilt"]
    assert len(me["fit"]) == len(me["resid"]) == len(il["pred"]) == len(il["resid"]) == n - 3
    # exactly the fit one gets by handing the included points to the engine
    t, y = np.array(res["decay"]["t"])[3:], np.array(res["decay"]["y"])[3:]
    sigma = echoes.noise_sigma(echoes.integrate_noise_series(_STATE["td"])[3:])
    direct = multiexp.fit_with_comparison(t, y, 1, True, 0.95, 1, sigma=sigma)["selected"]
    assert me["components"][0]["T2"] == pytest.approx(direct["components"][0]["T2"], rel=1e-6)
    assert res["decay"]["noise_sigma"] == pytest.approx(sigma)
    assert me["dof"] == base["multiexp"]["selected"]["dof"] - 3


def test_excluding_a_distorted_first_echo_improves_the_fit(tmp_path):
    """A first echo far off the trend (a real instrument artefact; made here by scaling it in a copy) ruins the fit; leaving it out repairs it."""
    d = tmp_path / "ds"
    shutil.copytree(CPMG_CLEAN, d)
    h, spec = read_kea_binary(d / "data.2d")
    td = spec.reshape((h["xDim"], h["yDim"]), order="F").copy()
    td[:, 0] *= 0.3
    write_kea_binary(d / "data.2d", td, 501)
    load(d)
    ft = {"nucleus": "1H", "G": 23.7, "zero_fill": 4, "z_min": -200.0, "z_max": 200.0}
    cfg = {"echo": "ft", "ft": ft, "multiexp": {**CPMG["multiexp"], "n": 2}, "ilt": {"enabled": False}}
    a = process(cfg)["multiexp"]["selected"]
    b = process({**cfg, "exclude_first": 1})["multiexp"]["selected"]
    assert a["resid"][0] < 0 and abs(a["resid"][0]) > 1e5                               # the outlier is real
    assert b["mse"] < a["mse"] / 100
    assert b["components"][0]["T2"] == pytest.approx(TRUTH["cpmg"]["T2_s"][0], rel=0.05) and abs(a["components"][0]["T2"] - b["components"][0]["T2"]) > 0.05 * b["components"][0]["T2"]


def test_ft_mode_spatial_fits_and_profile_ignore_the_excluded_echoes_but_the_map_keeps_them():
    load(CPMG_NOISY)
    res = process({"echo": "ft", "exclude_first": 2, "ft": {"nucleus": "1H", "G": 23.7, "zero_fill": 4, "z_min": -100.0, "z_max": 100.0},
                   "multiexp": CPMG["multiexp"], "ilt": CPMG["ilt"]})
    n = len(res["decay"]["t"])
    assert len(res["ft"]["heat"]["mag"][0]) == n                                          # the echo map still shows every echo
    sp = res["spatial"]
    assert sp["multiexp"] and any(v is not None for v in sp["multiexp"]["T2"][0])
    assert len(sp["ilt"]["spectra"]) == res["ft"]["n_ilt_positions"]
    assert len(res["multiexp"]["selected"]["fit"]) == n - 2
    assert _STATE["processed"]["positions"]["me"][1].shape[1] == n - 2                    # per-position decays are cut too


def test_t1_excludes_leading_delays():
    load(T1_CLEAN)
    base = process(T1)
    res = process({**T1, "exclude_first": 2})
    n = len(res["decay"]["t"])
    assert res["decay"]["skip"] == 2 and len(res["multiexp"]["selected"]["fit"]) == n - 2
    assert base["multiexp"]["selected"]["dof"] - 2 == res["multiexp"]["selected"]["dof"]
    with pytest.raises(ValueError, match="at least 8"):
        process({**T1, "exclude_first": n - 5})


def test_mcmc_and_bootstrap_use_the_included_points_only():
    pytest.importorskip("emcee")
    load(CPMG_NOISY)
    res = process({**CPMG, "exclude_first": 4, "ilt": {"enabled": False}})
    n = len(res["decay"]["t"])
    mcmc_start({"steps": 400, "burn": 100, "nwalkers": 16})
    assert len(_STATE["mcmc"].t) == len(_STATE["mcmc"].y) == n - 4
    mcmc_step(400)
    r = mcmc_finish()
    assert r["predictive"]["t"][0] == 0 and r["steps_run"] == 400
    boot_start({"replicates": 15, "do_ilt": False})
    assert len(_STATE["boot"].t) == n - 4
    boot_step(15)
    assert boot_finish()["replicates"] == 15


def test_params_file_carries_the_setting_in_both_directions():
    assert P.defaults()["processing"]["exclude_first"] == 0
    with pytest.raises(ValueError, match="exclude_first"):
        P.validate({"processing": {"exclude_first": -2}})
    p = P.defaults()
    p["processing"]["exclude_first"] = 3
    t = 2e-4 * np.arange(1, 129)
    s, _ = P.to_settings(p, t, "1H")
    assert s["exclude_first"] == 3 and P.to_settings_t1(p)[0]["exclude_first"] == 3
    assert P.from_settings(s)["processing"]["exclude_first"] == 3
    assert "exclude_first = 3" in P.dump_toml(p)


def test_script_outputs_show_what_was_excluded(tmp_path):
    d = tmp_path / "ds"
    shutil.copytree(CPMG_NOISY, d)
    p = P.defaults()
    p["processing"]["exclude_first"] = 2
    p["multiexp"]["compare"] = False
    p["ilt"]["max_grid_points"] = 40
    row = runner.run_dataset(d, p)
    assert row["status"] == "ok", row.get("reason")
    out = Path(row["output_dir"])
    rows = list(csv.DictReader(open(out / "decay.csv")))
    assert len(rows) == 256 and [r["included_in_fits"] for r in rows[:3]] == ["0", "0", "1"]
    assert rows[0]["exp2_fit"] == "" and rows[2]["exp2_fit"] != ""                        # no fit or residual for excluded points
    assert "first 2 echoes are excluded" in (out / "summary.md").read_text()
    assert json.loads((out / "provenance.json").read_text())["settings"]["exclude_first"] == 2
    assert (out / "decay_fit.png").read_bytes()[:4] == b"\x89PNG"
    rows2, b = runner.run_batch(d.parent, p, quiet=True)
    assert list(csv.DictReader(open(b / "batch_summary.csv")))[0]["excluded_first"] == "2"
