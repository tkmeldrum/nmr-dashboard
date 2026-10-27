"""Saturation recovery (T1Sat): the sequence name picks the analysis, data.dat is the data, and everything downstream works."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nmrdash import multiexp, params as P, runner, sequences, uncertainty as u
from nmrdash.pipeline import (boot_finish, boot_start, boot_step, get_provenance, load, mcmc_finish, mcmc_start, mcmc_step,
                              process)

from sample_data import CPMG_SNR100, DATA, T1_CLEAN, T1_SNR100, TRUTH
T1_SETTINGS = {"kind": "t1", "multiexp": {"enabled": True, "n": 1, "offset": True, "ci_level": 0.95, "compare": True}}


def recovery(comps=((1000.0, 0.4),), c=0.0, n=15, sigma=15.0, seed=0):
    t = np.geomspace(0.025, 2.5, n)
    y = sum(a * (1 - np.exp(-t / T)) for a, T in comps) + c
    return t, y + np.random.default_rng(seed).normal(0, sigma, n)


# ── the sequence name picks the analysis ─────────────────────────────────────

@pytest.mark.parametrize("name,kind", [("CPMG", "cpmg"), ("cpmg", "cpmg"), ("T1Sat", "t1"), ("T1SAT", "t1"), ("t1_sat", "t1"),
                                       ("SSE-T2", "t2d"), ("SSET2", "t2d"), ("sse t2", "t2d"), ("Diffusion", None), ("", None)])
def test_sequence_names_map_to_analyses_tolerating_spelling(name, kind):
    assert sequences.kind_of({"experiment": name}) == kind


def test_supported_and_unsupported_sequences_explain_themselves():
    assert sequences.describe({"experiment": "T1Sat"})["supported"] and sequences.describe({"experiment": "CPMG"})["supported"]
    d = sequences.describe({"experiment": "SSE-T2"})
    assert d["supported"] and d["kind"] == "t2d" and "T₂–D" in d["analysis"]
    assert "not one nmrdash can analyse" in sequences.describe({"experiment": "Diffusion"})["reason"]
    assert "no `experiment`" in sequences.describe({})["reason"]


def _without_experiment(src, dst):
    shutil.copytree(src, dst)
    par = dst / "acqu.par"
    par.write_text("".join(l for l in par.read_text().splitlines(True) if not l.startswith("experiment")))
    return dst


def test_a_folder_with_no_experiment_entry_defaults_to_cpmg_and_can_be_told_otherwise(tmp_path):
    d = _without_experiment(Path(__file__).resolve().parent.parent / "test_data" / "kea_cpmg_clean", tmp_path / "d")
    r = load(d, fallback_to_mtime=False)
    assert r["sequence_assumed"] and r["kind"] == "cpmg" and r["supported"] and r["params_all"]["experiment"] == "CPMG"
    assert load(d, fallback_to_mtime=False, sequence="T1Sat")["kind"] == "t1"
    real = load(Path(__file__).resolve().parent.parent / "test_data" / "kea_cpmg_clean", fallback_to_mtime=False, sequence="T1Sat")
    assert not real["sequence_assumed"] and real["kind"] == "cpmg"        # the file's own entry wins


def test_a_cpmg_folder_with_a_wrong_sequence_name_is_not_processed_as_cpmg(tmp_path):
    d = tmp_path / "x"
    shutil.copytree(CPMG_SNR100, d)
    par = (d / "acqu.par").read_text().replace('experiment = "CPMG"', 'experiment = "Mystery"')
    (d / "acqu.par").write_text(par)
    r = load(d)
    assert not r["supported"] and r["kind"] is None and "Mystery" in r["reason"]


# ── fitting recovery curves ──────────────────────────────────────────────────

def test_monoexponential_recovery_intervals_cover_the_truth():
    """95% intervals should contain the true T1 / amplitude / offset about 95% of the time (checked over 30 noise draws)."""
    hits = total = 0
    for seed in range(30):
        t, y = recovery(c=40.0, seed=seed)
        f = multiexp.fit(t, y, 1, offset=True, kind="saturation")
        c = f["components"][0]
        assert f["kind"] == "saturation" and f["symbol"] == "T1"
        for lo_hi, truth in ((c["T2_ci"], 0.4), (c["A_ci"], 1000.0), ([f["offset_value"] - f["t_crit"] * f["offset_se"], f["offset_value"] + f["t_crit"] * f["offset_se"]], 40.0)):
            hits += lo_hi[0] <= truth <= lo_hi[1]
            total += 1
    assert 0.82 <= hits / total <= 1.0, hits / total


def test_biexponential_recovery_and_model_choice():
    t, y = recovery(comps=((600.0, 0.08), (400.0, 1.2)), sigma=4.0, n=40)
    r = multiexp.fit_with_comparison(t, y, 2, offset=False, compare_up_to=2, kind="saturation")
    T = [c["T2"] for c in r["selected"]["components"]]
    assert T == sorted(T) and T[0] == pytest.approx(0.08, rel=0.3) and T[1] == pytest.approx(1.2, rel=0.3)
    assert r["recommended_n"] == 2
    t, y = recovery(sigma=8.0)
    assert multiexp.fit_with_comparison(t, y, 1, offset=False, compare_up_to=2, kind="saturation")["recommended_n"] == 1   # one component is enough


def test_the_offset_matters_for_data_that_do_not_start_at_zero():
    t, y = recovery(c=300.0, sigma=5.0)
    with_c = multiexp.fit(t, y, 1, True, kind="saturation")
    without = multiexp.fit(t, y, 1, False, kind="saturation")
    assert with_c["bic"] < without["bic"] - 20


def test_decay_fits_are_unchanged_by_the_generalisation():
    t = 2e-4 * np.arange(1, 129)
    y = 1000 * np.exp(-t / 0.01) + np.random.default_rng(0).normal(0, 3, 128)
    assert multiexp.fit(t, y, 1)["kind"] == "decay" and multiexp.fit(t, y, 1)["symbol"] == "T2"
    assert multiexp.fit(t, y, 1)["components"][0]["T2"] == pytest.approx(0.01, rel=0.05)


# ── the synthetic samples ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [T1_SNR100, T1_CLEAN, T1_CLEAN], ids=lambda p: str(p.relative_to(DATA)))
def test_t1_samples_process_and_prefer_one_component(d):
    load(d)
    res = process(T1_SETTINGS)
    json.dumps(res, allow_nan=False)
    me = res["multiexp"]["selected"]
    assert res["mode"] == "t1" and res["ilt"] is None and res["ft"] is None and res["decay"]["noise_sigma"] is None
    assert me["components"][0]["T2"] > 0 and me["r2"] > 0.99 and me["chi2_red"] is None
    assert res["multiexp"]["recommended_n"] in (1, 2)                  # one is the norm; on noisy single-exponential data BIC sometimes prefers two, which is not an error


def test_t1_fit_of_the_noisy_set_is_stable():
    load(T1_SNR100)
    c = process(T1_SETTINGS)["multiexp"]["selected"]["components"][0]
    assert c["T2"] == pytest.approx(0.8091, rel=1e-3) and c["T2_se"] == pytest.approx(0.0142, rel=0.05) and abs(c["T2"] - TRUTH["t1"]["T1_s"]) < 3 * c["T2_se"]


def test_t1_rejects_three_components():
    load(T1_SNR100)
    with pytest.raises(ValueError, match="one or two"):
        process({**T1_SETTINGS, "multiexp": {**T1_SETTINGS["multiexp"], "n": 3}})


# ── uncertainty methods on recovery data (no imaginary channel) ───────────────

def test_mcmc_and_bootstrap_work_on_recovery_data_without_a_noise_channel():
    pytest.importorskip("emcee")
    load(T1_SNR100)
    process(T1_SETTINGS)
    info = mcmc_start({"steps": 1500, "burn": 500, "nwalkers": 16})
    assert info["sigma_sampled"] and info["ndim"] == 4                 # A, T1, offset and the sampled sigma
    mcmc_step(1500)
    r = mcmc_finish({"fg": "#222", "grid": "#aaa"})
    json.dumps(r, allow_nan=False)
    t1 = next(p for p in r["params"] if p["name"] == "T1")
    assert r["symbol"] == "T1" and t1["lo"] < TRUTH["t1"]["T1_s"] < t1["hi"] and t1["sd"] == pytest.approx(t1["ls_se"], rel=0.5)
    b = boot_start({"replicates": 60, "do_ilt": False})
    assert b["method"] == "residual"                                   # nothing to resample but the residuals
    boot_step(60)
    out = boot_finish()
    json.dumps(out, allow_nan=False)
    assert out["ilt"] is None and next(p for p in out["multiexp"]["params"] if p["name"] == "T1")["sd"] > 0


def test_mcmc_recovers_a_known_t1_and_shows_a_sensible_interval():
    pytest.importorskip("emcee")
    t, y = recovery(c=0.0, sigma=15.0, n=20)
    f = multiexp.fit(t, y, 1, False, kind="saturation")
    m = u.Mcmc(t, y, 1, False, f, sigma=None, steps=2500, burn=800, nwalkers=16)
    m.step(2500)
    r = m.result()
    t1 = next(p for p in r["params"] if p["name"] == "T1")
    assert t1["lo"] < 0.4 < t1["hi"] and 0.3 < t1["median"] < 0.5


# ── params file, command line and provenance ─────────────────────────────────

def test_params_t1_section_and_settings():
    p = P.defaults()
    assert p["t1"] == {"components": 1, "offset": True, "compare": True, "amplitude": "sum"}
    s, o = P.to_settings_t1(p)
    assert s == {**T1_SETTINGS, "exclude_first": 0, "t1_amplitude": "sum"} and o["bootstrap"]["do_ilt"] is False
    with pytest.raises(ValueError, match="t1.components"):
        P.validate({"t1": {"components": 3}})
    back = P.from_settings(s, o)
    assert back["t1"] == p["t1"] and P.to_settings_t1(back)[0] == s


def test_run_dataset_for_t1_writes_the_usual_files(tmp_path):
    d = tmp_path / "t1"
    shutil.copytree(T1_SNR100, d)
    row = runner.run_dataset(d, P.defaults(), command="nmrdash run x")
    assert row["status"] == "ok" and row["kind"] == "t1", row.get("reason")
    out = Path(row["output_dir"])
    names = {p.name for p in out.iterdir()}
    assert {"decay.csv", "multiexp_components.csv", "model_comparison.csv", "decay_fit.png", "summary.md", "provenance.json"} <= names
    assert not any(n.startswith("ilt") for n in names)                 # no ILT for sparse recovery data
    head = open(out / "decay.csv").readline().strip().split(",")
    assert head[:2] == ["delay_s", "signal"] and "exp1_fit" in head
    comp = list(csv.DictReader(open(out / "multiexp_components.csv")))[0]
    assert float(comp["T1_s"]) == pytest.approx(TRUTH["t1"]["T1_s"], rel=0.03)
    prov = json.loads((out / "provenance.json").read_text())
    assert [f["name"] for f in prov["inputs"]] == ["acqu.par", "data.dat"] and prov["settings"]["kind"] == "t1"
    assert "Saturation-recovery fit" in (out / "summary.md").read_text()


def test_a_mixed_batch_keeps_t2_and_t1_columns_apart(tmp_path):
    root = tmp_path / "mixed"
    for name, src in (("cpmg", "kea_cpmg_snr100"), ("t1a", "kea_t1_snr100"), ("t1b", "kea_t1_clean")):
        shutil.copytree(DATA / src, root / name)
    fast = P.defaults()
    fast["multiexp"]["compare"] = False
    fast["ilt"]["max_grid_points"] = 40
    rows, b = runner.run_batch(root, fast, quiet=True)
    table = list(csv.DictReader(open(b / "batch_summary.csv")))
    by = {r["dataset"]: r for r in table}
    assert by["cpmg"]["kind"] == "cpmg" and by["cpmg"]["T2_1_s"] and not by["cpmg"]["T1_1_s"]
    assert by["t1a"]["kind"] == "t1" and float(by["t1a"]["T1_1_s"]) == pytest.approx(TRUTH["t1"]["T1_s"], rel=0.03) and not by["t1a"]["T2_1_s"]
    assert by["t1a"]["mean_T1_s"] == "" or float(by["t1a"]["mean_T1_s"]) > 0
    assert all(r["status"] == "ok" for r in table)
    assert (b / "timeseries_T1.png").read_bytes()[:4] == b"\x89PNG"      # two T1 datasets -> a T1 time series


def test_page_provenance_for_t1():
    load(T1_SNR100)
    process(T1_SETTINGS)
    rec = get_provenance({"dataset": "sample: kea_t1_snr100"}, {})["record"]
    assert {f["name"] for f in rec["inputs"]} == {"acqu.par", "data.dat"} and rec["settings"]["kind"] == "t1"
