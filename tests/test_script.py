"""The params file, the command-line runner (one dataset and batch) and provenance."""
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from nmrdash import formatting as F, params as P, provenance, runner
from nmrdash.pipeline import load

DATA = Path(__file__).resolve().parent.parent / "test_data"
FAST = {"multiexp": {"compare": False}, "ilt": {"max_grid_points": 50}}      # keeps the tests quick


def fast(**extra):
    p = P.defaults()
    for sec, vals in {**FAST, **extra}.items():
        p[sec].update(vals)
    return p


@pytest.fixture
def ds(tmp_path):
    d = tmp_path / "data" / "etgly"
    d.mkdir(parents=True)
    for f in ("acqu.par", "data.2d"):
        shutil.copy(DATA / "CPMG_EtGly" / "1" / f, d / f)
    return d


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── params ───────────────────────────────────────────────────────────────────

def test_default_toml_round_trips_and_validates():
    text = P.dump_toml()
    assert tomllib.loads(text) == P.defaults() and P.validate(tomllib.loads(text)) == P.defaults()
    assert "nmrdash run params.toml" in text and "# Number of components (1-4)." in text      # commented for the user


def test_every_value_is_checked_with_a_helpful_message():
    for bad, needle in [({"mulitexp": {}}, "unknown section"), ({"multiexp": {"nn": 2}}, "unknown setting multiexp.nn"),
                        ({"multiexp": {"n": 7}}, "multiexp.n"), ({"ilt": {"alpha": -1}}, "ilt.alpha"), ({"processing": {"echo": "fft"}}, "processing.echo"),
                        ({"ft": {"z_min": 100, "z_max": -100}}, "z_min must be below"), ({"ft": {"z_min": "auto"}}, "both"),
                        ({"mcmc": {"enabled": True, "steps": 500, "burn": 900}}, "burn"), ({"output": {"csv": "yes"}}, "true or false"),
                        ({"ilt": {"sweep_lo": 9, "sweep_hi": 4}}, "sweep_lo")]:
        with pytest.raises(ValueError, match=needle):
            P.validate(bad)


def test_to_settings_matches_what_the_page_sends():
    t = 2e-4 * np.arange(1, 129)
    s, o = P.to_settings(P.defaults(), t, "1H")
    assert s["echo"] == "integrate" and "ft" not in s
    assert s["multiexp"] == {"enabled": True, "n": 2, "offset": True, "ci_level": 0.95, "compare": True}
    assert (s["ilt"]["t2_min"], s["ilt"]["t2_max"]) == (1e-4, 1.0) and s["ilt"]["n_steps"] == 100 and s["ilt"]["alpha"] == 1e8
    s, _ = P.to_settings({"processing": {"echo": "ft"}, "ft": {"nucleus": "auto"}}, t, "19F")
    assert s["ft"]["nucleus"] == "19F" and (s["ft"]["z_min"], s["ft"]["z_max"]) == (-200.0, 200.0) and s["ft"]["spacing_um"] is None
    assert o["mcmc"]["enabled"] is False and o["bootstrap"]["method"] is None


def test_page_settings_convert_back_to_equivalent_params():
    t = 2e-4 * np.arange(1, 129)
    p = fast(processing={"echo": "ft"}, mcmc={"enabled": True, "steps": 3000, "burn": 500}, uncertainty={"ci_level": 0.99})
    s, o = P.to_settings(p, t, "1H")
    back = P.from_settings(s, o)
    s2, o2 = P.to_settings(back, t, "1H")
    assert s2 == s and o2 == o
    assert tomllib.loads(P.dump_toml(back)) == back


def test_formatting_matches_the_page_rules():
    assert F.pair(0.0198, 4.87e-4, sci=True) == "1.980e-2 ± 0.049e-2"          # one shared exponent
    assert F.interval(0.0199, 0.0131, 0.0267, 0.00347, sci=True) == "1.99e-2 [1.31e-2, 2.67e-2]"
    assert F.fmt(1310) == "1310" and F.fmt(0.0169) == "0.0169" and F.fmt(12345.6) == "1.23e+4"
    assert F.pair(4.1e-2, 1.8e3, sci=True).endswith("(uncertainty undetermined)")
    g = F.sci_group(9.996e-3, 3e-4)
    assert g(9.996e-3) == "1.000e-2"                                              # the 9.99.. -> 10.00 carry


# ── one dataset ──────────────────────────────────────────────────────────────

def test_run_dataset_writes_results_and_a_checkable_provenance(ds):
    before = sorted(p.name for p in ds.iterdir())
    row = runner.run_dataset(ds, fast(), command="nmrdash run test")
    assert row["status"] == "ok", row.get("reason")
    out = Path(row["output_dir"])
    assert out.parent == ds and out.name.startswith("processed_")
    assert sorted(p.name for p in ds.iterdir() if p != out) == before                     # raw files untouched, only a new folder
    names = {p.name for p in out.iterdir()}
    assert {"decay.csv", "multiexp_components.csv", "multiexp_fit_quality.csv", "ilt_spectrum.csv", "ilt_peaks.csv", "decay_fit.png", "ilt_spectrum.png",
            "params_used.toml", "provenance.json", "summary.md"} <= names
    assert len(read_csv(out / "decay.csv")) == 128 and (out / "decay_fit.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    prov = json.loads((out / "provenance.json").read_text())
    by = {f["name"]: f for f in prov["inputs"]}
    assert by["data.2d"]["sha256"] == hashlib.sha256((ds / "data.2d").read_bytes()).hexdigest() and by["acqu.par"]["bytes"] == (ds / "acqu.par").stat().st_size
    assert prov["command"] == "nmrdash run test" and prov["settings"]["multiexp"]["n"] == 2 and prov["acquisition"]["echoTime"] == 200
    assert prov["nmrdash"]["version"] and "dirty" in prov["nmrdash"] and prov["environment"]["packages"]["numpy"]
    assert set(prov["outputs"]) >= {"decay.csv", "provenance.json", "summary.md"}
    used = P.load(out / "params_used.toml")                                                # the settings used can be run again
    assert used["multiexp"]["n"] == 2 and used["ilt"]["max_grid_points"] == 50
    md = (out / "summary.md").read_text()
    assert "## Multi-exponential fit" in md and "## Provenance" in md and "sha256" in md


def test_two_runs_never_overwrite_each_other(ds):
    a = runner.run_dataset(ds, fast())["output_dir"]
    b = runner.run_dataset(ds, fast())["output_dir"]
    assert a != b and Path(a).is_dir() and Path(b).is_dir()


def test_output_directory_redirects_everything_away_from_the_data(ds, tmp_path):
    before = sorted(p.name for p in ds.iterdir())
    row = runner.run_dataset(ds, fast(output={"directory": str(tmp_path / "results")}))
    assert Path(row["output_dir"]).is_relative_to(tmp_path / "results") and sorted(p.name for p in ds.iterdir()) == before


def test_csv_and_figures_can_be_switched_off(ds):
    out = Path(runner.run_dataset(ds, fast(output={"csv": False, "figures": False}))["output_dir"])
    assert {p.name for p in out.iterdir()} == {"params_used.toml", "provenance.json", "summary.md"}


def test_ft_mode_writes_the_spatial_files(ds):
    row = runner.run_dataset(ds, fast(processing={"echo": "ft"}, ft={"z_min": -100.0, "z_max": 100.0}, multiexp={"n": 1, "compare": False}))
    assert row["status"] == "ok", row.get("reason")
    out = Path(row["output_dir"])
    assert {"ft_profile.csv", "spatial_multiexp.csv", "spatial_ilt_map.csv", "spatial_ilt_peaks.csv", "spatial.png"} <= {p.name for p in out.iterdir()}
    rows = read_csv(out / "spatial_multiexp.csv")
    assert len(rows) > 3 and "T2_1_s" in rows[0] and "reduced_chi2" in rows[0]
    assert "Spatial analysis (FT)" in (out / "summary.md").read_text()


def test_mcmc_and_bootstrap_outputs_and_seeds_in_provenance(ds):
    pytest.importorskip("emcee")
    p = fast(multiexp={"n": 1, "compare": False}, ilt={"enabled": False}, mcmc={"enabled": True, "steps": 400, "burn": 100, "walkers": 16, "seed": 5},
             bootstrap={"enabled": True, "replicates": 25, "ilt": False, "seed": 9})
    row = runner.run_dataset(ds, p)
    assert row["status"] == "ok", row.get("reason")
    out = Path(row["output_dir"])
    assert {"mcmc_summary.csv", "mcmc_diagnostics.csv", "mcmc_samples.csv", "mcmc_corner.png", "bootstrap_summary.csv", "bootstrap_samples.csv", "bootstrap_distributions.png"} <= {x.name for x in out.iterdir()}
    assert json.loads((out / "provenance.json").read_text())["random_seeds"] == {"mcmc": 5, "bootstrap": 9}
    assert len(read_csv(out / "mcmc_samples.csv")) > 100


def mystery(d):
    """A folder whose sequence nmrdash does not analyse."""
    d.mkdir(parents=True)
    shutil.copy(DATA / "T1SAT" / "1" / "acqu.par", d / "acqu.par")
    (d / "acqu.par").write_text((d / "acqu.par").read_text().replace('experiment = "T1Sat"', 'experiment = "Mystery"'))
    shutil.copy(DATA / "T1SAT" / "1" / "data.1d", d / "data.1d")


def test_unsupported_sequences_are_skipped_not_crashed(tmp_path):
    d = tmp_path / "mystery"
    mystery(d)
    row = runner.run_dataset(d, fast())
    assert row["status"] == "skipped" and "Mystery" in row["reason"] and row["output_dir"] is None


# ── batch ────────────────────────────────────────────────────────────────────

@pytest.fixture
def series(tmp_path):
    root = tmp_path / "series"
    base = datetime.datetime(2026, 10, 1, 9, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    for name, src, minutes in (("c_third", "CPMG_EtGly/2", 90), ("a_first", "CPMG_EtGly/1", 0), ("b_second", "CPMG_EtGly/1", 30)):
        d = root / "samples" / name
        d.mkdir(parents=True)
        for f in ("acqu.par", "data.2d"):
            shutil.copy(DATA / src / f, d / f)
        os.utime(d / "data.2d", (base + 60 * minutes,) * 2)
    mystery(root / "other")                                              # a sequence nmrdash does not analyse
    return root


def test_batch_orders_by_timestamp_and_computes_elapsed_time(series):
    rows, bdir = runner.run_batch(series, fast(), quiet=True)
    table = read_csv(bdir / "batch_summary.csv")
    done = [r for r in table if r["status"] == "ok"]
    assert [r["dataset"] for r in done] == [str(Path("samples/a_first")), str(Path("samples/b_second")), str(Path("samples/c_third"))]   # by time, not by name
    assert [float(r["elapsed_min"]) for r in done] == [0.0, 30.0, 90.0]
    assert done[0]["timestamp_utc"] == "2026-10-01T09:00:00+00:00"
    skipped = [r for r in table if r["status"] == "skipped"]
    assert len(skipped) == 1 and skipped[0]["elapsed_min"] == "" and "Mystery" in skipped[0]["reason"]
    assert float(done[0]["T2_2_s"]) > float(done[0]["T2_1_s"]) > 0
    assert {"params_used.toml", "provenance.json", "batch_summary.csv"} <= {p.name for p in bdir.iterdir()}
    prov = json.loads((bdir / "provenance.json").read_text())
    assert prov["kind"] == "batch" and prov["timestamps"]["source"].startswith("modification time of the main data file") and len(prov["datasets"]) == 4
    assert all(Path(r["output_dir"]).is_dir() for r in rows if r["status"] == "ok")


def test_batch_ignores_its_own_earlier_results(series):
    runner.run_batch(series, fast(), quiet=True)
    assert len(runner.discover(series)) == 4                                   # processed_* and batch_* folders are not datasets


def test_batch_time_zero_options_and_timestamp_override(series, tmp_path):
    _, b = runner.run_batch(series, fast(batch={"t0": "2026-10-01T08:00:00+00:00"}), quiet=True)
    assert [float(r["elapsed_min"]) for r in read_csv(b / "batch_summary.csv") if r["status"] == "ok"] == [60.0, 90.0, 150.0]
    _, b = runner.run_batch(series, fast(batch={"t0": "none"}), quiet=True)
    assert all(r["elapsed_min"] == "" for r in read_csv(b / "batch_summary.csv"))
    over = tmp_path / "times.csv"
    over.write_text("folder,timestamp\nsamples/a_first,2026-10-02T00:00:00+00:00\n")      # as if files had been copied
    _, b = runner.run_batch(series, fast(batch={"timestamps_file": str(over)}), quiet=True)
    done = [r for r in read_csv(b / "batch_summary.csv") if r["status"] == "ok"]
    assert [r["dataset"] for r in done] == [str(Path("samples") / n) for n in ("b_second", "c_third", "a_first")]      # a_first now sits last
    assert [float(r["elapsed_min"]) for r in done] == [0.0, 60.0, 870.0]                                          # from b_second at 09:30 to 00:00 next day
    assert json.loads((b / "provenance.json").read_text())["timestamps"]["source"].endswith(f"overridden from {over}")


def test_parallel_batch_gives_the_same_numbers(series):
    _, b1 = runner.run_batch(series, fast(), jobs=1, quiet=True)
    _, b2 = runner.run_batch(series, fast(), jobs=2, quiet=True)
    skip = {"seconds", "output_dir"}
    a, b = read_csv(b1 / "batch_summary.csv"), read_csv(b2 / "batch_summary.csv")
    assert [{k: v for k, v in r.items() if k not in skip} for r in a] == [{k: v for k, v in r.items() if k not in skip} for r in b]


def test_batch_time_series_figure_when_there_are_several_datasets(series):
    _, b = runner.run_batch(series, fast(), quiet=True)
    assert (b / "timeseries.png").read_bytes()[:4] == b"\x89PNG"


# ── command line ─────────────────────────────────────────────────────────────

def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "nmrdash", *args], capture_output=True, text=True, cwd=cwd)


def test_cli_params_run_and_batch(ds, series, tmp_path):
    r = cli("params")
    assert r.returncode == 0 and tomllib.loads(r.stdout) == P.defaults()
    f = tmp_path / "p.toml"
    f.write_text(P.dump_toml(fast()))
    r = cli("run", str(f), str(ds))
    assert r.returncode == 0 and "done in" in r.stdout, r.stderr
    out = next(ds.glob("processed_*"))
    assert json.loads((out / "provenance.json").read_text())["command"].startswith("nmrdash run")
    r = cli("batch", str(f), str(series), "--list")
    assert r.returncode == 0 and r.stdout.count("\n") == 4 and "a_first" in r.stdout
    r = cli("batch", str(f), str(series), "--jobs", "2")
    assert r.returncode == 0 and "3 ok, 1 skipped, 0 failed" in r.stdout, r.stdout + r.stderr


def test_cli_reports_a_bad_params_file_without_a_traceback(tmp_path, ds):
    f = tmp_path / "bad.toml"
    f.write_text("[multiexp]\nn = 9\n")
    r = cli("run", str(f), str(ds))
    assert r.returncode == 2 and "multiexp.n" in r.stderr and "Traceback" not in r.stderr
    assert cli("run", str(tmp_path / "missing.toml"), str(ds)).returncode == 2
    assert cli("run", "defaults", str(tmp_path)).returncode == 1                          # a folder with no dataset


# ── provenance record itself ─────────────────────────────────────────────────

def test_provenance_build_for_a_page_style_run():
    rec = provenance.build(DATA / "CPMG_EtGly" / "1", {"echo": "integrate"}, {"mcmc": {"enabled": True, "seed": 3}}, dataset="sample: x",
                           acquisition={"echoTime": 200, "junk": 1}, modified_ms=1760000000000, extra={"userAgent": "UA"})
    assert rec["acquisition"] == {"echoTime": 200} and rec["random_seeds"] == {"mcmc": 3} and rec["browser"] == {"userAgent": "UA"}
    d2d = next(f for f in rec["inputs"] if f["name"] == "data.2d")
    assert d2d["modified_utc"].startswith("2025-10-09")                                   # the original write time, not the copy's
    assert any("sha256" in line for line in provenance.describe(rec))


# ── what the web page asks for ───────────────────────────────────────────────

def test_page_provenance_and_export_round_trip():
    from nmrdash import pipeline
    load(DATA / "CPMG_EtGly" / "1")
    settings, options = P.to_settings(fast(mcmc={"enabled": True, "steps": 800, "burn": 200}), 2e-4 * np.arange(1, 129), "1H")
    with pytest.raises(RuntimeError, match="Run the processing first"):
        pipeline.get_provenance()
    pipeline.process(settings)
    out = pipeline.get_provenance({"dataset": "sample: CPMG_EtGly/1", "browser": {"userAgent": "UA"}, "uncertainty": {"mcmc": {"steps_run": 800}}}, options)
    rec = out["record"]
    json.dumps(out, allow_nan=False)
    assert rec["dataset"] == "sample: CPMG_EtGly/1" and rec["browser"]["userAgent"] == "UA" and rec["browser"]["uncertainty_runs"]["mcmc"]["steps_run"] == 800
    assert rec["inputs"][1]["sha256"] == hashlib.sha256((DATA / "CPMG_EtGly" / "1" / "data.2d").read_bytes()).hexdigest()
    assert rec["random_seeds"] == {"mcmc": 0} and any("never left" in n for n in rec["notes"]) and out["lines"]
    # the params the page would export reproduce the same settings
    text = pipeline.export_params(settings, options, "sample")
    again, opt2 = P.to_settings(tomllib.loads(text), 2e-4 * np.arange(1, 129), "1H")
    assert again == settings and opt2 == options and "Exported from the nmrdash web dashboard" in text
    assert tomllib.loads(rec["params_file"]["text"]) == tomllib.loads(P.dump_toml(P.from_settings(settings, options)))


def test_exported_script_runs(tmp_path, ds):
    from nmrdash import pipeline
    script = tmp_path / "process_nmr.py"
    script.write_text(pipeline.export_script())
    compile(script.read_text(), "process_nmr.py", "exec")
    (tmp_path / "params.toml").write_text(P.dump_toml(fast()))
    r = subprocess.run([sys.executable, str(script), str(ds)], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr
    out = next(ds.glob("processed_*"))
    assert json.loads((out / "provenance.json").read_text())["command"].startswith("python ")
    r = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--batch" in r.stdout


def test_style_and_figure_overrides_round_trip_and_validate():
    import tomllib
    p = P.defaults()
    p["style"].update({"font_size": 14, "time_unit": "ms", "width": 6.0})
    p["figures"] = {"decay_fit": {"title": "Sample A", "xmin": 0.002, "yscale": "log"}}
    p = P.validate(p)
    assert P.validate(tomllib.loads(P.dump_toml(p))) == p
    assert P.validate(tomllib.loads(P.dump_toml(P.defaults()))) == P.defaults()
    st = P.style_dict(p)
    assert st["width"] == 6.0 and st["height"] is None and st["figures"]["decay_fit"]["xlim"] == [0.002, None] and st["figures"]["decay_fit"]["yscale"] == "log"
    assert P.style_dict(P.defaults(), titles=False)["title"] == "none"
    for bad, msg in (({"style": {"font_size": 2}}, "style.font_size"), ({"style": {"time_unit": "min"}}, "style.time_unit"),
                     ({"figures": {"nope": {}}}, "unknown figure"), ({"figures": {"decay_fit": {"xmin": "a"}}}, "figures.decay_fit.xmin"),
                     ({"figures": {"decay_fit": {"zmin": 1}}}, "unknown setting")):
        with pytest.raises(ValueError, match=msg):
            P.validate(bad)


def test_the_script_applies_the_style_to_its_figures(tmp_path):
    import shutil
    d = tmp_path / "d"
    shutil.copytree(DATA / "CPMG_EtGly" / "1", d)
    p = P.defaults()
    p["multiexp"].update({"enabled": True, "n": 1})
    p["output"].update({"figure_format": "svg", "samples": False})
    p["style"].update({"time_unit": "ms", "title": "custom", "title_text": "From the params file"})
    row = runner.run_dataset(d, p)
    assert row["status"] == "ok", row
    svg = (Path(row["output_dir"]) / "decay_fit.svg").read_text()
    assert "(ms)" in svg and "From the params file" in svg

def test_params_for_page_fills_settings_for_the_loaded_dataset():
    from nmrdash.pipeline import load, params_for_page
    load(DATA / "CPMG_EtGly" / "1")
    p = P.defaults()
    p["processing"]["exclude_first"] = 2
    p["multiexp"].update({"enabled": True, "n": 3}); p["style"]["font_size"] = 13
    p["figures"] = {"ilt_spectrum": {"title": "T", "xmax": 5.0}}
    out = params_for_page(P.dump_toml(p), "my.toml")
    assert out["kind"] == "cpmg" and out["settings"]["exclude_first"] == 2 and out["settings"]["multiexp"]["n"] == 3
    assert out["settings"]["ilt"]["t2_min"] > 0 and out["style"]["font_size"] == 13 and out["style"]["figures"]["ilt_spectrum"]["xlim"] == [None, 5.0]
    json.dumps(out, allow_nan=False)
    load(DATA / "T1SAT" / "1")
    assert params_for_page(P.dump_toml(p), "my.toml")["kind"] == "t1"
    with pytest.raises(ValueError, match="my.toml"):
        params_for_page("this is = not [toml", "my.toml")
    with pytest.raises(ValueError, match="multiexp.n"):
        params_for_page("[multiexp]\nn = 9\n", "x.toml")
