"""The few-population T2-D model through the params file, the script, the saved files and a batch."""
import csv
import os
import shutil
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("emcee")

from nmrdash import params as P, runner  # noqa: E402

from sample_data import T2D_CLEAN, T2D_NOISY, t2d_with_a_slow_population


def params(**over):
    p = P.defaults()
    p["t2d"].update({"t2_steps": 20, "d_steps": 10})
    p["output"].update({"figure_dpi": 50, "samples": True})
    for sec, vals in over.items():
        p[sec].update(vals)
    return P.validate(p)


def copy(src, dst):
    shutil.copytree(src, dst)
    for old in dst.glob("processed_*"):
        shutil.rmtree(old)
    return dst


def table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_the_params_file_has_a_t2d_model_section_that_round_trips_and_is_checked():
    p = P.defaults()
    assert p["t2d_model"] == {"enabled": True, "populations": "auto", "d_window": "resolvable", "seed": 0}
    p["t2d_model"].update({"populations": 3, "d_window": "map"})
    assert P.validate(tomllib.loads(P.dump_toml(p))) == p
    settings, options = P.to_settings_t2d(p)
    assert settings["t2d_model"] == {"enabled": True, "k": 3, "d_window": "map", "seed": 0}
    assert P.from_settings(settings, options)["t2d_model"]["populations"] == 3
    for bad, msg in (({"t2d_model": {"populations": 7}}, "t2d_model.populations"), ({"t2d_model": {"d_window": "all"}}, "t2d_model.d_window"), ({"t2d_model": {"enabled": "yes"}}, "true or false")):
        with pytest.raises(ValueError, match=msg):
            P.validate(bad)


def test_the_script_writes_the_model_mcmc_and_bootstrap_results(tmp_path):
    d = t2d_with_a_slow_population(tmp_path / "d")
    p = params(mcmc={"enabled": True, "walkers": 24, "steps": 800, "burn": 300}, bootstrap={"enabled": True, "replicates": 20})
    row = runner.run_dataset(d, p)
    assert row["status"] == "ok", row
    out = Path(row["output_dir"])
    names = {f.name for f in out.iterdir()}
    assert {"t2d_model_populations.csv", "t2d_model_comparison.csv", "t2d_model_statistics.csv", "mcmc_summary.csv", "mcmc_diagnostics.csv", "mcmc_samples.csv",
            "bootstrap_summary.csv", "bootstrap_samples.csv", "t2d_populations.png", "model_comparison.png", "mcmc_corner.png", "mcmc_trace.png", "summary.md"} <= names
    pops = table(out / "t2d_model_populations.csv")
    assert len(pops) == row["result"]["t2d"]["model"]["K"] == 3 and all(float(r["T2_s"]) > 0 and float(r["D_m2s"]) > 0 for r in pops)
    assert any(r["D_at_limit"] == "low" for r in pops)
    assert [r["parameter"] for r in table(out / "mcmc_summary.csv")][:3] == ["A1", "T2,1", "D1"]
    md = (out / "summary.md").read_text()
    assert "## Few-population model" in md and "## MCMC" in md and "## Bootstrap" in md and "a bound only" in md
    assert "t2d_model" in (out / "params_used.toml").read_text()


def test_mcmc_for_t2d_without_the_model_is_skipped_with_a_reason(tmp_path):
    d = copy(T2D_NOISY, tmp_path / "d")
    row = runner.run_dataset(d, params(t2d_model={"enabled": False}, mcmc={"enabled": True, "steps": 300, "burn": 100}))
    assert row["status"] == "ok" and any("few-population model" in w for w in row["warnings"])
    names = {f.name for f in Path(row["output_dir"]).iterdir()}
    assert "mcmc_summary.csv" not in names and "t2d_model_populations.csv" not in names and "t2d_spectrum.csv" in names


def test_a_batch_tabulates_each_population_over_time(tmp_path):
    root = tmp_path / "series"
    for i, src in enumerate((T2D_CLEAN, T2D_CLEAN, T2D_NOISY)):
        d = copy(src, root / f"run{i}")
        for f in ("data2D_Cp.2d", "data2D.dat", "data.dat"):
            if (d / f).exists():
                os.utime(d / f, (1_700_000_000 + 600 * i, 1_700_000_000 + 600 * i))
    rows, bdir = runner.run_batch(root, params(), quiet=True)
    assert [r["status"] for r in rows] == ["ok"] * 3
    t = table(bdir / "batch_summary.csv")
    assert t[0]["t2d_model_populations"] and "t2d_pop1_T2_s" in t[0] and float(t[1]["elapsed_min"]) == pytest.approx(10)
    assert all(float(r["t2d_pop1_fraction"]) > 0 for r in t)
    assert (bdir / "timeseries_T2D.png").is_file()


def test_the_new_figure_has_per_figure_settings(tmp_path):
    d = copy(T2D_CLEAN, tmp_path / "d")
    p = params(output={"figure_format": "svg"}, style={"time_unit": "ms"})
    p["figures"] = {"t2d_populations": {"title": "Populations of my sample"}}
    row = runner.run_dataset(d, P.validate(p))
    svg = (Path(row["output_dir"]) / "t2d_populations.svg").read_text()
    assert "Populations of my sample" in svg


def test_the_populations_figure_tells_the_two_sampling_methods_apart(tmp_path):
    """Bootstrap and MCMC clouds use different markers, and the legend shows one neutral symbol for each."""
    from nmrdash import figures
    from nmrdash.pipeline import boot_finish, boot_start, boot_step, load, mcmc_finish, mcmc_start, mcmc_step, process, _STATE
    load(T2D_CLEAN)
    res = process({"kind": "t2d", "t2d": {"t2_steps": 20, "d_steps": 10}, "t2d_model": {"enabled": True}})
    mcmc_start({"steps": 300, "burn": 100, "nwalkers": 24}); mcmc_step(300); mc = mcmc_finish({"fg": "#222", "grid": "#aaa"}, 0)
    boot_start({"replicates": 20}); boot_step(20); bs = boot_finish()
    fig = figures.t2d_populations_figure(res, mc, bs)
    leg = fig.axes[0].get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["bootstrap", "MCMC"]
    markers = [h.get_marker() for h in leg.legend_handles]
    assert markers[0] != markers[1] and markers[0] == "x"
    assert {l.get_marker() for l in fig.axes[0].lines} >= {"x", "."}
