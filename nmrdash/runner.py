"""Run nmrdash from Python or the command line: one dataset, or a whole folder of them (batch / time series).

Every dataset gets a processed_<timestamp> folder with CSV tables, figures, summary.md, params_used.toml and
provenance.json. A batch also writes one combined table (a row per dataset, with timestamps and elapsed time).
"""
import csv
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import outputs, params as P, pipeline, provenance, tecmag

CHUNK_STEPS, CHUNK_REPS = 500, 25
LIGHT_THEME = {"fg": "#1a1a1a", "grid": "#bbbbbb"}


DATA_FILES = ("data.2d", "data2D_Cp.2d", "data2D.dat", "data.dat")      # the main data file of CPMG, then T2-D, then T1 (the page uses the same order)


def _data_file(folder):
    """The file whose modification time says when the data were written: the first of DATA_FILES that exists."""
    return next((f for f in (Path(folder) / n for n in DATA_FILES) if f.is_file()), None)


def _resolve(path):
    """(folder, .tnt file or None) for a dataset path: a Kea folder, a Tecmag .tnt file, or a folder holding one .tnt (and no acqu.par)."""
    p = Path(path)
    if p.is_file():
        return p.parent, p
    return p, (None if (p / "acqu.par").is_file() else tecmag.find(p))


def _utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def _stamp(now=None):
    return (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")


def _unique_dir(parent, base):
    """parent/base, or base-2, base-3 ... so a re-run never overwrites earlier results."""
    d, k = Path(parent) / base, 2
    while d.exists():
        d = Path(parent) / f"{base}-{k}"
        k += 1
    d.mkdir(parents=True)
    return d


def _log(quiet, *a):
    if not quiet:
        print(*a, flush=True)


# ── one dataset ──────────────────────────────────────────────────────────────

def run_dataset(folder, params=None, *, params_text=None, params_path=None, rel_name=None, command=None, quiet=True, only_kind=None):
    """Process one data folder (acqu.par + data.2d). Returns a summary dict; never raises for bad data (status says why)."""
    t_start = time.perf_counter()
    folder, tnt = _resolve(folder)
    p = P.validate(params) if params is not None else P.defaults()
    row = {"dataset": rel_name or (tnt.name if tnt else folder.name), "folder": str(folder), "status": "failed", "reason": "", "timestamp_utc": None,
           "timestamp_local": None, "mode": p["processing"]["echo"], "output_dir": None, "warnings": []}
    try:
        data2d = tnt or _data_file(folder)
        mtime = data2d.stat().st_mtime if data2d else None
        info = pipeline.load(tnt or folder, mtime * 1000 if mtime else None, fallback_to_mtime=True)
        if info.get("source_format") == "tecmag":                 # Tecmag: the experiment date stored in the file, not the file's time
            mtime = pipeline._STATE["source"]["timestamp_ms"] / 1000 if info["supported"] and pipeline._STATE.get("source") else mtime
            row["warnings"] += info.get("warnings", [])
        if mtime:
            row["timestamp_utc"] = _utc(mtime).isoformat(timespec="seconds")
            row["timestamp_local"] = datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
        row["kind"] = info["kind"]
        if only_kind and info["kind"] != only_kind:               # a batch uses one sequence's settings; other sequences are left alone
            row.update(status="skipped", reason=f"a {info['kind']} dataset; this batch is {only_kind}")
            return row
        if not info["supported"]:
            row.update(status="skipped", reason=info["reason"])
            return row
        raw = info["raw"]
        row["kind"] = info["kind"]
        if info["kind"] == "t1":
            row.update(n_points=raw["n_points"], mode="t1")
            settings, options = P.to_settings_t1(p)
        elif info["kind"] == "t2d":
            row.update(n_echoes=raw["n_echoes"], n_points=raw["n_tau"], mode="t2d")
            settings, options = P.to_settings_t2d(p, info["defaults"]["gradient"])
        else:
            row.update(n_echoes=raw["n_echoes"], n_points=raw["n_points"], echo_time_us=info["params_all"].get("echoTime"))
            settings, options = P.to_settings(p, raw["echo_time_s"], info["defaults"]["nucleus"], info["defaults"]["G"])
        row["exclude_first"] = settings.get("exclude_first", 0)
        res = pipeline.process(settings)
        warnings = []
        mcmc = boot = None
        samples_n = 5000 if p["output"]["samples"] else 0
        mc, bs = options.get("mcmc", {"enabled": False}), options.get("bootstrap", {"enabled": False})
        if info["kind"] == "t2d" and (p["mcmc"]["enabled"] or p["bootstrap"]["enabled"]) and not p["t2d_model"]["enabled"]:
            warnings.append("MCMC and the bootstrap for T2-D sample the few-population model, which is switched off ([t2d_model] enabled = false); they were skipped.")
            mc, bs = {**mc, "enabled": False}, {**bs, "enabled": False}
        if info["kind"] in ("t1", "t2d"):
            mc = {**mc, "target": "bulk"}
        if mc["enabled"]:
            try:
                mcmc = _run_mcmc(mc, samples_n)
            except Exception as exc:                              # e.g. emcee not installed: keep everything else
                warnings.append(f"MCMC was not run: {exc}")
        if bs["enabled"]:
            try:
                boot = _run_bootstrap(bs)
            except Exception as exc:
                warnings.append(f"Bootstrap was not run: {exc}")
        for r in (res.get("multiexp") or {}).get("selected", {}).get("warnings", []):
            warnings.append(r)

        # ── write everything ──
        out_cfg = p["output"]
        base = f"processed_{tnt.stem + '_' if tnt else ''}{_stamp()}"
        parent = folder if not out_cfg["directory"] else Path(out_cfg["directory"]) / (rel_name or (tnt.name if tnt else folder.name))
        parent.mkdir(parents=True, exist_ok=True)
        out_dir = _unique_dir(parent, base)
        files = []
        if out_cfg["csv"]:
            files += outputs.write_tables(out_dir, res, mcmc, boot, samples=out_cfg["samples"])
        if out_cfg["figures"]:
            band = mcmc if (mcmc and mc["target"] == "bulk") else None
            files += outputs.write_figures(out_dir, res, mcmc, boot, out_cfg["figure_format"], out_cfg["figure_dpi"], out_cfg["titles"], band, style=P.style_dict(p, out_cfg["titles"]),
                                          raw=pipeline._STATE.get("raw_out"), kind=info["kind"])
        text = params_text if params_text is not None else P.dump_toml(p)
        (out_dir / "params_used.toml").write_text(P.dump_toml(p, header=f"Resolved and used for: {row['dataset']}\nOriginal params file: {params_path or '(defaults)'}"))
        files.append("params_used.toml")
        names = files + ["provenance.json", "summary.md"] + (["report.pdf"] if out_cfg["report"] else [])
        prov = provenance.build(folder, settings, options, params_text=text, params_path=params_path, command=command, dataset=row["dataset"],
                                acquisition=info["params_all"], notes=warnings, outputs=names,
                                **({"input_names": (tnt.name,), "modified_ms": pipeline._STATE["source"]["timestamp_ms"]} if tnt else {}))
        (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        (out_dir / "summary.md").write_text(outputs.summary_markdown(res, prov, mcmc, boot, names, warnings))
        if out_cfg["report"]:
            from . import report
            (out_dir / "report.pdf").write_bytes(report.build(res, prov, mcmc, boot, raw=pipeline._STATE.get("raw_out"), style=P.style_dict(p, out_cfg["titles"])))
        row.update(status="ok", output_dir=str(out_dir), warnings=warnings, settings=settings, result=_key_results(res, mcmc, boot))
    except Exception as exc:
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=6))
    finally:
        row["seconds"] = round(time.perf_counter() - t_start, 2)
    return row


def _run_mcmc(cfg, max_samples):
    import importlib.util
    if importlib.util.find_spec("emcee") is None:
        raise RuntimeError("the emcee package is not installed (pip install emcee)")
    pipeline.mcmc_start({"target": cfg["target"], "z_um": cfg["z_um"], "nwalkers": cfg["nwalkers"], "steps": cfg["steps"], "burn": cfg["burn"],
                         "ci_level": cfg["ci_level"], "sample_sigma": cfg["sample_sigma"], "seed": cfg["seed"]})
    done = 0
    while done < cfg["steps"]:
        done = pipeline.mcmc_step(CHUNK_STEPS)["done"]
    return pipeline.mcmc_finish(LIGHT_THEME, max_samples)


def _run_bootstrap(cfg):
    info = pipeline.boot_start({"target": cfg["target"], "z_um": cfg["z_um"], "replicates": cfg["replicates"], "method": cfg["method"],
                                "do_multiexp": cfg["do_multiexp"], "do_ilt": cfg["do_ilt"], "ci_level": cfg["ci_level"], "seed": cfg["seed"]})
    done = info["done"]
    while done < cfg["replicates"]:
        done = pipeline.boot_step(CHUNK_REPS)["done"]
    return pipeline.boot_finish()


def _key_results(res, mcmc, boot):
    """The numbers a batch table needs, from one dataset's results."""
    me = (res.get("multiexp") or {}).get("selected")
    il = res.get("ilt")
    out = {"multiexp": None, "ilt": None, "t2d": None, "noise_sigma": res["decay"].get("noise_sigma")}
    if res.get("t2d"):
        r = res["t2d"]
        out["t2d"] = {"peaks": r["peaks"][:3], "rmse_over_amp": r["rmse_over_amp"], "chi2_per_point": r["chi2_per_point"], "model": None}
        m = r.get("model")
        if m:
            out["t2d"]["model"] = {"K": m["K"], "recommended_k": m["recommended_k"], "chi2_per_point": m["chi2_per_point"],
                                   "populations": [{k: p[k] for k in ("T2", "T2_se", "D", "D_se", "A", "fraction", "fraction_se", "supported", "D_at_limit", "T2_at_limit")} for p in m["populations"]]}
    if me:
        out["multiexp"] = {"n": me["n"], "components": [{"T2": c["T2"], "T2_se": c["T2_se"], "A": c["A"], "A_se": c["A_se"], "fraction": c["fraction"]} for c in me["components"]],
                           "offset": me["offset_value"], "mean_T2": me["mean_T2"], "mean_T2_se": me["mean_T2_se"], "total_amplitude": me["total_amplitude"],
                           "chi2_red": me["chi2_red"], "resid_sd_over_amp": me["resid_sd_over_amp"], "rss": me["rss"], "mse": me["mse"],
                           "recommended_n": (res["multiexp"].get("recommended_n"))}
    if il:
        out["ilt"] = {"alpha": il["alpha"], "peaks": il["peaks"][:3], "rmse_over_amp": il["rmse_over_amp"]}
    return out


# ── batch ────────────────────────────────────────────────────────────────────

def discover(root):
    """Every dataset under root, except in results folders (processed_*, batch_*): each folder holding acqu.par (Kea), and each .tnt file (Tecmag) in a folder without one.
    Folders are returned as folders, Tecmag datasets as the .tnt file paths. Sorted by path."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not (d.startswith("processed_") or d.startswith("batch_")))
        if "acqu.par" in filenames:
            found.append(Path(dirpath))
        else:
            found += [Path(dirpath) / f for f in sorted(filenames) if f.lower().endswith(".tnt")]
    return found


def _read_overrides(path, root):
    out = {}
    if not path:
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[str((Path(root) / r["folder"]).resolve())] = datetime.datetime.fromisoformat(r["timestamp"].strip())
    return out


def _parse_t0(spec, stamps):
    if spec == "none":
        return None
    if spec == "first":
        return min((t for t in stamps if t), default=None)
    t0 = datetime.datetime.fromisoformat(spec)
    return t0 if t0.tzinfo else t0.replace(tzinfo=datetime.timezone.utc if stamps and stamps[0] and stamps[0].tzinfo else None)


def _timestamp(folder, overrides):
    key = str(Path(folder).resolve())
    if key in overrides:
        t = overrides[key]
        return t if t.tzinfo else t.astimezone()
    if Path(folder).is_file() and Path(folder).suffix.lower() == ".tnt":      # Tecmag: the experiment date stored in the file
        dt, _note = tecmag.experiment_date(folder)
        return dt.astimezone().astimezone(datetime.timezone.utc) if dt else None
    f = _data_file(folder)
    return _utc(f.stat().st_mtime) if f else None


def _work(args):
    return run_dataset(*args[0], **args[1])


def run_batch(root, params=None, *, params_text=None, params_path=None, jobs=None, out_dir=None, command=None, quiet=False):
    """Process every dataset under root; returns the list of per-dataset summaries (also written as batch_summary.csv)."""
    root = Path(root)
    p = P.validate(params) if params is not None else P.defaults()
    jobs = jobs or p["batch"]["jobs"]
    folders = discover(root)
    overrides = _read_overrides(p["batch"]["timestamps_file"], root)
    stamps = {f: _timestamp(f, overrides) for f in folders}
    order = sorted(folders, key=lambda f: (stamps[f] is None, stamps[f] or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc), str(f)))
    _log(quiet, f"{len(order)} datasets under {root}; {jobs} job(s)")
    t0 = _parse_t0(p["batch"]["t0"], [stamps[f] for f in order])
    text = params_text if params_text is not None else P.dump_toml(p)
    # every dataset is processed with the same params; per-dataset results go inside each data folder (or the output directory)
    tasks = [((f, p), {"params_text": text, "params_path": params_path, "rel_name": str(f.relative_to(root)) or f.name, "command": command}) for f in order]
    rows = []
    if jobs == 1:
        for i, t in enumerate(tasks, 1):
            r = _work(t)
            rows.append(r)
            _log(quiet, f"[{i}/{len(tasks)}] {r['dataset']}: {r['status']}{(' — ' + r['reason']) if r['reason'] else ''} ({r['seconds']} s)")
    else:
        from concurrent.futures import ProcessPoolExecutor          # not at the top: Pyodide (the web page) has no multiprocessing
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, r in enumerate(pool.map(_work, tasks), 1):
                rows.append(r)
                _log(quiet, f"[{i}/{len(tasks)}] {r['dataset']}: {r['status']}{(' — ' + r['reason']) if r['reason'] else ''} ({r['seconds']} s)")
    for r, f in zip(rows, order):
        t = stamps[f]
        r["elapsed_min"] = (t - t0).total_seconds() / 60 if (t and t0) else None

    parent = Path(out_dir or p["output"]["directory"] or root)
    parent.mkdir(parents=True, exist_ok=True)
    bdir = _unique_dir(parent, f"batch_{_stamp()}")
    write_batch(bdir, rows, p, text, params_path, command, root, t0)
    _log(quiet, f"batch summary: {bdir / 'batch_summary.csv'}")
    return rows, bdir


def write_batch(bdir, rows, p, params_text, params_path, command, root, t0):
    fams = [("T2", "cpmg"), ("T1", "t1")]                                   # a folder may mix sequences: separate column families
    has = {lab: any(r.get("kind") == kind for r in rows) for lab, kind in fams}
    maxn = {lab: max((r["result"]["multiexp"]["n"] for r in rows if r.get("kind") == kind and r.get("result") and r["result"]["multiexp"]), default=0) for lab, kind in fams}
    head = ["dataset", "kind", "timestamp_utc", "timestamp_local", "elapsed_min", "status", "reason", "mode", "n_echoes", "excluded_first", "echo_time_us", "seconds", "output_dir"]
    for lab, _ in fams:
        for k in range(1, maxn[lab] + 1):
            head += [f"{lab}_{k}_s", f"{lab}_{k}_se", f"amplitude_{k}" if lab == "T2" else f"T1_amplitude_{k}", f"fraction_{k}" if lab == "T2" else f"T1_fraction_{k}"]
    head += ["offset", "mean_T2_s", "mean_T2_se"] + (["mean_T1_s", "mean_T1_se"] if has["T1"] else [])
    head += ["total_amplitude", "reduced_chi2", "residual_sd_over_signal_range", "noise_sigma", "ilt_alpha"]
    head += [f"ilt_peak{k}_{q}" for k in (1, 2, 3) for q in ("T2_s", "fraction")]
    has_t2d = any(r.get("kind") == "t2d" for r in rows)
    if has_t2d:
        head += [f"t2d_peak{k}_{q}" for k in (1, 2, 3) for q in ("T2_s", "D_m2s", "fraction")] + ["t2d_rmse_over_signal_range", "t2d_chi2_per_point"]
    maxk = max((len(r["result"]["t2d"]["model"]["populations"]) for r in rows if r.get("kind") == "t2d" and r.get("result") and (r["result"].get("t2d") or {}).get("model")), default=0)
    if maxk:
        head += ["t2d_model_populations", "t2d_model_chi2_per_point"]
        for k in range(1, maxk + 1):
            head += [f"t2d_pop{k}_T2_s", f"t2d_pop{k}_T2_se", f"t2d_pop{k}_D_m2s", f"t2d_pop{k}_D_se", f"t2d_pop{k}_fraction", f"t2d_pop{k}_fraction_se"]
    head += ["warnings"]
    table = []
    for r in rows:
        res = r.get("result") or {}
        me, il = res.get("multiexp"), res.get("ilt")
        line = [r["dataset"], r.get("kind"), r["timestamp_utc"], r["timestamp_local"], r.get("elapsed_min"), r["status"], r["reason"], r["mode"], r.get("n_echoes"), r.get("exclude_first"), r.get("echo_time_us"), r.get("seconds"), r["output_dir"]]
        for lab, kind in fams:
            for k in range(maxn[lab]):
                c = me["components"][k] if (me and r.get("kind") == kind and k < len(me["components"])) else None
                line += [c["T2"], c["T2_se"], c["A"], c["fraction"]] if c else [None] * 4
        t1row = r.get("kind") == "t1"
        line += [me["offset"] if me else None]
        line += [me["mean_T2"], me["mean_T2_se"]] if (me and not t1row) else [None, None]
        if has["T1"]:
            line += [me["mean_T2"], me["mean_T2_se"]] if (me and t1row) else [None, None]
        line += [me["total_amplitude"], me["chi2_red"], me["resid_sd_over_amp"]] if me else [None] * 3
        line += [res.get("noise_sigma"), il["alpha"] if il else None]
        for k in range(3):
            pk = il["peaks"][k] if il and k < len(il["peaks"]) else None
            line += [pk["T2_peak"], pk["fraction"]] if pk else [None, None]
        if has_t2d:
            td = res.get("t2d")
            for k in range(3):
                pk = td["peaks"][k] if td and k < len(td["peaks"]) else None
                line += [pk["T2_mean"], pk["D_mean"], pk["fraction"]] if pk else [None] * 3
            line += [td["rmse_over_amp"], td["chi2_per_point"]] if td else [None, None]
        if maxk:
            tm = (res.get("t2d") or {}).get("model")
            line += [tm["K"], tm["chi2_per_point"]] if tm else [None, None]
            for k in range(maxk):
                pp = tm["populations"][k] if tm and k < len(tm["populations"]) else None
                line += [pp["T2"], pp["T2_se"], pp["D"], pp["D_se"], pp["fraction"], pp["fraction_se"]] if pp else [None] * 6
        line.append("; ".join(r.get("warnings", [])))
        table.append(line)
    outputs.write_csv(bdir / "batch_summary.csv", head, table)
    (bdir / "params_used.toml").write_text(P.dump_toml(p, header=f"Batch under {root}"))
    prov = {"schema": provenance.SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "kind": "batch",
            "root": str(root), "command": command, "nmrdash": provenance.nmrdash_build(), "environment": provenance.environment(),
            "params_file": {"path": str(params_path) if params_path else None, "text": params_text},
            "timestamps": {"source": "modification time of the main data file (data.2d, else data2D_Cp.2d, data2D.dat or data.dat); for Tecmag .tnt files the experiment date stored in the file" + (f", overridden from {p['batch']['timestamps_file']}" if p["batch"]["timestamps_file"] else ""),
                           "t0": t0.isoformat() if t0 else None, "t0_rule": p["batch"]["t0"]},
            "datasets": [{"dataset": r["dataset"], "status": r["status"], "reason": r["reason"], "timestamp_utc": r["timestamp_utc"], "elapsed_min": r.get("elapsed_min"),
                          "output_dir": r["output_dir"]} for r in rows]}
    (bdir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if p["output"]["report"]:
        from . import report
        (bdir / "batch_report.pdf").write_bytes(report.build_batch(rows, p, str(root), t0))
    if p["output"]["figures"]:
        from . import figures
        for sym, kind, name in (("T2", "cpmg", "timeseries"), ("T1", "t1", "timeseries_T1")):
            ok = [r for r in rows if r["status"] == "ok" and r.get("kind") == kind and r.get("result") and r["result"]["multiexp"]]
            if len(ok) >= 2:
                fig = figures.timeseries_figure(ok, p["output"]["titles"], sym)
                figures.save_figure(fig, bdir / f"{name}.{p['output']['figure_format']}", p["output"]["figure_format"], p["output"]["figure_dpi"])
        okd = [r for r in rows if r["status"] == "ok" and r.get("kind") == "t2d" and ((r.get("result") or {}).get("t2d") or {}).get("model")]
        if len(okd) >= 2:
            fig = figures.timeseries_t2d_figure(okd, p["output"]["titles"])
            figures.save_figure(fig, bdir / f"timeseries_T2D.{p['output']['figure_format']}", p["output"]["figure_format"], p["output"]["figure_dpi"])
