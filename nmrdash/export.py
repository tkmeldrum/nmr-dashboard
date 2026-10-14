"""What the web page's Export button produces: a params file and a small runner script."""
import datetime

from . import params as P

SCRIPT = '''#!/usr/bin/env python3
"""Reproduce a nmrdash web-dashboard analysis on your own computer.

Setup (once):
    pip install "nmrdash[mcmc] @ git+https://github.com/tkmeldrum/nmr-dashboard"

Use (params.toml sits next to this script and holds every processing setting; edit it to change the analysis):
    python process_nmr.py DATA_FOLDER                 # one dataset (a folder with acqu.par + data.2d)
    python process_nmr.py DATA_ROOT --batch           # every dataset under a folder: a time series
    python process_nmr.py DATA_ROOT --batch --jobs 8  # ... eight at a time
    python process_nmr.py DATA_FOLDER --params other.toml --out results/

Results go in a processed_<timestamp> folder inside each data folder (raw data is never touched), each with CSV tables,
figures, summary.md, params_used.toml and provenance.json (inputs with SHA-256, settings, software versions, seeds).
A batch also writes batch_summary.csv with one row per dataset, including timestamps and elapsed time.
"""
import argparse
import sys
from pathlib import Path

from nmrdash import params as P, runner


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\\n")[0])
    ap.add_argument("folder", help="a data folder, or with --batch a folder holding many datasets")
    ap.add_argument("--batch", action="store_true", help="process every dataset under FOLDER")
    ap.add_argument("--params", default=str(Path(__file__).with_name("params.toml")), help="params file (default: params.toml next to this script)")
    ap.add_argument("--jobs", type=int, help="datasets processed in parallel (batch)")
    ap.add_argument("--out", help="write results here instead of inside the data folders")
    args = ap.parse_args()

    params = P.load(args.params)
    text = Path(args.params).read_text()
    if args.out:
        params["output"]["directory"] = args.out
    command = "python " + " ".join(sys.argv)
    if args.batch:
        rows, out = runner.run_batch(args.folder, params, params_text=text, params_path=args.params, jobs=args.jobs, command=command)
        print(f"{sum(r['status'] == 'ok' for r in rows)} of {len(rows)} datasets processed; table: {out / 'batch_summary.csv'}")
        return 1 if any(r["status"] == "failed" for r in rows) else 0
    row = runner.run_dataset(args.folder, params, params_text=text, params_path=args.params, command=command, quiet=False)
    print(row["status"], row["output_dir"] or row["reason"])
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def params_text(settings, options=None, dataset=None, style=None):
    """TOML for the settings the page used (and its uncertainty options)."""
    p = P.from_settings(settings, options)
    if style:
        p = P.with_style(p, style)
    header = "Exported from the nmrdash web dashboard on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + (f"\nDataset: {dataset}" if dataset else "")
    return P.dump_toml(p, header=header)
