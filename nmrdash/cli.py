"""Command line: nmrdash run | batch | params."""
import argparse
import sys
from pathlib import Path

from . import __version__, params as P, runner

EPILOG = """examples:
  nmrdash params > params.toml                    write a commented params file to edit
  nmrdash run params.toml data/CPMG/1             process one dataset
  nmrdash batch params.toml data/series --jobs 8  process every dataset under a folder (a time series)
use "defaults" in place of params.toml to run with the built-in defaults."""


def _load(path):
    if path == "defaults":
        return P.defaults(), P.dump_toml(), None
    text = Path(path).read_text()
    return P.load(path), text, str(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nmrdash", description="Process Kea CPMG NMR data: the same analysis as the web dashboard.",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"nmrdash {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one data folder (acqu.par + data.2d)")
    r.add_argument("params", help="params.toml (or the word 'defaults')")
    r.add_argument("folder")
    r.add_argument("--out", help="write results here instead of inside the data folder")
    b = sub.add_parser("batch", help="process every dataset under a folder; writes a combined table with timestamps")
    b.add_argument("params", help="params.toml (or the word 'defaults')")
    b.add_argument("root")
    b.add_argument("--jobs", type=int, help="datasets in parallel (default: [batch] jobs in the params file)")
    b.add_argument("--out", help="write results here instead of inside the data folders")
    b.add_argument("--list", action="store_true", help="only list the datasets found, with their timestamps")
    g = sub.add_parser("params", help="write a fully commented default params file")
    g.add_argument("-o", "--output", help="file to write (default: standard output)")
    args = ap.parse_args(argv)
    command = "nmrdash " + " ".join(sys.argv[1:] if argv is None else argv)

    if args.cmd == "params":
        text = P.dump_toml()
        if args.output:
            Path(args.output).write_text(text)
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(text)
        return 0
    try:
        params, text, path = _load(args.params)
    except (OSError, ValueError) as exc:
        print(f"nmrdash: {exc}", file=sys.stderr)
        return 2
    if args.out:
        params["output"]["directory"] = args.out
    if args.cmd == "run":
        row = runner.run_dataset(args.folder, params, params_text=text, params_path=path, command=command, quiet=False)
        if row["status"] == "ok":
            print(f"done in {row['seconds']} s: {row['output_dir']}")
            for w in row["warnings"]:
                print("warning:", w)
            return 0
        print(f"{row['status']}: {row['reason']}", file=sys.stderr)
        return 1
    if args.list:
        for f in runner.discover(args.root):
            t = runner._timestamp(f, {})
            print(f"{t.isoformat(timespec='seconds') if t else 'no data.2d':26} {f}")
        return 0
    rows, bdir = runner.run_batch(args.root, params, params_text=text, params_path=path, jobs=args.jobs, command=command, quiet=False)
    failed = [r for r in rows if r["status"] == "failed"]
    print(f"{sum(r['status'] == 'ok' for r in rows)} ok, {sum(r['status'] == 'skipped' for r in rows)} skipped, {len(failed)} failed; summary in {bdir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
