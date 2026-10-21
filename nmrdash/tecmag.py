"""Tecmag (.tnt) datasets as the dashboard sees them.

A .tnt file is turned into the same objects a Kea folder gives (a params dict in Kea's vocabulary, the echo array of a CPMG, the T1 curve, the T₂–D matrix), so
everything downstream (integration, FT, fits, ILT, T₂–D, uncertainty, outputs) is shared and unchanged. tecmag_io.py reads the file; this module decides what it is.

What the sequences store (checked on files from this lab's magnets, see docs/DEVELOPMENT.md §9):

* **CPMG**: one record; the echo windows follow each other, ``nrEchoes`` windows of ``points per window`` points (the window length is the acquisition time).
* **T1 saturation recovery**: one record per recovery delay, each holding the CPMG echo train that reads out the recovery. The delays are in a table
  (microseconds) attached to the ``T1Wait`` delay; the delay between pulse centres is the table value plus the pulse width.
* **Stimulated-echo T₂–D**: one record per encoding delay τ (a table in microseconds attached to the first delay after the first 90° pulse). A record is one
  continuous acquisition through the whole train: window 0 is the stimulated echo, windows 1..nrEchoes the CPMG echoes, ``tE`` apart, each ``EchoAcqTime`` long.
  The stimulated echo is left out (the Kea convention: the first echo is at one echo time); τ between pulse centres is the table value plus the pulse width,
  and Δ (the dashboard variable ``DELTA``) is already centre to centre.

Conventions: the data are conjugated (re − i·im) as in the lab's MATLAB reader readTecmag_new.m, so the sign of the position axis in an FT follows that convention.
The gradient of the magnet is not in the file: the page and the params file carry it (default 6.59 T/m for Tecmag data, see GRADIENT_T_PER_M).
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import inference, physics
from .tecmag_io import TntError, parse_value, read_header, read_tnt

SOURCE = "tecmag"
GRADIENT_T_PER_M = 6.59                      # about this magnet's static gradient (Tyler, 2026-10-20); the user can change it in the processing settings
FIRST_PLAUSIBLE_YEAR = 2000

NUCLEI = {"H1": "1H", "F19": "19F", "1H": "1H", "19F": "19F"}
KIND_NAMES = {"cpmg": "CPMG", "t1": "T1Sat", "t2d": "SSE-T2"}          # the name Kea's acqu.par uses: sequences.kind_of understands these


def normalise(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def kind_of_sequence(name):
    """('cpmg' | 't1' | 't2d' | None, reason) from the sequence file name TNMR stores (e.g. 'CPMG', 'T1_SaturationRecovery', 'StimEchoDiffusion_...')."""
    n = normalise(name)
    if "stimecho" in n and "diffusion" in n or "sset2" in n:
        return "t2d", ""
    if n.startswith("t1") and "saturation" in n:
        return "t1", ""
    if "cpmg" in n and "t1" not in n:
        return "cpmg", ""
    what = " (a diffusion series, not supported yet)" if "diff" in n else ""
    return None, f'the sequence "{name}" is not one nmrdash can analyse (CPMG, T1 saturation recovery, stimulated-echo T₂–D){what}'


def find(path):
    """The .tnt file for a dropped path: the file itself, or the only .tnt in a folder; None if there is none. Several in one folder is an error (choose one)."""
    p = Path(path)
    if p.is_file():
        return p if p.suffix.lower() == ".tnt" else None
    found = sorted(q for q in p.glob("*") if q.is_file() and q.suffix.lower() == ".tnt") if p.is_dir() else []
    if len(found) > 1:
        raise ValueError(f"{p.name} holds {len(found)} .tnt files ({', '.join(q.name for q in found[:4])}…): open one file at a time")
    return found[0] if found else None


def _r(x):
    """A value converted from seconds without the float noise (59.99999999999999 µs -> 60)."""
    return float(f"{x:.10g}")


def _tables_in_delay_row(tnt):
    """[(setting, table values in microseconds)] for the cells of the Delay row that use a table, in sequence order."""
    out = []
    for cell in tnt.pseq["rows"].get("Delay", []):
        for kind, name in cell["tables"].items():
            t = tnt.table(name) if kind != "phase" else None
            if t and t["values"]:
                vals = [parse_value(v) for v in t["values"]]
                out.append((cell["setting"], name, [float(x) for x in vals]))
                break
    return out


def _echo_windows(tnt, kind):
    """Where the echoes are: {'pts', 'starts' (index of each CPMG echo's first point), 'n_echo', 'continuous', 'note'}."""
    h = tnt.header
    per_record = h["points_per_record"]
    acq = h["points"][0]
    dwell = h["dwell_s"][0]
    n_var = tnt.value("nrEchoes")
    notes = []
    if acq and per_record % acq == 0 and per_record != acq:                                # separate windows, one after the other
        n = per_record // acq
        if n_var is not None and int(round(n_var)) != n:
            notes.append(f"the sequence says nrEchoes = {int(round(n_var))} but the record holds {n} windows of {acq} points; {n} were used")
        return {"pts": acq, "starts": np.arange(n) * acq, "n_echo": n, "continuous": False, "notes": notes}
    # one continuous acquisition through the whole train (the stimulated-echo T2-D sequence)
    win, tE = tnt.value("EchoAcqTime"), tnt.value("tE")
    if kind != "t2d" or win is None or tE is None or n_var is None or not dwell:
        raise TntError("the layout of the echoes in this record is not one nmrdash knows (separate acquisition windows, or the stimulated-echo T₂–D train)")
    pts, step, n = int(round(win / dwell)), int(round(tE / dwell)), int(round(n_var))
    starts = np.arange(1, n + 1) * step                                                      # window 0 is the stimulated echo
    if starts[-1] + pts > per_record:
        raise TntError(f"{n} echoes of {pts} points every {step} points do not fit in the {per_record}-point record")
    return {"pts": pts, "starts": starts, "n_echo": n, "continuous": True, "notes": notes}


def _td(record, layout):
    """(points per echo, echoes), complex, from one record, conjugated to the MATLAB reader's convention."""
    idx = layout["starts"][None, :] + np.arange(layout["pts"])[:, None]
    return np.conj(record[idx])


def _centre_note(td):
    """A sanity check on the window: the echo should peak near its middle. Returns a note, or None."""
    prof = np.abs(td).mean(axis=1)
    k, n = int(np.argmax(prof)), td.shape[0]
    if n >= 8 and not (0.25 * n <= k + 0.5 <= 0.75 * n):
        return f"the echo peaks at point {k + 1} of {n} in its window (expected near the middle): check the acquisition window"
    return None


def experiment_date(path):
    """(datetime or None, note or None) from a .tnt file's header alone (fast, for listing many files)."""
    return _timestamp(read_header(path))


def _timestamp(tnt):
    """(datetime or None, note or None): the experiment date stored in the file, with a note when it cannot be right. ``tnt``: a TntFile or a header dict."""
    header = getattr(tnt, "header", tnt)
    m = re.match(r"\s*(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})", header.get("date", ""))
    if not m:
        return None, "the file has no readable experiment date"
    dt = datetime(*(int(g) for g in m.groups()))
    if dt.year < FIRST_PLAUSIBLE_YEAR or dt > datetime.now():
        return dt, f"the experiment date stored in the file ({dt:%Y-%m-%d %H:%M}) cannot be right (the instrument clock may be wrong): check it before using it to order or compare data"
    return dt, None


def _array_labels(tnt, n):
    """How to name the n records of an arrayed CPMG: from the pulse-amplitude tables of the sequence (a linear scale of 0-100, i.e. percent amplitude), else by number.

    Returns (labels, description): e.g. (['90°: 20%, 180°: 40%', ...], 'pulse amplitude (%)'). The table of each amplitude cell of the sequence is used when it holds one value
    per record, the first as the 90° pulse and the second as the 180° pulse (a CPMG's order); the user remains responsible for knowing which record is which.
    """
    cols = []
    for name, cells in tnt.pseq["rows"].items():
        if "ampl" not in name.lower():
            continue
        for cell in cells:
            for kind, tname in cell["tables"].items():
                t = tnt.table(tname) if kind != "phase" else None
                vals = [parse_value(v) for v in t["values"]] if t else []
                if len(vals) == n and all(isinstance(v, float) and v == v for v in vals):
                    cols.append((tname, vals))
    if not cols:
        return [f"record {k + 1}" for k in range(n)], "record number"
    names = ["90°", "180°"] if len(cols) == 2 else ["pulse"] if len(cols) == 1 else [f"table {c[0]}" for c in cols]
    labels = [", ".join(f"{nm}: {c[1][k]:g}%" for nm, c in zip(names, cols)) for k in range(n)]
    return labels, "pulse amplitude (%) " + " / ".join(f"{nm} {c[0]}" for nm, c in zip(names, cols))


def record_count(path):
    """How many datasets a .tnt file holds for the batch: the records of an arrayed CPMG, else 1 (cheap for files that cannot be arrayed: only the header is read)."""
    try:
        if read_header(path)["points"][1] <= 1:
            return 1
        r = load_tnt(path)
    except (TntError, ValueError):
        return 1
    return (r.get("records") or {}).get("n", 1)


def load_tnt(path, record=None):
    """Read a .tnt file as the dashboard needs it.

    An arrayed CPMG (one echo train per record, e.g. one per pulse power) gives one dataset per record: ``record`` (0-based; default the first) picks which, and the result's
    ``records`` ({n, labels, index, description}) names them all.

    Returns a dict: ``kind`` ('cpmg', 't1', 't2d' or None), ``reason`` (why not, when None), ``params`` (Kea vocabulary), ``warnings`` (list of str),
    ``timestamp`` (datetime or None), ``file`` (name), and by kind: ``td`` (CPMG: points × echoes, complex), ``t1`` ({'t', 'y'}), ``t2d`` ({'raw', 'tau_s',
    'delta_s', 'echo_s'}). Raises TntError for a file that cannot be read.
    """
    path = Path(path)
    tnt = read_tnt(path)
    seq_file = tnt.pseq.get("file") or ""
    kind, reason = kind_of_sequence(seq_file)
    h = tnt.header
    ts, ts_note = _timestamp(tnt)
    warnings = list(tnt.warnings) + ([ts_note] if ts_note else [])
    dwell_us = h["dwell_s"][0] * 1e6
    nuc = NUCLEI.get(h["nucleus"], "1H" if not h["nucleus"] else None)
    params = {"experiment": KIND_NAMES.get(kind, seq_file or "unknown"), "sequence_file": seq_file, "source": SOURCE, "dwellTime": dwell_us,
              "b1Freq": h["observe_MHz"][0], "nrScans": h["scans"], "software": tnt.software}
    if nuc:
        params["nucleus"] = nuc
    for key, var, scale in (("pulseLength", "pw", 1e6), ("repTime", "trep", 1e3)):
        v = tnt.value(var)
        if v is not None:
            params[key] = _r(v * scale)
    out = {"kind": kind, "reason": reason, "params": params, "warnings": warnings, "timestamp": ts, "file": path.name, "tnt": tnt}
    if kind is None:
        return out

    data = tnt.data
    records = data[None, :] if data.ndim == 1 else data.reshape(-1, data.shape[-1])
    try:
        layout = _echo_windows(tnt, kind)
    except TntError as exc:
        out.update(kind=kind, reason=str(exc), supported=False)
        return out
    warnings += layout["notes"]
    tE = tnt.value("tEcho") if kind == "t1" else tnt.value("tE")
    if tE is None:
        tE = tnt.value("tE") if tnt.value("tE") is not None else tnt.value("tEcho")
    if tE is None:
        out["reason"] = "the sequence has no echo time variable (tE or tEcho)"
        return out
    params.update(echoTime=_r(tE * 1e6), nrEchoes=layout["n_echo"], nrPnts=layout["pts"], acqTime=layout["pts"] * dwell_us * 1e-6)
    pw_s = (params.get("pulseLength") or 0.0) * 1e-6

    if kind == "cpmg":
        n_rec = len(records)
        k = 0 if record is None else int(record)
        if not 0 <= k < n_rec:
            raise ValueError(f"record {k + 1} does not exist: the file holds {n_rec}")
        if n_rec > 1:
            labels, desc = _array_labels(tnt, n_rec)
            out["records"] = {"n": n_rec, "labels": labels, "index": k, "description": desc}
            params.update(record=k + 1, records=n_rec, array_label=labels[k])
        td = _td(records[k], layout)
        out["td"] = td
        note = _centre_note(td)
        if note:
            warnings.append(note)
        return out

    tables = _tables_in_delay_row(tnt)
    if not tables:
        out["reason"] = "the sequence has no delay table (the recovery or encoding delays are read from one)"
        return out
    setting, name, delays_us = tables[0]
    if len(delays_us) != len(records):
        out["reason"] = f"the delay table {name} has {len(delays_us)} values but the file holds {len(records)} records"
        return out
    delays_s = np.asarray(delays_us) * 1e-6 + pw_s                                          # between pulse centres
    if kind == "t1":
        order = np.argsort(delays_s)
        tds = [_td(r, layout) for r in records]
        ref = tds[int(np.argmax(delays_s))]
        theta = -np.angle(ref.sum())                                                        # one phase, from the fully recovered record, applied to all
        amps = np.array([(td * np.exp(1j * theta)).real.sum(axis=0) for td in tds])        # (delays, echoes): each echo's amplitude
        out["t1"] = {"t": delays_s[order], "y": amps.sum(axis=1)[order], "echoes": amps[order]}      # y: the sum over the echo train (the default amplitude)
        params.update(nrPntsT1=len(delays_s))
        out["delays_from"] = f"table {name} (the delay {setting})"
        note = _centre_note(ref)
        if note:
            warnings.append(note)
        return out

    delta = tnt.value("DELTA")
    if delta is None:
        out["reason"] = "the sequence has no DELTA variable (the storage time Δ the b-values are calculated from)"
        return out
    raw = np.stack([_td(r, layout).sum(axis=0) for r in records], axis=1)                   # (echoes, delays), complex: one number per echo, as Kea's data2D_Cp.2d
    order = np.argsort(delays_s)
    out["t2d"] = {"raw": raw[:, order], "tau_s": delays_s[order], "delta_s": delta, "echo_s": tE * np.arange(1, layout["n_echo"] + 1)}
    params.update(nrPntsTau=len(delays_s), tauMin=_r(delays_s.min() * 1e3), tauMax=_r(delays_s.max() * 1e3), DELTA=_r(delta * 1e3))
    out["delays_from"] = f"table {name} (the delay {setting}) plus the pulse width"
    note = _centre_note(_td(records[0], layout))
    if note:
        warnings.append(note)
    return out
