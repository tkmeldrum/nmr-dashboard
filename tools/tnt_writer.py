"""Write minimal but valid Tecmag .tnt files, for synthetic test data (nmrdash.tecmag_io reads them back).

The layout is the one tecmag_io documents: a version tag, the TMAG header, DATA, TMG2, the PSEQ section (rows, tables, dashboard variables), LOCV, TMG4, the empty
trailing sections and INFO. Only what the reader needs is meaningful; the display and processing blocks are zeros.
"""
import re
import struct
from datetime import datetime

import numpy as np


def _s(text):
    b = text.encode("latin-1")
    return struct.pack("<i", len(b)) + b


def _i(*v):
    return struct.pack(f"<{len(v)}i", *v)


def _variable(name, value, default=None, depends="", field=1):
    """One dashboard/LOCV entry (see tecmag_io._read_variables)."""
    default = value if default is None else default
    return (_s(name) + _i(0) + _s(value) + _i(0) + _s("") + _s("") + _i(0, 0) + _s(depends) + _s(name) + _s(default) + _i(0, 0, 0, field))


def _table(name, values, kind):
    stream = " ".join(str(v) for v in values)
    head = _s(name) + _s(stream) + _s("") + _s("") + _s("")
    if kind == "phase":
        mid = _i(0) + b"HP\0\0" + _i(1)
    else:                                                              # a delay table: unknown tag, then "1" and the dimension
        mid = _i(0) + _i(1) + _i(2)
    return head + mid + _i(4) + _i(0, 0, 0, 0, 0) + _s("") + _s("") + _s("") + _i(0, 0) + _i(0)


def _row(name, cells):
    """A sequence row: ``cells`` is a list of (setting, {table kind: table name}); the first 'event' of a row is its name."""
    out = _i(len(cells) + 1) + _i(0, 0) + b"    " + _i(1, 0, 0)
    out += _s("0") + _s(name) + _s(name) + b"\0" * 56
    kinds = {"gradient_waveform": 2, "phase": 4, "gradient_amplitude": 6, "delay": 10}
    for setting, tables in cells:
        clusters = [0] * 15
        out_c = b""
        for k in range(1, 16):
            if k == 1:
                out_c += _s(setting) if setting else _i(0)
            else:
                nm = next((t for kind, t in tables.items() if kinds[kind] == k), None)
                out_c += _s(nm) if nm else _i(0)
        out += out_c
    return out


def write_tnt(path, data, *, points_per_window, dwell_s, observe_mhz=13.235, nucleus="H1", scans=64, date=None, sequence_file="CPMG", rows=(), tables=None,
              variables=None, defaults=None, locv=(), software="3.4.5 Build 151002"):
    """Write ``data`` (complex; shape (n_records, points) or (points,)) as a .tnt file.

    points_per_window : the points of one acquisition window (what TNMR shows as the 1D size; the file stores the length of the whole record separately)
    rows              : [(row name, [(setting, {table kind: table name}), ...]), ...]
    tables            : {name: (values, 'delay' | 'phase')}
    variables         : {name: value string} for the dashboard block; names in ``locv`` go to the LOCV section instead
    defaults          : {name: value string} saved defaults (the value TNMR keeps in the sequence file; default: the same as the value)
    """
    data = np.atleast_2d(np.asarray(data, complex)) if np.ndim(data) > 1 else np.asarray(data, complex)[None, :]
    n_rec, per_record = data.shape
    tables, variables, defaults = tables or {}, variables or {}, defaults or {}
    date = date or datetime(2026, 1, 15, 10, 30, 0)

    hdr = bytearray(1024)                                              # block starts at file offset 20; fields are addressed by file offset
    def put(fmt, off, *v):
        struct.pack_into(fmt, hdr, off - 20, *v)
    put("<4i", 20, points_per_window, n_rec, 1, 1)
    put("<4i", 36, per_record, n_rec, 1, 1)
    put("<i", 52, points_per_window)
    put("<3i", 72, scans, scans, 0)
    put("<d", 104, observe_mhz)
    put("<4d", 260, 0.5 / dwell_s, 5000.0, 5000.0, 5000.0)             # stored as +/- half the sweep width
    put("<4d", 292, dwell_s, 1e-4, 1e-4, 1e-4)
    put("<d", 340, dwell_s * points_per_window)
    stamp = f"{date.year}/{date.month}/{date.day} {date:%H:%M:%S}".encode()
    hdr[884 - 20:884 - 20 + len(stamp)] = stamp
    hdr[916 - 20:916 - 20 + len(nucleus)] = nucleus.encode()

    raw = np.empty(per_record * n_rec * 2, "<f4")
    flat = data.reshape(-1)
    raw[0::2], raw[1::2] = flat.real, flat.imag

    out = bytearray(b"TNT1.008" + b"TMAG" + _i(1, 1024) + bytes(hdr))
    out += b"DATA" + _i(1, raw.nbytes) + raw.tobytes()
    out += b"TMG2" + _i(1, 2048) + bytes(2048)

    pseq = b"PSEQ" + _i(1) + b"1.18 BIN" + _s(sequence_file) + _i(4, 0) + _s("NMRLab@TEST") + _s("\x00" * 8)
    pseq += _i(len(rows), 11)
    for name, cells in rows:
        pseq += _row(name, cells)
    pseq += _i(0)                                                      # the "unknown table"
    pseq += _i(len(tables))
    for name, (values, kind) in tables.items():
        pseq += _table(name, values, kind)
    pseq += _i(0)                                                      # dashboard block flag
    dash = [n for n in variables if n not in locv]
    pseq += _i(len(dash))
    for k, n in enumerate(dash):
        pseq += _variable(n, variables[n], defaults.get(n), field=k + 1)
    out += pseq
    out += b"LOCV" + _i(len(locv))
    for k, n in enumerate(locv):
        out += _variable(n, variables[n], defaults.get(n), field=2)
    out += b"TMG4" + _i(1, 16) + struct.pack("<4I", per_record, n_rec, 1, 1)
    out += b"PEAK" + _i(0) + b"INTG" + _i(0) + b"CMNT" + _i(0) + b"TMG3" + _i(0, 0)
    info = (f"Saved with 32-bit TNMR version:  {software}#Experiment Date:  {date:%Y/%m/%d %H:%M:%S}#Experiment by User:  synthetic@test").encode() + b"\0"
    out += b"INFO" + _i(1, len(info)) + info
    with open(path, "wb") as f:
        f.write(out)
    return path


def _norm(name):
    return re.sub(r"[ /]", "_", re.sub(r"[()\.:+-]", "", name.strip()))
