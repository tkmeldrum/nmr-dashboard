"""
nmrdash/tecmag_io.py
--------------------
Read-only reader for Tecmag TNMR/NTNMR ``.tnt`` files, the counterpart of kea_io.py.

The file layout is the one the lab's MATLAB readers follow (MeldrumLabCode/TecmagFunctions/readTecmag_new.m, and Read_TNT_pseq_delays.m, which comes from
Tecmag's own format notes; the field offsets below are theirs). A .tnt file is a version string followed by tagged sections::

    "TNT1.00x"   8 bytes
    TMAG  flag  size   the 1024-byte acquisition header (points, scans, frequencies, spectral widths, dwell times, date, nucleus)
    DATA  flag  size   float32 re/im pairs: every record (one per 2D/3D/4D step) is acqPoints long
    TMG2  flag  size   display and processing state (only the spectrum-mode flag is used here)
    PSEQ  ...          the pulse sequence: its rows, the tables (delays, phases, gradients) and the dashboard variables (nrEchoes, tE, pw, ...)
    LOCV / MATV        user variables (name, value), the same ones the dashboard shows
    TMG4  PEAK  INTG  LNFT  CMNT  TMG3  TMG5  PGLB   (TMG4 holds the points actually acquired)

Functions
---------
    read_tnt(source)            →  TntFile  (header, data, pseq)
    parse_value(text)           →  float (SI) or list of floats, from strings like "100u", "1.5m", "64"

``source`` is a path or the bytes of the file (so the page can pass a file held in memory). Nothing here depends on MATLAB, and the file is never written.

Conventions
-----------
* ``data`` is returned as the file stores it, ``re + 1j*im``. The MATLAB reader readTecmag_new.m returns ``re - 1j*im`` (the complex conjugate); this changes the
  sign of the frequency axis and so the direction of a profile, never the decay. ``TntFile.conjugated()`` gives that convention.
* ``data`` has shape ``(acq_points,)`` for one record and ``(n2, acq_points)``, ``(n3, n2, acq_points)``, ``(n4, n3, n2, acq_points)`` for 2D, 3D, 4D data
  (first index = the slowest dimension), as in the MATLAB reader. Here ``acq_points`` is the length of every record as stored; see ``header["points_per_record"]``.
"""
import re
import struct
from pathlib import Path

import numpy as np

# TMAG header, absolute file offsets (readTecmag_new.m / Read_TNT_hdr)
_HDR_LEN = 1044          # the TMAG block ends here; DATA follows
_DATA_TAG = 1044
_DATA_START = 1056

TIME_SUFFIXES = (("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9), ("ps", 1e-12),
                 ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("s", 1.0))


class TntError(ValueError):
    """The file is not a readable .tnt file (the message says what is wrong)."""


# ── numbers with units ───────────────────────────────────────────────────────────────────────────────────────────────────

def _parse_one(s):
    s = s.strip()
    if not s:
        return float("nan")
    scale, num = 1.0, s
    for suffix, factor in TIME_SUFFIXES:                     # longest suffixes first, as in parseTNMRvalue.m
        if len(s) > len(suffix) and s.lower().endswith(suffix.lower()):
            scale, num = factor, s[: -len(suffix)].strip()
            break
    try:
        return float(num) * scale
    except ValueError:
        return float("nan")


def parse_value(text):
    """A TNMR parameter string as a number in SI units: '100u' → 1e-4, '1.5m' → 1.5e-3, '64' → 64.0, 'Hz'/'kHz'/'MHz' scaled.
    Several lines (a table) give a list. Anything that is not a number gives nan."""
    if text is None:
        return float("nan")
    lines = [ln.strip() for ln in re.split(r"\r\n|\n|\r|\s{2,}", str(text).strip()) if ln.strip()]
    if not lines:
        return float("nan")
    vals = [_parse_one(ln) for ln in lines]
    return vals[0] if len(vals) == 1 else vals


# ── byte stream ──────────────────────────────────────────────────────────────────────────────────────────────────────────

class _Stream:
    def __init__(self, buf):
        self.b = buf
        self.pos = 0

    def seek(self, pos):
        self.pos = pos

    def eof(self):
        return self.pos >= len(self.b)

    def take(self, n):
        if n < 0 or self.pos + n > len(self.b):
            raise TntError(f"the file ends inside a section (wanted {n} bytes at offset {self.pos}, the file has {len(self.b)})")
        out = self.b[self.pos:self.pos + n]
        self.pos += n
        return out

    def int(self):
        return struct.unpack("<i", self.take(4))[0]

    def uint(self):
        return struct.unpack("<I", self.take(4))[0]

    def text(self, n):
        return self.take(n).decode("latin-1")

    def string(self, limit=1 << 20):
        """A length-prefixed string; the length is checked so a misaligned read fails loudly instead of swallowing the file."""
        n = self.int()
        if n < 0 or n > limit:
            raise TntError(f"unexpected string length {n} at offset {self.pos - 4}")
        return self.text(n)


def _fixed(b):
    return b.split(b"\0")[0].decode("latin-1").strip()


def _f64(buf, off, n=1):
    v = struct.unpack_from(f"<{n}d", buf, off)
    return v[0] if n == 1 else list(v)


def _i32(buf, off, n=1):
    v = struct.unpack_from(f"<{n}i", buf, off)
    return v[0] if n == 1 else list(v)


# ── the pulse sequence ───────────────────────────────────────────────────────────────────────────────────────────────────

def _format_name(name):
    """Tecmag's row/variable names as plain keys (dots, colons and brackets removed, spaces to underscores), as the MATLAB reader does."""
    out = re.sub(r"[()\.:+-]", "", name.strip())
    return re.sub(r"[ /]", "_", out)


_TABLE_KIND = {2: "gradient_waveform", 4: "phase", 6: "gradient_amplitude", 8: "gradient_amplitude_post", 10: "delay"}


def _read_pseq(s, warnings):
    """Walk the PSEQ section (the stream is just after the tag). Returns the sequence file name, rows, tables and variables."""
    out = {"file": None, "revision": None, "rows": {}, "tables": {}, "variables": {}, "defaults": {}, "depends": {}}
    flag = s.int()
    if flag:
        out["revision"] = s.text(8).strip()
        out["file"] = s.string()
        if out["revision"].lower().startswith("1.18"):                # TNMR 3: two extra blocks after the path
            s.int(); s.int()
            s.take(s.int())
            s.take(s.int())
        n_fields, _n_events = s.int(), s.int()
        for _ in range(n_fields):
            n_local = s.int()
            s.take(4 * 3)                                             # address, bit length, icon library type
            s.take(4 * 3)                                             # visible flag, private data, group
            field = None
            for event in range(1, n_local + 1):
                if event == 1:
                    s.string()                                       # default string
                    s.string()                                       # label
                    field = _format_name(s.string())
                    s.take(60 - 4)
                    out["rows"].setdefault(field, [])
                else:
                    cell = {"setting": "", "tables": {}}
                    out["rows"][field].append(cell)
                    for k in range(1, 16):                            # 15 four-byte clusters per event
                        n = s.int()
                        if n > 0:
                            if k == 1:                                # the value shown on the sequence screen
                                cell["setting"] = s.text(n)
                            elif k in (2, 4, 6, 8, 10):               # names of gradient, phase, amplitude and delay tables used by this cell
                                cell["tables"][_TABLE_KIND[k]] = _format_name(s.text(n))
                            elif k == 15 and n == 1:                  # acquisition properties of an Acq row: points, sweep, filter, dwell, acquisition time, dashboard link
                                for _ in range(5):
                                    s.take(s.int())
                                s.int()
                                s.take(2)
    # unknown table
    s.take(4 * s.int())
    n_vars = s.int()
    for _ in range(n_vars):
        name = _format_name(s.string())
        stream = s.string()
        values = [t for t in re.split(r"[ \r\n]+", stream.strip()) if t]
        for _ in range(3):                                            # increment schedule, value, "every pass"
            s.take(s.int())
        s.int()
        kind = s.text(4)
        dim = s.int()
        steps = None
        label = {"HS": "gradient waveform", "2G": f"{dim}D gradient amplitude (pre-acquisition)", "3G": f"{dim}D gradient amplitude (post-acquisition)",
                 "HP": f"{dim}D phase"}.get(kind[:2])
        if label is None:
            s.seek(s.pos - 8)
            num = s.int()
            s.take(4)
            label = {1: f"{dim}D delay", 14: f"{dim}D branch"}.get(num, f"{dim}D table")
        steps = s.int()
        s.take(4 * 5)
        for _ in range(3):
            s.take(s.int())
        s.int(); s.int()
        nxt = s.int()                                                  # an extra word in newer TNMR versions: back up if it starts the next table
        if nxt > 0:
            s.seek(s.pos - 4)
        out["tables"][name] = {"values": values, "type": label, "steps": steps}
    # dashboard variables: [flag][count][entries]. (The MATLAB reader treats a zero flag as "no block" and finds the same variables again in LOCV; the files
    # here have the entries right after the tables whatever the flag says, so they are read from here and LOCV only adds to them.)
    flag = s.int()
    if flag:                                                          # a named subsection ("Sequence") with a list of the variable names first
        s.string()
        for _ in range(s.int()):
            s.string()
    _read_variables(s, s.int(), out)
    return out


def _read_variables(s, count, out):
    """The entries of the dashboard block and of LOCV share one layout, per variable::

        name  a  value  b  low  high  i1  i2  depends_on  name  default_value  j1  j2  j3  j4

    ``value`` is what the dashboard shows for this run (the one that was used); ``default_value`` is what the sequence file was saved with. Both are kept:
    reading the later of two occurrences (as a text scan does) returns the saved default instead of the value used (e.g. nrEchoes 64 instead of 128).
    """
    if count < 0 or count > 10000:
        raise TntError(f"unexpected number of variables ({count})")
    for _ in range(count):
        name = _format_name(s.string())
        s.int()
        value = s.string().strip()
        s.int()
        s.string(); s.string()
        s.int(); s.int()
        depends = s.string().strip()
        s.string()
        default = s.string().strip()
        s.take(16)
        out["variables"][name] = value
        out["defaults"][name] = default
        if depends:
            out["depends"][name] = depends


def _read_locv(s, out):
    _read_variables(s, s.int(), out)


def _read_matv(s):
    return {_format_name(s.string()): s.string().strip() for _ in range(s.int())}


# ── the file ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

class TntFile:
    """The contents of one .tnt file: ``header`` (dict), ``data`` (complex array), ``pseq`` (dict), ``warnings`` (list of str)."""

    def __init__(self, header, data, pseq, warnings, path=None):
        self.header, self.data, self.pseq, self.warnings, self.path = header, data, pseq, warnings, path

    def conjugated(self):
        """The data in the convention of readTecmag_new.m (re − i·im)."""
        return np.conj(self.data)

    @property
    def software(self):
        """The TNMR version that wrote the file, from the INFO section ('3.4.5 Build 151002'), or '' for older files."""
        m = re.search(r"version:\s*([^#]+)", self.pseq.get("info", ""))
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    @property
    def variables(self):
        """The dashboard/user variables as strings (nrEchoes, tE, pw, trep, ...), with the values of the same names in tables where the dashboard has none."""
        return self.pseq["variables"]

    def value(self, name, default=None):
        """A dashboard variable in SI units (seconds for '100u'), or ``default`` if the sequence has no such variable. Names are matched ignoring case."""
        for k, v in self.variables.items():
            if k.lower() == name.lower():
                x = parse_value(v)
                return default if (isinstance(x, float) and np.isnan(x)) else x
        return default

    def table(self, name):
        """A table (delays, phases, ...) by name, ignoring case: ``{'values': [...], 'type': str, 'steps': int}`` with the values still as the strings TNMR stores, or None."""
        for k, v in self.pseq["tables"].items():
            if k.lower() == name.lower():
                return v
        return None


def _header(buf):
    h = {}
    h["version"] = buf[:8].decode("latin-1").strip("\0 ")
    h["points"] = _i32(buf, 20, 4)                                    # requested, per dimension (1D = points in one acquisition window)
    h["points_done"] = _i32(buf, 36, 4)                               # completed
    h["acq_points"] = _i32(buf, 52)
    h["scans"], h["scans_done"], h["dummy_scans"] = _i32(buf, 72), _i32(buf, 76), _i32(buf, 80)
    h["magnet_field_T"] = _f64(buf, 96)
    ob = _f64(buf, 104, 4)
    h["observe_MHz"] = ob
    h["sweep_hz"] = [2 * v for v in _f64(buf, 260, 4)]               # stored as +/- half width
    h["dwell_s"] = _f64(buf, 292, 4)
    h["acq_time_s"] = _f64(buf, 340)
    h["last_delay_s"] = _f64(buf, 348)
    h["date"] = _fixed(buf[884:916])
    h["nucleus"] = _fixed(buf[916:932])
    h["start_time"], h["finish_time"] = _i32(buf, 872), _i32(buf, 876)
    return h


def read_tnt(source):
    """Read a .tnt file (a path, or the bytes of the file) and return a TntFile."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        buf, path = bytes(source), None
    else:
        path = Path(source)
        buf = path.read_bytes()
    if len(buf) < _DATA_START or buf[:3] != b"TNT":
        raise TntError("this is not a Tecmag .tnt file (it does not start with 'TNT')" if len(buf) >= 8 else "the file is too short to be a Tecmag .tnt file")
    if buf[8:12] != b"TMAG" or buf[_DATA_TAG:_DATA_TAG + 4] != b"DATA":
        raise TntError("the file does not have the TMAG header and DATA sections of a .tnt file")
    warnings = []
    h = _header(buf)
    data_bytes = struct.unpack_from("<I", buf, _DATA_TAG + 8)[0]
    data_end = _DATA_START + data_bytes
    if data_end > len(buf):
        raise TntError(f"the file is truncated: the data section needs {data_bytes} bytes but only {len(buf) - _DATA_START} are there")
    n_complex = data_bytes // 8
    raw = np.frombuffer(buf, dtype="<f4", count=n_complex * 2, offset=_DATA_START)
    flat = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)

    # sections after the data: TMG2, PSEQ, ... (each is a tag followed by its own layout; the tag names are searched in order, as the stream is walked)
    s = _Stream(buf)
    s.seek(data_end)
    pseq = {"file": None, "revision": None, "rows": {}, "tables": {}, "variables": {}, "defaults": {}, "depends": {}}
    h["smode"] = 0
    comment, tmg4 = "", None
    try:
        while not s.eof():
            tag = s.text(4).upper()
            if tag == "TMG2":
                s.int()
                size = s.int()
                start = s.pos
                h["smode"] = struct.unpack_from("<h", buf, start + 828)[0] if size > 830 else 0
                s.seek(start + size)
            elif tag == "PSEQ":
                parsed = _read_pseq(s, warnings)
                for k in ("file", "revision", "rows", "tables"):
                    pseq[k] = parsed[k]
                for k in ("variables", "defaults", "depends"):
                    pseq[k].update(parsed[k])
            elif tag == "SEQC":
                s.take(s.int())
            elif tag == "LOCV":
                _read_locv(s, pseq)
            elif tag == "MATV":
                for k, v in _read_matv(s).items():                         # the same variables again (name, value as text)
                    pseq["variables"].setdefault(k, v)
            elif tag == "TMG4":
                if s.int():
                    size = s.int()
                    tmg4 = list(struct.unpack_from(f"<{size // 4}I", buf, s.pos))
                    s.take(size)
            elif tag in ("PEAK", "TEQA", "LNFT"):
                if s.int():
                    s.take(s.int())
            elif tag == "INTG":
                if s.int():
                    s.take(48 * s.int())
            elif tag == "CMNT":
                if s.int():
                    comment = s.text(s.int()).strip()
            elif tag == "TMG3":
                s.int()
                s.take(520 * s.int())
            elif tag == "TMG5":
                if s.int():
                    s.take(152 * s.int())
            elif tag == "PGLB":
                s.int()
            elif tag == "INFO":                                        # "Saved with ... version: x#Experiment Date: ...#..." (TNMR 3 and later)
                if s.int():
                    pseq["info"] = s.text(s.int()).strip("\0 ")
            else:
                warnings.append(f"unknown section {tag!r} at offset {s.pos - 4}: the rest of the file was not read")
                break
    except TntError as exc:
        warnings.append(f"the sequence information could not be read completely: {exc}")
    pseq["comment"] = comment
    pseq.setdefault("info", "")

    # points actually acquired: the TMG4 section knows (the TMAG header holds what was requested)
    done = list(h["points_done"])
    if tmg4 and len(tmg4) >= 4 and all(v >= 1 for v in tmg4[:4]):
        done = [int(v) for v in tmg4[:4]]
    h["points_done"] = done
    per_record = max(done[0], h["points"][0], 1)
    n2, n3, n4 = max(done[1], h["points"][1], 1), max(done[2], h["points"][2], 1), max(done[3], h["points"][3], 1)
    h["points_per_record"] = per_record
    h["records"] = [n2, n3, n4]
    h["dimensions"] = 4 if n4 > 1 else 3 if n3 > 1 else 2 if n2 > 1 else 1
    need = per_record * n2 * n3 * n4
    if flat.size < need:
        warnings.append(f"the data section holds {flat.size} points but the header says {need}: the missing records are zero")
        flat = np.concatenate([flat, np.zeros(need - flat.size, complex)])
    elif flat.size > need:
        # the header's counts describe fewer points than are stored: the stored length is the record length when there is just one record
        if n2 == n3 == n4 == 1:
            per_record, need = flat.size, flat.size
            h["points_per_record"] = per_record
        else:
            warnings.append(f"the data section holds {flat.size} points but the header accounts for {need}: the extra points were ignored")
    data = flat[:need].reshape((n4, n3, n2, per_record))
    data = data.reshape(tuple(d for d in (n4, n3, n2) if d > 1) + (per_record,))
    return TntFile(h, data, pseq, warnings, path)
