"""Which acqu.par fields the dashboard shows up front, and how they are labelled.

Edit DISPLAY_ROWS to change the curated list. Everything else in acqu.par is
available behind "Show all parameters".
"""
from datetime import datetime

# (label, acqu.par key, unit); the sequence, timestamp and pulse rows are added in build_display
DISPLAY_ROWS = [
    ("Echo time", "echoTime", "µs"),
    ("Number of echoes", "nrEchoes", ""),
    ("Points per echo", "nrPnts", ""),
    ("Number of T1 delays", "nrPntsT1", ""),
    ("Number of diffusion delays", "nrPntsTau", ""),
    ("Diffusion time Δ", "DELTA", "ms"),
    ("Dwell time", "dwellTime", "µs"),
    ("Repetition time", "repTime", "ms"),
    ("Number of scans", "nrScans", ""),
    ("Data directory", "dataDirectory", ""),
]


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def pulse_row(params):
    """Pulse length with the 90/180 amplitudes, e.g. '2.25 µs (90°/180° amp: -10/-4 dB)'."""
    pl = params.get("pulseLength")
    if pl is None:
        return None
    a90, a180 = params.get("amp90"), params.get("amp180")
    amps = ""
    if a90 is not None and a180 is not None:
        amps = f" (90°/180° amp: {_fmt(a90)}/{_fmt(a180)} dB)"
    return {"label": "Pulse length", "value": f"{_fmt(pl)} µs{amps}"}


def build_display(params, timestamp=None):
    """Curated [{label, value}, ...] list.

    timestamp : datetime when data.2d was written, or None if unknown (shown as "not available").
    """
    rows = []
    if "experiment" in params:
        rows.append({"label": "Sequence", "value": str(params["experiment"])})
    rows.append({
        "label": "File timestamp",
        "value": timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp is not None else "not available",
    })
    pr = pulse_row(params)
    if pr:
        rows.append(pr)
    for label, key, unit in DISPLAY_ROWS:
        if key in params:
            rows.append({"label": label, "value": f"{_fmt(params[key])} {unit}".strip()})
    return rows
