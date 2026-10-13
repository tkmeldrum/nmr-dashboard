"""
nmrdash/kea_io.py
-----------------
Read-only utilities for Magritek Kea binary data files.
Canonical copy: originally from MeldrumLabCode/pythonic/kea_io.py.

Functions
---------
    read_kea_binary(filepath)      →  header (dict), data (ndarray)
    read_kea_params(par_file)      →  params (dict)
    prepare_t2_data(data_file, params)  →  echo_vec (s), intdata (amplitudes)

These functions never need to change — drop this file next to your notebook
and import with:

    from nmrdash.kea_io import read_kea_binary, read_kea_params, prepare_t2_data
"""

import re
import struct
import numpy as np


# ── Binary file reader ────────────────────────────────────────────────────────

def read_kea_binary(filepath):
    """
    Read a Kea .2d or .1d binary file.

    Header layout (little-endian):
        owner     4 chars
        format    4 chars
        version   4 chars
        dataType  int32   (500–504)
        xDim      int32
        yDim      int32
        zDim      int32
        qDim      int32

    dataType key
    ------------
    500  float32, real
    501  float32 complex, interleaved (r, i, r, i, ...)
    502  float64, real
    503  float32 x-y pairs (non-interleaved)
    504  float32 scalar + complex pairs

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    header : dict
        Keys: dataType, xDim, yDim, zDim, qDim, owner, format, version
    data : ndarray
        Real or complex depending on dataType.
    """
    with open(filepath, "rb") as f:
        owner    = f.read(4)
        fmt      = f.read(4)
        version  = f.read(4)
        data_type = struct.unpack("<i", f.read(4))[0]
        x_dim     = struct.unpack("<i", f.read(4))[0]
        y_dim     = struct.unpack("<i", f.read(4))[0]
        z_dim     = struct.unpack("<i", f.read(4))[0]
        q_dim     = struct.unpack("<i", f.read(4))[0]

        total = x_dim * y_dim * z_dim * q_dim

        if data_type == 500:                     # float32, real
            data = np.frombuffer(f.read(total * 4), dtype="<f4").copy()

        elif data_type == 501:                   # float32 complex, interleaved
            raw  = np.frombuffer(f.read(total * 8), dtype="<f4")
            data = (raw[0::2] + 1j * raw[1::2]).copy()

        elif data_type == 502:                   # float64, real
            data = np.frombuffer(f.read(total * 8), dtype="<f8").copy()

        elif data_type == 503:                   # float32 x-y pairs
            raw  = np.frombuffer(f.read(total * 2 * 4), dtype="<f4")
            data = np.column_stack([raw[:total], raw[total:]])

        elif data_type == 504:                   # scalar + complex pairs
            raw  = np.frombuffer(f.read(total * 3 * 4), dtype="<f4").reshape(-1, 3)
            cpx  = raw[:x_dim, 1:3]
            data = (cpx[:, 0] + 1j * cpx[:, 1]).copy()
            y_dim = x_dim
            x_dim = 1

        else:
            raise ValueError(f"Unknown dataType in Kea binary header: {data_type}")

    header = dict(
        dataType=data_type,
        xDim=x_dim, yDim=y_dim, zDim=z_dim, qDim=q_dim,
        owner=owner, format=fmt, version=version,
    )
    return header, data


# ── Parameter file reader ─────────────────────────────────────────────────────

def _parse_value(val_str):
    """
    Convert a Kea acqu.par value string to an appropriate Python type.

    'text' or "text"  →  str
    [a b c]    →  list[float]
    integer    →  int
    float/sci  →  float
    """
    val_str = val_str.strip()
    if len(val_str) >= 2 and val_str[0] == val_str[-1] and val_str[0] in "'\"":
        return val_str[1:-1]
    if val_str.startswith("[") and val_str.endswith("]"):
        parts = re.split(r"[\s,]+", val_str[1:-1].strip())
        return [float(p) for p in parts if p]
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        return val_str


def read_kea_params(par_file):
    """
    Parse a Kea acqu.par file into a plain dict.

    Special key handling (mirrors readAllKeaParams.m):
        Keys starting with '9'  →  stored as 'amp90'
        Keys starting with '1'  →  stored as 'amp180'
        All other keys stored verbatim with their parsed value.

    Parameters
    ----------
    par_file : str or Path

    Returns
    -------
    params : dict
    """
    params = {}
    with open(par_file, "r") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if key[0] == "9":
                params["amp90"] = _parse_value(val)
            elif key[0] == "1":
                params["amp180"] = _parse_value(val)
            else:
                try:
                    params[key] = _parse_value(val)
                except Exception:
                    params[key] = val
    return params


# ── Echo amplitude extractor ──────────────────────────────────────────────────

def _autophase(timedata):
    """
    Zero-order phase correction for a complex array.
    Rotates so that total signal power sits in the real channel.
    Returns real data unchanged.
    """
    if not np.iscomplexobj(timedata):
        return timedata
    theta = -np.angle(timedata.sum())
    return timedata * np.exp(1j * theta)


def prepare_t2_data(data_file, params):
    """
    Read a Kea binary file and return the echo-time vector and
    echo-amplitude vector ready for ILT / curve fitting.

    Layout logic
    ------------
    2D file  (yDim > 1):
        timedata = reshape(spec, xDim, yDim)  [column-major, like MATLAB]
        intdata  = real(autophase(timedata)).sum(axis=0)  if complex
                   timedata.sum(axis=0)                   if real
        nr_echoes = yDim

    1D file  (yDim == 1):
        intdata  = real(autophase(spec))  if complex
                   spec                  if real
        nr_echoes = xDim

    echo_vec = echoTime [µs] × (1, 2, …, nr_echoes) × 1e-6  →  seconds

    Parameters
    ----------
    data_file : str or Path
        Path to the Kea binary data file (data.2d, data.1d, …).
    params : dict
        Must contain 'echoTime' (echo spacing in µs).
        Typically the output of read_kea_params().

    Returns
    -------
    echo_vec : ndarray, shape (nr_echoes,)
        Echo times in seconds.
    intdata  : ndarray, shape (nr_echoes,)
        Echo amplitudes (real, summed over spatial points if 2D).
    """
    header, spec = read_kea_binary(data_file)

    x_dim     = header["xDim"]
    y_dim     = header["yDim"]
    echo_time = float(params["echoTime"])   # µs

    if y_dim > 1:
        timedata = spec[: x_dim * y_dim].reshape((x_dim, y_dim), order="F")
        if np.iscomplexobj(timedata):
            timedata = _autophase(timedata)
            intdata  = timedata.real.sum(axis=0)
        else:
            intdata  = timedata.sum(axis=0)
        nr_echoes = y_dim
    else:
        if np.iscomplexobj(spec):
            spec    = _autophase(spec)
            intdata = spec.real
        else:
            intdata = np.asarray(spec, dtype=float)
        nr_echoes = x_dim

    echo_vec = 1e-6 * echo_time * np.arange(1, nr_echoes + 1)   # seconds
    return echo_vec, intdata
