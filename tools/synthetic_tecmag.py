"""Synthetic Tecmag .tnt datasets (used by make_synthetic_data.py): a CPMG train, a T1 saturation recovery and a stimulated-echo T2-D set.

The files follow the structure real TNMR sequence files have (what nmrdash.tecmag reads): the sequence name, the dashboard variables (value used and saved
default), a delay table in microseconds attached to the delay that varies, and for the T2-D sequence one continuous acquisition through the whole echo train.
The data are stored as the conjugate of the signal (re - i im), the convention nmrdash.tecmag undoes.
"""
from datetime import datetime

import numpy as np

from make_synthetic_data import cpmg_echoes, noise_sigma
from nmrdash import physics
from tnt_writer import write_tnt

DATE = datetime(2026, 1, 15, 10, 30, 0)


def _fmt_us(x):
    return f"{x:.6g}u"


def _row_delay(extra=()):
    return [("Delay", [("5u", {}), ("10u", {}), ("=pw", {}), *extra, ("=trep", {}), ("1u", {})])]


def _vars(c, **more):
    v = {"90amp": "26", "180amp": "=[90amp]*2", "pw": _fmt_us(c["pulse_us"]), "trep": f"{c['trep_ms']:g}m"}
    v.update(more)
    return v


def write_cpmg(path, noise, rng, c):
    td = cpmg_echoes(noise, rng, c)                                  # (points, echoes)
    rec = np.conj(td.T.reshape(-1))                                  # windows one after the other; stored conjugated
    return write_tnt(path, rec, points_per_window=c["n_points"], dwell_s=c["dwell_us"] * 1e-6, scans=64, date=DATE, sequence_file="CPMG",
                     rows=_row_delay([("=tEwait1", {}), ("=pw", {}), ("=tEwait2", {})]), tables={"ph0": (["0", "2"], "phase")},
                     variables=_vars(c, nrEchoes=str(c["n_echoes"]), tE=_fmt_us(c["echo_time_us"])), defaults={"nrEchoes": "64", "tE": "100u"}, locv=("tE",))


def write_t1(path, noise, rng, c):
    n_pts, n_echo, n_del = c["n_points"], c["n_echoes"], c["n_delays"]
    t = physics.geometric_grid(c["t_min_s"], c["t_max_s"], n_del)
    pw = c["pulse_us"] * 1e-6
    table_us = np.round((t - pw) * 1e6)                              # the delay between pulses; the recovery time is this plus the pulse width
    t = table_us * 1e-6 + pw
    k = np.arange(n_pts) - n_pts // 2
    shape = np.exp(-(k / (n_pts / 6)) ** 2)
    shape /= shape.sum()
    te = c["echo_time_us"] * 1e-6 * np.arange(1, n_echo + 1)
    recs = []
    plateau = c["amplitude"] * (c["offset_fraction"] + (1 - c["offset_fraction"]) * (1 - np.exp(-t[-1] / c["T1_s"]))) * shape.sum() * np.exp(-te / c["T2_s"]).sum()   # the recovered amplitude: sum over the echo train
    sigma = noise_sigma(noise, c, plateau, np.sqrt(n_pts * n_echo)) if noise else 0.0
    for ti in t:
        amp = c["amplitude"] * (c["offset_fraction"] + (1 - c["offset_fraction"]) * (1 - np.exp(-ti / c["T1_s"])))
        td = amp * np.outer(shape, np.exp(-te / c["T2_s"])) * np.exp(1j * np.radians(30.0))
        if noise:
            td = td + sigma * (rng.standard_normal(td.shape) + 1j * rng.standard_normal(td.shape))
        recs.append(np.conj(td.T.reshape(-1)))
    rows = _row_delay([("=T1Wait", {"delay": "de02"}), ("=pw", {}), ("=d1", {}), ("=pw", {}), ("=d2", {})])
    return write_tnt(path, np.array(recs), points_per_window=n_pts, dwell_s=c["dwell_us"] * 1e-6, scans=32, date=DATE, sequence_file="T1_SaturationRecovery",
                     rows=rows, tables={"ph0": (["0", "2"], "phase"), "de02": ([f"{v:.0f}" for v in table_us], "delay")},
                     variables=_vars(c, nrEchoes=str(n_echo), tEcho=_fmt_us(c["echo_time_us"]), T1Wait="1n"), locv=("tEcho",))


def write_t2d(path, noise, rng, c):
    dwell = c["dwell_us"] * 1e-6
    step, wp = int(round(c["echo_time_us"] / c["dwell_us"])), int(round(c["window_us"] / c["dwell_us"]))
    n_echo, n_tau = c["n_echoes"], c["n_tau"]
    per_record = (n_echo + 1) * step + 4
    pw = c["pulse_us"] * 1e-6
    table_us = np.round(np.linspace(*c["tau_table_us"], n_tau), 1)
    tau = table_us * 1e-6 + pw                                       # centre to centre
    b = physics.b_value_stimulated_echo(tau, c["DELTA_ms"] * 1e-3, c["G_t_per_m"], "1H")
    k = np.arange(wp) - wp // 2
    shape = np.exp(-(k / (wp / 5)) ** 2)
    shape /= shape.sum()
    te = c["echo_time_us"] * 1e-6 * np.arange(0, n_echo + 1)         # window 0 is the stimulated echo
    recs = []
    first = c["amplitude"] * sum(f * np.exp(-te[1] / T2) * np.exp(-b[0] * D) for f, T2, D in zip(c["fraction"], c["T2_s"], c["D_m2s"]))          # first CPMG echo, shortest delay
    sigma = noise_sigma(noise, c, first, np.sqrt(wp)) if noise else 0.0
    for j in range(n_tau):
        amp = np.zeros(n_echo + 1)
        for f, T2, D in zip(c["fraction"], c["T2_s"], c["D_m2s"]):
            amp += f * np.exp(-te / T2) * np.exp(-b[j] * D)
        rec = np.zeros(per_record, complex)
        for w in range(n_echo + 1):
            rec[w * step:w * step + wp] = c["amplitude"] * amp[w] * shape
        rec *= np.exp(1j * np.radians(c["phase_deg"]))
        if noise:
            rec = rec + sigma * (rng.standard_normal(rec.shape) + 1j * rng.standard_normal(rec.shape))
        recs.append(np.conj(rec))
    rows = _row_delay([("=d1", {"delay": "de02"}), ("=pw", {}), ("=d2", {}), ("=pw", {}), ("=d3", {"delay": "de12"}), ("=EchoAcqTime", {})])
    variables = _vars(c, nrEchoes=str(n_echo), tE=_fmt_us(c["echo_time_us"]), DELTA=f"{c['DELTA_ms'] * 1e3:g}u", EchoAcqTime=_fmt_us(c["window_us"]), d1="1n", d2="=[DELTA]-[pw]")
    return write_tnt(path, np.array(recs), points_per_window=per_record, dwell_s=dwell, scans=128, date=DATE, sequence_file="StimEchoDiffusion_T2D", rows=rows,
                     tables={"ph0": (["0", "2"], "phase"), "de02": ([f"{v:g}" for v in table_us], "delay"), "de12": ([f"{v - 31.5:g}" for v in table_us], "delay")},
                     variables=variables, defaults={"nrEchoes": "255"}, locv=("DELTA", "EchoAcqTime", "tE"))
