"""Generate the synthetic sample datasets in test_data/ (no real measurements are kept in this repository).

    python tools/make_synthetic_data.py [out_dir]          # default: test_data/

Kea datasets (acqu.par + the files Kea writes) for CPMG, T1Sat and SSE-T2, each as a noise-free and a noisy copy, with round-number
"truth" values (listed in TRUTH below and in test_data/README.md). Everything is seeded, so the files are reproducible byte for byte.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nmrdash import physics  # noqa: E402

SEED = 20261019

# ── what the datasets contain (the tests check that the analyses recover these) ──────────────────────────────────────────
TRUTH = {
    "cpmg": {   # a slab where a short-T2 layer (40 %) and a long-T2 layer (60 %) overlap around z = 0: the echo-window sum (the slice at z = 0) sees both,
                # the ends of the profile see only one (so the z-resolved analyses have structure)
        "T2_s": (0.005, 0.020), "fraction": (0.4, 0.6), "layers_um": ((-120.0, 30.0), (-30.0, 120.0)),
        "echo_time_us": 200, "n_echoes": 256, "n_points": 64, "dwell_us": 1.0, "noise_sigma": 12.0, "phase_deg": 35.0,
    },
    "t1": {"T1_s": 0.8, "offset_fraction": 0.03, "n_delays": 20, "t_min_s": 0.01, "t_max_s": 5.0, "noise_sigma": 60.0, "amplitude": 20000.0},
    # Tecmag-style sets (nmrdash.tecmag reads them): the same kinds of sample, at this lab's Tecmag magnet (gradient 6.59 T/m, 13.235 MHz), written as .tnt files
    "tecmag_cpmg": {
        "T2_s": (0.005, 0.020), "fraction": (0.4, 0.6), "layers_um": ((-431.0, 108.0), (-108.0, 431.0)), "G_t_per_m": 6.59,       # the same layers as the Kea set, scaled by 23.67 / 6.59
        "echo_time_us": 150, "n_echoes": 256, "n_points": 32, "dwell_us": 2.0, "noise_sigma": 12.0, "phase_deg": 35.0, "pulse_us": 6.0, "trep_ms": 1000.0,
    },
    # an arrayed CPMG in one .tnt file: six echo trains at different pulse amplitudes (linear 0-100 scale, 180 pulse = twice the 90), the signal following a nutation curve
    # (flip angle 90 degrees at power_for_90), the same two layers and T2 values at every power
    "tecmag_cpmg_series": {
        "T2_s": (0.005, 0.020), "fraction": (0.4, 0.6), "layers_um": ((-431.0, 108.0), (-108.0, 431.0)), "G_t_per_m": 6.59,
        "echo_time_us": 150, "n_echoes": 128, "n_points": 32, "dwell_us": 2.0, "noise_sigma": 12.0, "phase_deg": 35.0, "pulse_us": 4.0, "trep_ms": 400.0,
        "power90": (10, 14, 18, 22, 26, 30), "power180": (20, 28, 36, 44, 52, 60), "power_for_90": 20.0,
    },
    "tecmag_t1": {"T1_s": 0.8, "offset_fraction": 0.03, "n_delays": 16, "t_min_s": 0.01, "t_max_s": 5.0, "noise_sigma": 3.0, "amplitude": 300.0, "T2_s": 0.02,
                  "echo_time_us": 150, "n_echoes": 16, "n_points": 16, "dwell_us": 2.0, "pulse_us": 6.0, "trep_ms": 1000.0},
    "tecmag_t2d": {
        "T2_s": (0.003, 0.012), "D_m2s": (1.5e-9, 3e-10), "fraction": (0.4, 0.6), "echo_time_us": 60, "n_echoes": 300, "n_tau": 16, "tau_table_us": (40.0, 400.0),
        "DELTA_ms": 10.0, "G_t_per_m": 6.59, "dwell_us": 2.0, "window_us": 40.0, "pulse_us": 6.0, "amplitude": 300.0, "noise_sigma": 0.25, "phase_deg": 40.0, "trep_ms": 1500.0,
    },
    "t2d": {
        "T2_s": (0.004, 0.020), "D_m2s": (1.5e-10, 2e-11), "fraction": (0.4, 0.6),     # both inside what these b-values resolve (about 3.5e-13 .. 7.7e-10 m^2/s)
        "echo_time_us": 100, "n_echoes": 600, "n_tau": 25, "tau_ms": (0.178, 3.2), "DELTA_ms": 5.0, "gradient_khz_per_mm": 1000, "G_t_per_m": physics.DEFAULT_GRADIENT_T_PER_M,
        "amplitude": 300.0, "noise_sigma": 1.0, "phase_deg": 40.0,
    },
}


# ── Kea file writers ─────────────────────────────────────────────────────────────────────────────────────────────────────
def write_kea_binary(path, arr, data_type):
    """The Kea .1d/.2d layout read by nmrdash.kea_io.read_kea_binary: 12-byte tag, dataType, four dims, then little-endian float32."""
    arr = np.asarray(arr)
    dims = list(arr.shape[:1]) + ([arr.shape[1]] if arr.ndim > 1 else [1])
    if data_type == 501:                                   # complex, interleaved; arr is (x, y) and stored x-fastest
        flat = arr.reshape(-1, order="F")
        raw = np.empty(flat.size * 2, "<f4")
        raw[0::2], raw[1::2] = flat.real, flat.imag
        dims = [arr.shape[0], arr.shape[1], 1, 1]
    elif data_type == 500:
        raw = np.asarray(arr, "<f4").reshape(-1, order="F")
        dims = [arr.shape[0], 1, 1, 1]
    else:
        raise ValueError(data_type)
    with open(path, "wb") as f:
        f.write(b"SORP" + b"ATAD" + b"1.1V")
        f.write(np.array([data_type, *dims], "<i4").tobytes())
        f.write(raw.tobytes())


def write_acqu_par(path, params):
    lines = []
    for k, v in params.items():
        lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v!r}" if isinstance(v, float) else f"{k} = {v}")
    Path(path).write_text("\n".join(lines) + "\n")


def _common(experiment, name, **kw):
    p = {"90Amplitude": -10, "180Amplitude": -4, "b1Freq": 19.44, "bandwidth": "1e+003", "dataDirectory": "C:\\data\\synthetic",
         "dwellTime": 1.0, "experiment": experiment, "expName": name, "expNr": 1, "pulseLength": 2.25, "rxGain": 31, "saveData": "true"}
    p.update(kw)
    return dict(sorted(p.items(), key=lambda kv: kv[0].lower()))


def noise_sigma(noise, c, reference, norm=1.0):
    """The noise level (per point, per channel). noise: False (none), True (the dataset's own noise_sigma: used by tests that want a fixed level) or a target
    SNR: the reference amplitude (the first point of the decay, or the plateau of a recovery) divided by the SNR, and by `norm`: the factor by which the noise grows
    when the points that make up that amplitude are summed (sqrt of their number)."""
    return c["noise_sigma"] if noise is True else float(reference) / (float(noise) * norm)


# ── signal models ────────────────────────────────────────────────────────────────────────────────────────────────────────
def cpmg_echoes(noise, rng, c=None):
    """Complex echo train (n_points x n_echoes): a slab of two layers with different T2, read out in a static gradient."""
    c = c or TRUTH["cpmg"]
    n_pts, n_echo = c["n_points"], c["n_echoes"]
    t_pt = (np.arange(n_pts) - n_pts // 2) * c["dwell_us"] * 1e-6                       # time within the echo window, echo centre at index n_pts/2
    te = c["echo_time_us"] * 1e-6 * np.arange(1, n_echo + 1)
    gamma_hz_per_t = physics.GAMMA_HZ_PER_T["1H"]
    td = np.zeros((n_pts, n_echo), complex)
    for (z0, z1), w, T2 in zip(c["layers_um"], c["fraction"], c["T2_s"]):
        z = np.linspace(z0, z1, 151) * 1e-6
        rho = w / z.size
        f = gamma_hz_per_t * c.get("G_t_per_m", physics.DEFAULT_GRADIENT_T_PER_M) * z                          # Hz
        shape = np.exp(2j * np.pi * np.outer(t_pt, f)).sum(axis=1) * rho                  # (n_pts,)
        td += np.outer(shape, np.exp(-te / T2))
    td *= 3000.0 * np.exp(1j * np.radians(c["phase_deg"]))
    if noise:
        sigma = noise_sigma(noise, c, np.real(td[:, 0] * np.exp(-1j * np.radians(c["phase_deg"]))).sum(), np.sqrt(n_pts))        # SNR of the first echo's integral
        td = td + sigma * (rng.standard_normal(td.shape) + 1j * rng.standard_normal(td.shape))
    return td


def write_cpmg(d, noise, rng, name, **override):
    """`override` replaces entries of TRUTH["cpmg"] (the tests use it to make, for example, a single-exponential decay)."""
    c = {**TRUTH["cpmg"], **override}
    d.mkdir(parents=True, exist_ok=True)
    write_acqu_par(d / "acqu.par", _common(
        "CPMG", name, acqTime=c["n_points"] * c["dwell_us"] * 1e-3, echoTime=c["echo_time_us"], nrEchoes=c["n_echoes"], nrPnts=c["n_points"],
        nrScans=32, repTime=1000, usePhaseCycle="yes"))
    write_kea_binary(d / "data.2d", cpmg_echoes(noise, rng, c), 501)


def write_t1(d, noise, rng, name):
    c = TRUTH["t1"]
    d.mkdir(parents=True, exist_ok=True)
    t = physics.geometric_grid(c["t_min_s"], c["t_max_s"], c["n_delays"])
    y = c["amplitude"] * (c["offset_fraction"] + (1 - c["offset_fraction"]) * (1 - np.exp(-t / c["T1_s"])))
    if noise:
        y = y + noise_sigma(noise, c, y[-1]) * rng.standard_normal(t.size)                                  # SNR of the recovered plateau
    write_acqu_par(d / "acqu.par", _common(
        "T1Sat", name, echoTime=100, nrEchoes=64, nrPnts=16, nrPntsT1=c["n_delays"], nrScans=16, repTime=300, t1Est=500, tMax=2500, index="exp"))
    np.savetxt(d / "data.dat", np.column_stack([t, y]), fmt="%.6g", delimiter="\t")
    write_kea_binary(d / "data.1d", y, 500)


def write_t2d(d, noise, rng, name, **override):
    """`override` replaces entries of TRUTH["t2d"] (the tests use it to make a population that is slower than the b-values can resolve)."""
    c = {**TRUTH["t2d"], **override}
    d.mkdir(parents=True, exist_ok=True)
    n_echo, n_tau = c["n_echoes"], c["n_tau"]
    tau_s = physics.geometric_grid(c["tau_ms"][0] * 1e-3, c["tau_ms"][1] * 1e-3, n_tau)
    te = c["echo_time_us"] * 1e-6 * np.arange(1, n_echo + 1)
    b = physics.b_value_stimulated_echo(tau_s, c["DELTA_ms"] * 1e-3, c["G_t_per_m"], "1H")        # s/m^2, what the signal really contains
    sig = np.zeros((n_echo, n_tau))
    for f, T2, D in zip(c["fraction"], c["T2_s"], c["D_m2s"]):
        sig += f * np.exp(-te[:, None] / T2) * np.exp(-b[None, :] * D)
    sig = c["amplitude"] * sig * np.exp(1j * np.radians(c["phase_deg"]))
    if noise:
        sigma = noise_sigma(noise, c, abs(sig[0, 0]))                                                         # SNR of the first echo at the shortest delay
        sig = sig + sigma * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    write_acqu_par(d / "acqu.par", _common(
        "SSE-T2", name, acqTime=0.016, DELTA=c["DELTA_ms"], echoTime=c["echo_time_us"], gradient=c["gradient_khz_per_mm"], index="log", nrEchoes=n_echo,
        nrPnts=16, nrPntsTau=n_tau, nrScans=128, repTime=1500, tauMax=c["tau_ms"][1], tauMin=c["tau_ms"][0], usePhaseCycle="yes"))
    write_kea_binary(d / "data2D_Cp.2d", sig, 501)
    # the spectrometer's own tau / b / amplitude table, with its own gradient (23.49 T/m from gradient = 1000 kHz/mm): used only as a cross-check
    b_kea = physics.b_value_stimulated_echo(tau_s, c["DELTA_ms"] * 1e-3, physics.gradient_implied_by_kea(c["gradient_khz_per_mm"]), "1H")
    first = np.abs(sig[0]) if noise else np.abs(sig[0])
    np.savetxt(d / "data.dat", np.column_stack([tau_s * 1e3, b_kea * 1e-9, first]), fmt="%.6g", delimiter="\t")


def _tecmag():
    import synthetic_tecmag
    return synthetic_tecmag


def write_tecmag_cpmg(path, noise, rng, name, **override):
    return _tecmag().write_cpmg(path, noise, rng, {**TRUTH["tecmag_cpmg"], **override})


def write_tecmag_cpmg_series(path, noise, rng, name, **override):
    return _tecmag().write_cpmg_series(path, noise, rng, {**TRUTH["tecmag_cpmg_series"], **override})


def write_tecmag_t1(path, noise, rng, name, **override):
    return _tecmag().write_t1(path, noise, rng, {**TRUTH["tecmag_t1"], **override})


def write_tecmag_t2d(path, noise, rng, name, **override):
    return _tecmag().write_t2d(path, noise, rng, {**TRUTH["tecmag_t2d"], **override})


# each kind comes noise-free and at a signal-to-noise ratio of 100, 30 and 10: the first point of the decay (first echo; for T1 the recovered plateau; for T2-D the
# first echo at the shortest delay) over the noise standard deviation of that value (what the page measures from the imaginary channel)
SERIES_LEVELS = (("clean", False), ("snr30", 30))                                   # the arrayed CPMG comes noise-free and at SNR 30 (of its strongest record)
LEVELS = (("clean", False), ("snr100", 100), ("snr30", 30), ("snr10", 10))
NAMES = {"cpmg": ("kea_cpmg", write_cpmg), "t1": ("kea_t1", write_t1), "t2d": ("kea_t2d", write_t2d)}
TECMAG_NAMES = {"cpmg": ("tecmag_cpmg", write_tecmag_cpmg), "t1": ("tecmag_t1", write_tecmag_t1), "t2d": ("tecmag_t2d", write_tecmag_t2d)}


def generate(out):
    out = Path(out)
    made = []
    for kind, (prefix, writer) in NAMES.items():
        for variant, noise in LEVELS:
            name = f"{prefix}_{variant}"
            rng = np.random.default_rng([SEED, list(NAMES).index(kind), int(noise)])
            writer(out / name, noise, rng, name)
            made.append(name)
    for kind, (prefix, writer) in TECMAG_NAMES.items():                                  # Tecmag: one .tnt file per dataset
        for variant, noise in LEVELS:
            name = f"{prefix}_{variant}.tnt"
            rng = np.random.default_rng([SEED, 10 + list(TECMAG_NAMES).index(kind), int(noise)])
            writer(out / name, noise, rng, name)
            made.append(name)
    for variant, noise in SERIES_LEVELS:
        name = f"tecmag_cpmg_power_series_{variant}.tnt"
        write_tecmag_cpmg_series(out / name, noise, np.random.default_rng([SEED, 20, int(noise)]), name)
        made.append(name)
    return made


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "test_data"
    for n in generate(dest):
        print("wrote", dest / n)
