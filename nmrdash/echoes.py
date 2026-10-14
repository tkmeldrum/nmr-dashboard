"""Turn the (n_points x n_echoes) complex echo train into a decay."""
import numpy as np

from . import inference, physics


def noise_sigma(im):
    """Noise SD of a decay from the imaginary channel (robust MAD estimate): see inference.noise_sigma_mad."""
    return inference.noise_sigma_mad(im)


def integrate_noise_series(td):
    """The imaginary channel of the integrated decay, one value per echo (pure noise after phasing); None if real data."""
    td = np.asarray(td)
    if not np.iscomplexobj(td):
        return None
    return (td * np.exp(-1j * np.angle(td.sum()))).imag.sum(axis=0)


def integrate_noise(td):
    """Noise SD of the integrated decay (see noise_sigma); None for real-valued data."""
    series = integrate_noise_series(td)
    return None if series is None else noise_sigma(series)


def integrate(td):
    """Phase-correct and integrate each echo (inference.phase_and_integrate); same result as kea_io.prepare_t2_data / pythonic/run_kea_ilt.py."""
    return inference.phase_and_integrate(td)


AUTO_WINDOW_FRACTION = 0.10   # default z-window: where the profile exceeds 10% of its maximum


def nextpow2(n):
    return int(2 ** np.ceil(np.log2(max(n, 1))))


def ft(td, dwell_us, gamma_mhz_per_t, g_t_per_m, zero_fill=4):
    """Fourier-transform every echo along its points -> signal vs position z.

    The echo centre (the point where the mean magnitude peaks) is moved to t=0 before the
    FFT, with the zero-filling inserted in the middle, so the spectrum has no linear phase
    ramp across z. Then a zero-order autophase (as in integrate) puts the signal in the real
    channel. z is in micrometres: z = f / (gamma * G), f over fftshift(fftfreq(nfft, dwell)).

    Returns dict: z_um (nz,), S (nz, n_echoes) complex phased, center, nfft, phase_deg.
    """
    td = np.asarray(td)
    n_pts, n_echo = td.shape
    center = int(np.argmax(np.abs(td).mean(axis=1)))
    nfft = nextpow2(n_pts * 2 ** zero_fill)
    arr = np.concatenate([td[center:], np.zeros((nfft - n_pts, n_echo), td.dtype), td[:center]], axis=0)
    S = np.fft.fftshift(np.fft.fft(arr, axis=0), axes=0)
    z_um = physics.z_um_from_frequency(np.fft.fftshift(np.fft.fftfreq(nfft, d=dwell_us * 1e-6)), gamma_mhz_per_t, g_t_per_m)
    phase_deg = 0.0
    if np.iscomplexobj(S):
        theta = -np.angle(S.sum())
        S = S * np.exp(1j * theta)
        phase_deg = float(np.degrees(theta))
    return {"z_um": z_um, "S": S, "center": center, "nfft": nfft, "phase_deg": phase_deg}


def profile(S):
    """Spatial profile: magnitude of the echo-summed spectrum."""
    return np.abs(S.sum(axis=1))


def auto_window(z_um, prof, fraction=AUTO_WINDOW_FRACTION):
    """(z_min, z_max) spanning every point where the profile exceeds fraction * max."""
    above = np.flatnonzero(prof >= fraction * prof.max())
    return float(z_um[above[0]]), float(z_um[above[-1]])


def window_decay(S, z_um, z_min, z_max):
    """Decay = real part of the spectrum summed over z in [z_min, z_max], one value per echo."""
    mask = (z_um >= z_min) & (z_um <= z_max)
    if not mask.any():
        raise ValueError("The z-window contains no points; widen it.")
    return S[mask].sum(axis=0).real


def native_resolution_um(n_pts, dwell_us, gamma_mhz_per_t, g_t_per_m):
    """Spatial resolution set by the acquisition window (physics.native_resolution_um)."""
    return physics.native_resolution_um(n_pts, dwell_us, gamma_mhz_per_t, g_t_per_m)


def position_indices(z_um, z_min, z_max, spacing_um, max_positions=400):
    """Indices into z_um of the positions to analyse: z_min, z_min+spacing, ... <= z_max."""
    if spacing_um <= 0:
        raise ValueError("The z spacing must be positive.")
    n = int(np.floor((z_max - z_min) / spacing_um + 1e-9)) + 1
    if n > max_positions:
        raise ValueError(f"{n} positions requested (maximum {max_positions}); increase the z spacing or narrow the window.")
    targets = z_min + spacing_um * np.arange(n)
    return np.unique(np.abs(z_um[None, :] - targets[:, None]).argmin(axis=1))


def grid_indices(z_um, z_min, z_max):
    """Every zero-filled FFT point with z_min <= z <= z_max."""
    idx = np.flatnonzero((z_um >= z_min) & (z_um <= z_max))
    if len(idx) == 0:
        raise ValueError("The z-window contains no points; widen it.")
    return idx


def position_decays(S, idx):
    """Decay at each chosen z, with its own zero-order phase so the signal is in the real channel.

    Returns (n_positions, n_echoes).
    """
    X = S[idx]
    if np.iscomplexobj(X):
        X = X * np.exp(-1j * np.angle(X.sum(axis=1)))[:, None]
    return X.real


def window_noise_series(S, z_um, z_min, z_max):
    """Imaginary channel of the window-summed decay, one value per echo; None for real data."""
    if not np.iscomplexobj(S):
        return None
    mask = (z_um >= z_min) & (z_um <= z_max)
    return S[mask].sum(axis=0).imag


def window_noise(S, z_um, z_min, z_max):
    """Noise SD of the window-summed decay, from the imaginary channel of the phased spectrum."""
    series = window_noise_series(S, z_um, z_min, z_max)
    return None if series is None else noise_sigma(series)


def position_noise_series(S, idx):
    """Imaginary channel at each chosen z (n_positions, n_echoes), phased like position_decays; None for real data."""
    X = S[idx]
    if not np.iscomplexobj(X):
        return None
    return (X * np.exp(-1j * np.angle(X.sum(axis=1)))[:, None]).imag


def position_noise(S, idx):
    """Noise SD at each chosen z (NaN where it cannot be measured), matching position_decays' phasing."""
    series = position_noise_series(S, idx)
    if series is None:
        return np.full(len(idx), np.nan)
    return np.array([noise_sigma(row) or np.nan for row in series])
