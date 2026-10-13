"""Turn the (n_points x n_echoes) complex echo train into a decay."""
import numpy as np


def integrate(td):
    """Phase-correct and integrate each echo.

    Zero-order autophase (rotate so the total signal sits in the real channel),
    then sum the real part over all points of each echo.

    Returns (decay, phase_deg); decay has one value per echo. Same result as
    kea_io.prepare_t2_data / pythonic/run_kea_ilt.py.
    """
    td = np.asarray(td)
    if np.iscomplexobj(td):
        theta = -np.angle(td.sum())
        return (td * np.exp(1j * theta)).real.sum(axis=0), float(np.degrees(theta))
    return td.sum(axis=0).astype(float), 0.0


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
    z_um = np.fft.fftshift(np.fft.fftfreq(nfft, d=dwell_us * 1e-6)) / (gamma_mhz_per_t * g_t_per_m)
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
    """Spatial resolution set by the acquisition window: 1/(n_pts*dwell) in Hz, converted to µm.

    Zero-filling interpolates but does not add resolution, so neighbouring z points of the
    FFT are not independent on scales finer than this.
    """
    return 1e6 / (n_pts * dwell_us) / (gamma_mhz_per_t * g_t_per_m)


def position_indices(z_um, z_min, z_max, spacing_um, max_positions=400):
    """Indices into z_um of the positions to analyse: z_min, z_min+spacing, ... <= z_max."""
    if spacing_um <= 0:
        raise ValueError("The z spacing must be positive.")
    n = int(np.floor((z_max - z_min) / spacing_um + 1e-9)) + 1
    if n > max_positions:
        raise ValueError(f"{n} positions requested (maximum {max_positions}); increase the z spacing or narrow the window.")
    targets = z_min + spacing_um * np.arange(n)
    return np.unique(np.abs(z_um[None, :] - targets[:, None]).argmin(axis=1))


def position_decays(S, idx):
    """Decay at each chosen z, with its own zero-order phase so the signal is in the real channel.

    Returns (n_positions, n_echoes).
    """
    X = S[idx]
    if np.iscomplexobj(X):
        X = X * np.exp(-1j * np.angle(X.sum(axis=1)))[:, None]
    return X.real
