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
