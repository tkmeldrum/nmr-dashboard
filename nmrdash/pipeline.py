"""Thin entry points called by the browser worker."""
import numpy as np
import scipy

from . import __version__


def hello():
    return {
        "nmrdash": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sum": float(np.arange(10).sum()),
    }
