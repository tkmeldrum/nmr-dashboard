"""1D ILT (regularized NNLS). Canonical copy: originally MeldrumLabCode/pythonic/run_1dilt.py."""
import numpy as np
from scipy.optimize import lsq_linear

from . import inference, physics


def build_tau_grid(tau_min, tau_max, n_steps):
    """
    Build a logarithmically spaced grid of relaxation times (tau values).

    This matches the tau grid construction in upnnlsmooth1D.m:
        pas = (log10(Tamax) - log10(Tamin)) / (steps - 1)
        tau = 10.^(log10(Tamin) : pas : log10(Tamax))

    Parameters
    ----------
    tau_min : float
        Minimum tau value (e.g. shortest T2 time in seconds).
    tau_max : float
        Maximum tau value (e.g. longest T2 time in seconds).
    n_steps : int
        Number of tau points. Should be >= 4 (Matlab enforces this).

    Returns
    -------
    tau : ndarray of shape (n_steps,)
        Log-spaced tau grid.
    """
    return np.logspace(np.log10(tau_min), np.log10(tau_max), n_steps)


def build_kernel(echo_vec, tau):
    """The exponential decay kernel K[i, j] = exp(-echo_vec[i] / tau[j]) (physics.t2_kernel; matches systemmat = exp(-time' ./ tau) in nnlsmooth1Dsvd.m)."""
    return physics.t2_kernel(echo_vec, tau)


second_derivative_matrix = inference.second_derivative_matrix      # the curvature penalty; defined in inference.py


def run_1dilt(data, echo_vec, params, alpha=None):
    """
    Nonnegative 1D inverse Laplace transform (ILT) for T2 relaxometry.

    Solves the regularized nonnegative least-squares problem:

        min  ||K @ x - data||^2 + weighting^2 * ||D2 @ x||^2
        s.t. x >= 0

    where K is the exponential kernel, D2 is the second-difference
    (curvature) matrix, and weighting is a scalar derived from alpha
    and the log-spacings of the tau and echo grids.

    This is designed to match the behavior of run_1DILT.m / upnnlsmooth1D.m
    / nnlsmooth1Dsvd.m from the Matlab ILT codebase, specifically when
    called with beta = -1 (as run_1DILT.m always does). With beta = -1,
    the Matlab UP loop runs only once and uses a uniform (non-adaptive)
    weighting scalar. Only D2 regularization is applied — there is no
    identity (L2) or D1 (slope) term, unlike some other ILT formulations.

    Key design decisions made to match Matlab:
    - Only D2 (curvature) regularization is used. Earlier Python versions
      also included alpha*I and alpha*D1 terms, which are NOT present in
      the Matlab 1D solver and produced over-regularized results.
    - The weighting scalar is computed from the log-spacings of the tau
      and echo grids (pas and pasdata), exactly as in upnnlsmooth1D.m.
    - No column normalization is applied (unlike some earlier versions),
      to keep the numerical path as close to Matlab as possible.
    - The thin SVD compression step from nnlsmooth1Dsvd.m is replicated:
      K and data are projected onto the left singular vectors of K before
      building the augmented system. This is NOT merely a numerical detail —
      it reduces the data block from n_echo rows to n_tau rows, preserving
      the correct ratio of data vs regularization rows regardless of how
      many echoes were acquired. Without it, regularization is diluted by a
      factor of n_echo/n_tau relative to MATLAB for the same alpha.
    - The solver is scipy lsq_linear (trust-region reflective), whereas
      Matlab uses the Lawson-Hanson active-set NNLS algorithm. Both
      enforce nonnegativity and minimize the same objective; solutions
      should agree to within numerical precision.

    Parameters
    ----------
    data : array-like of shape (n_echoes,)
        Measured echo amplitudes.
    echo_vec : array-like of shape (n_echoes,)
        Echo times, in the same units as ILTminmax. Must be positive
        and strictly increasing (required for pasdata calculation).
    params : dict with keys:
        'ILTminmax' : sequence of two floats
            [tau_min, tau_max] — range of the tau grid.
        'nrILTSteps' : int
            Number of tau grid points. Will be capped at len(echo_vec),
            matching the Matlab behavior in run_1DILT.m.
        'alpha' : float, optional
            Regularization strength. Used if alpha argument is not given.
    alpha : float, optional
        Regularization strength. Overrides params['alpha'] if provided.
        Must be positive. Larger values produce smoother spectra.
        Matlab enforces that 1/sqrt(pasdata^2 * pas^3 * alpha) <= 300;
        values of alpha that violate this are too small to be meaningful.

    Returns
    -------
    spectrum : ndarray of shape (n_tau,)
        Nonnegative T2 amplitude spectrum (the ILT solution).
    tau : ndarray of shape (n_tau,)
        Log-spaced tau grid used in the inversion.
    pred : ndarray of shape (n_echoes,)
        Predicted echo decay: K @ spectrum.
    resid : ndarray of shape (n_echoes,)
        Residuals: data - pred.
    rmse : float
        Root mean square error of the fit.
    """
    data = np.asarray(data, dtype=float).ravel()
    echo_vec = np.asarray(echo_vec, dtype=float).ravel()

    # Drop t=0 points: exp(-0/tau)=1 for all tau, so they add no information
    # and cause log10(0) -> -inf in the pasdata calculation.
    pos_mask = echo_vec > 0
    if not np.all(pos_mask):
        echo_vec = echo_vec[pos_mask]
        data = data[pos_mask]

    if alpha is None:
        if "alpha" not in params:
            raise ValueError("Alpha must be provided either as argument or in params.")
        alpha = params["alpha"]
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")

    tau_min, tau_max = map(float, params["ILTminmax"])
    n_steps = int(min(params["nrILTSteps"], len(echo_vec)))
    tau = build_tau_grid(tau_min, tau_max, n_steps)
    n = len(tau)

    # Compute log-spacings matching Matlab's pas and pasdata variables.
    # pas: spacing of the tau grid in log10 space.
    # pasdata: spacing of the echo grid in log10 space.
    # These appear in the Matlab weighting formula:
    #   weighting = 1 / sqrt(pasdata^2 * pas^3 * alpha)
    pas = (np.log10(tau_max) - np.log10(tau_min)) / (n_steps - 1)
    echo_pos = echo_vec[echo_vec > 0]
    if len(echo_pos) < 2:
        raise ValueError("echo_vec must contain at least two positive values for pasdata calculation.")
    pasdata = (np.log10(echo_pos[-1]) - np.log10(echo_pos[0])) / (len(echo_pos) - 1)
    weighting = inference.tikhonov_weighting(alpha, pas, pasdata)

    K = build_kernel(echo_vec, tau)

    # SVD compression: matches the svdecon step in nnlsmooth1Dsvd.m.
    #   [u,~,~] = svdecon(systemmat)
    #   systemmat = u'*systemmat   →  (n_tau, n_tau)
    #   data      = u'*data'       →  (n_tau,)
    # This reduces the data block from n_echo rows to n_tau rows.
    # Without it the regularization rows are diluted by n_echo/n_tau
    # relative to MATLAB, producing under-regularized spectra.
    U, _, _ = np.linalg.svd(K, full_matrices=False)  # U: (n_echo, n_tau)
    K_c = U.T @ K       # (n_tau, n_tau)
    data_c = U.T @ data  # (n_tau,)

    # Build the augmented system [K_c; weighting * D2] @ x ~ [data_c; 0]
    # Matches the structure of systemmat in nnlsmooth1Dsvd.m after SVD
    # compression, where the curvature rows are appended with per-element
    # weighting (uniform when beta=-1 and initial spectrum is zero).
    D2 = second_derivative_matrix(n)
    A = np.vstack([K_c, weighting * D2])
    b = np.concatenate([data_c, np.zeros(n - 2)])

    # Solve nonnegative least squares. scipy's TRF method is not identical
    # to the Lawson-Hanson algorithm used in Matlab, but minimizes the same
    # objective and enforces x >= 0.
    result = lsq_linear(A, b, bounds=(0, np.inf), method="trf")
    spectrum = result.x

    pred = K @ spectrum
    resid = data - pred
    rmse = np.sqrt(np.mean(resid**2))

    return spectrum, tau, pred, resid, rmse