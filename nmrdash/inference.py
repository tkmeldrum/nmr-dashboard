"""How numbers are estimated from the physical models in physics.py: noise, least-squares measures, model selection, intervals, regularisation.

Only NumPy and SciPy's t distribution; nothing here loads data or draws. Each function states what it computes, the assumptions behind it
and where it comes from. METHODS at the bottom lists them; the physics page (a subpage of About) is generated from that list, with links to these lines.
"""
import numpy as np
from scipy import stats


# ── preparing the signal ─────────────────────────────────────────────────────

def phase_and_integrate(td):
    """Phase-correct and integrate each echo: rotate so the total signal lies in the real channel, then sum the real part over each echo's points.

    td: (n_points, n_echoes), complex. The rotation is one zero-order phase, theta = -angle(sum of all points). Returns (decay, theta_degrees);
    decay has one value per echo. Real-valued data are summed as they are. After this, the imaginary channel holds only noise (see noise_sigma_mad).
    """
    td = np.asarray(td)
    if np.iscomplexobj(td):
        theta = -np.angle(td.sum())
        return (td * np.exp(1j * theta)).real.sum(axis=0), float(np.degrees(theta))
    return td.sum(axis=0).astype(float), 0.0


def noise_sigma_mad(values):
    """Noise standard deviation from a channel that holds only noise (the imaginary channel after phasing): 1.4826 x the median absolute deviation.

    The factor 1.4826 makes the MAD equal the standard deviation for Gaussian noise; unlike the plain SD it is insensitive to a few outliers.
    Returns None if it cannot be measured (fewer than 8 values, or zero spread). Assumes the noise is Gaussian, with the same level in both channels.
    References: docs/REFERENCES.md (noise).
    """
    v = np.asarray(values, float)
    if v.size < 8:
        return None
    sigma = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return sigma if np.isfinite(sigma) and sigma > 0 else None


# ── how well a model fits ────────────────────────────────────────────────────

def sum_squared_residuals(y, model):
    """RSS = sum_i (y_i - model_i)^2."""
    return float(np.sum((np.asarray(y, float) - np.asarray(model, float)) ** 2))


def reduced_chi2(rss, sigma, dof):
    """chi^2 / dof = RSS / (sigma^2 dof): close to 1 when the model explains everything the noise (sigma) allows; None without a noise level."""
    return float(rss / (sigma ** 2 * dof)) if sigma else None


def chi2_per_point(rss, sigma, n_points):
    """RSS / (sigma^2 N): used for the regularised inversions, which have no clean parameter count to subtract."""
    return rss / (sigma ** 2 * n_points) if sigma else None


def information_criteria(rss, n_points, n_params):
    """Akaike and Bayesian information criteria for least squares with unknown Gaussian noise:

        AIC = N ln(RSS/N) + 2 k          BIC = N ln(RSS/N) + k ln N

    Only differences between models fitted to the same data mean anything (lower is better); BIC penalises parameters more as N grows.
    References: docs/REFERENCES.md (criteria).
    """
    base = n_points * np.log(rss / n_points)
    return base + 2 * n_params, base + n_params * np.log(n_points)


# ── intervals ────────────────────────────────────────────────────────────────

def student_t_critical(level, dof):
    """The two-sided Student-t multiplier for a confidence level: an interval is estimate +/- t * standard error (e.g. 2.0 for 95 % at ~60 dof).

    Assumes the parameter estimates are approximately Gaussian with the standard errors from the least-squares covariance (linearised
    around the solution). References: docs/REFERENCES.md (interval).
    """
    return float(stats.t.ppf(0.5 + level / 2, dof))


# ── regularisation of the inversions ─────────────────────────────────────────

def second_derivative_matrix(n):
    """The curvature matrix D2, (n-2) x n, with (D2 x)[i] = x[i] - 2 x[i+1] + x[i+2]: penalising ||D2 x|| favours smooth spectra.

    Matches the curvature rows of systemmat in nnlsmooth1Dsvd.m (Matlab). Only curvature is penalised (no identity or slope term).
    """
    D2 = np.zeros((n - 2, n))
    D2[np.arange(n - 2), np.arange(n - 2)] = 1
    D2[np.arange(n - 2), np.arange(1, n - 1)] = -2
    D2[np.arange(n - 2), np.arange(2, n)] = 1
    return D2


def tikhonov_weighting(alpha, log10_step_grid, log10_step_data):
    """The weight on the curvature rows: w = 1 / sqrt(step_data^2 * step_grid^3 * alpha), as in upnnlsmooth1D.m (Matlab).

    step_grid is the spacing of the T2 (or D) grid in log10, step_data that of the echo grid (the Matlab code calls them pas and pasdata: French for "step");
    larger alpha means a smaller w, i.e. less smoothing.
    The inversion solves  min ||K x - y||^2 + w^2 ||D2 x||^2  subject to  x >= 0  (after compressing K and y with the SVD of K).
    References: docs/REFERENCES.md (regularisation).
    """
    return 1.0 / np.sqrt(log10_step_data ** 2 * log10_step_grid ** 3 * alpha)


# ── the list the physics page is generated from ────────────────────────────────

METHODS = [
    {"id": "phasing", "name": "Phasing and integrating the echoes", "equation": "θ = −arg(Σ all points);  decay_k = Σ_points Re(e^{iθ} · echo_k)",
     "describes": "One zero-order phase puts the signal in the real channel; each echo is then summed to one number, giving the decay.",
     "functions": [phase_and_integrate]},
    {"id": "noise", "name": "Noise level from the imaginary channel", "equation": "σ = 1.4826 · median(|x − median(x)|)",
     "describes": "After phasing the imaginary channel holds only noise, so its robust spread measures the noise on the real decay without any model.",
     "functions": [noise_sigma_mad]},
    {"id": "fit_quality", "name": "Fit quality", "equation": "RSS = Σ (yᵢ − mᵢ)²;   χ²_red = RSS / (σ² · dof)",
     "describes": "The residual sum of squares and its comparison with the measured noise (about 1 when the model explains everything the noise allows).",
     "functions": [sum_squared_residuals, reduced_chi2, chi2_per_point]},
    {"id": "criteria", "name": "Choosing the number of components", "equation": "AIC = N ln(RSS/N) + 2k;   BIC = N ln(RSS/N) + k ln N",
     "describes": "Penalised measures for comparing fits with different numbers of components; only differences between models mean anything.",
     "functions": [information_criteria]},
    {"id": "interval", "name": "Confidence intervals from least squares", "equation": "estimate ± t(level, dof) · SE",
     "describes": "Intervals from the least-squares covariance with a Student-t multiplier (derived quantities such as fractions use the delta method).",
     "functions": [student_t_critical]},
    {"id": "regularisation", "name": "Regularisation of the inversions", "equation": "min ‖K x − y‖² + w² ‖D₂ x‖²,  x ≥ 0;   w = 1/√(δ_data² · δ_grid³ · α), δ = spacing in log10 of the echo grid / the T₂ (or D) grid",
     "describes": "The smoothness-penalised, non-negative least-squares problem the 1D and 2D inversions solve; larger α means less smoothing.",
     "functions": [second_derivative_matrix, tikhonov_weighting]},
]
