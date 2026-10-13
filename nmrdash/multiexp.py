"""Multi-exponential decay fitting with confidence intervals and model comparison.

    y(t) = sum_i A_i exp(-t / T2_i)  [+ c]

Uncertainties come from the least-squares covariance (Jacobian at the optimum,
scaled by the residual variance), as Student-t intervals; derived quantities
(total amplitude, fractions, amplitude-weighted mean T2) use first-order error
propagation with the full covariance. MCMC is a separate, later option.
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

N_STARTS = 12


def model(t, p, n, offset):
    y = np.zeros_like(t, dtype=float)
    for i in range(n):
        y += p[2 * i] * np.exp(-t / p[2 * i + 1])
    if offset:
        y += p[2 * n]
    return y


def _bounds(t, n, offset):
    lo_T, hi_T = t[0] / 10.0, t[-1] * 10.0
    lo = [0.0, lo_T] * n + ([-np.inf] if offset else [])
    hi = [np.inf, hi_T] * n + ([np.inf] if offset else [])
    return (lo, hi), (lo_T, hi_T)


def _starts(t, y, n, offset, lo_T, hi_T, rng, n_starts):
    base = float(np.mean(y[-max(3, len(y) // 20):])) if offset else 0.0
    amp = max(float(y[0] - base), 1e-12 * max(1.0, float(np.max(np.abs(y)))))
    T = np.geomspace(max(t[0] * 2, lo_T * 5), max(t[-1] / 2, lo_T * 10), n) if n > 1 else np.array([t[-1] / 3])
    yield np.ravel(np.column_stack([np.full(n, amp / n), T])).tolist() + ([base] if offset else [])
    for _ in range(n_starts - 1):
        T = np.sort(np.exp(rng.uniform(np.log(lo_T * 3), np.log(hi_T / 3), n)))
        A = amp * rng.dirichlet(np.ones(n))
        yield np.ravel(np.column_stack([A, T])).tolist() + ([base] if offset else [])


def _numeric_jacobian(f, p):
    f0 = np.atleast_1d(f(p))
    J = np.empty((len(f0), len(p)))
    for k in range(len(p)):
        h = 1e-6 * max(abs(p[k]), 1e-12)
        q = p.copy()
        q[k] += h
        J[:, k] = (np.atleast_1d(f(q)) - f0) / h
    return f0, J


def fit(t, y, n, offset=True, ci_level=0.95, seed=0, n_starts=N_STARTS, sigma=None):
    """Fit one model; returns a JSON-friendly dict (NaN where not estimable).

    n_starts : number of starting guesses (the best of them is kept); fewer is faster.
    sigma : noise SD of y (e.g. from the imaginary channel); enables the reduced chi-square.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    N = len(t)
    k = 2 * n + int(offset)
    if N <= k + 1:
        raise ValueError(f"{n} components need more than {k + 1} echoes (have {N})")
    bounds, (lo_T, hi_T) = _bounds(t, n, offset)
    rng = np.random.default_rng(seed)

    best = None
    for p0 in _starts(t, y, n, offset, lo_T, hi_T, rng, n_starts):
        try:
            popt, pcov = curve_fit(lambda tt, *p: model(tt, np.array(p), n, offset), t, y,
                                   p0=p0, bounds=bounds, maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum((y - model(t, popt, n, offset)) ** 2))
        if best is None or rss < best[0]:
            best = (rss, popt, pcov)
    if best is None:
        raise RuntimeError(f"{n}-component fit did not converge")
    rss, p, pcov = best

    # order components fast -> slow, permuting the covariance to match
    order = np.argsort(p[1:2 * n:2])
    idx = [j for o in order for j in (2 * o, 2 * o + 1)] + ([2 * n] if offset else [])
    p, pcov = p[idx], pcov[np.ix_(idx, idx)]

    dof = N - k
    s2 = rss / dof
    tcrit = float(stats.t.ppf(0.5 + ci_level / 2, dof))
    finite = bool(np.all(np.isfinite(pcov)))
    se = np.sqrt(np.diag(pcov)) if finite else np.full(len(p), np.nan)

    A, T = p[0:2 * n:2], p[1:2 * n:2]

    def derived(q):
        a, tt = q[0:2 * n:2], q[1:2 * n:2]
        tot = a.sum()
        return np.concatenate([[tot, (a * tt).sum() / tot], a / tot])

    d0 = derived(p)
    if finite and d0[0] > 0:
        _, J = _numeric_jacobian(derived, p)
        d_se = np.sqrt(np.clip(np.diag(J @ pcov @ J.T), 0, None))
    else:
        d_se = np.full(len(d0), np.nan)

    warnings = []
    if not finite:
        warnings.append("Parameter uncertainties could not be estimated (components are not resolved; try fewer components).")
    at_lo = [i for i in range(n) if T[i] <= lo_T * 1.01]
    at_hi = [i for i in range(n) if T[i] >= hi_T * 0.99]
    if at_lo or at_hi:
        warnings.append("A T2 hit the edge of the allowed range; treat it as unresolved.")
    if n > 1 and np.any(np.diff(T) / T[1:] < 0.1):
        warnings.append("Two T2 components are within 10% of each other; consider fewer components.")
    if np.any(A < 1e-6 * max(abs(d0[0]), 1e-300)):
        warnings.append("A component amplitude is ~0; consider fewer components.")

    if finite:
        for i in range(n):
            if T[i] - tcrit * se[2 * i + 1] <= 0:
                warnings.append(f"The T2 of component {i + 1} is poorly determined (its confidence interval includes 0).")
    comps = []
    for i in range(n):
        comps.append({
            "T2": T[i], "T2_se": se[2 * i + 1], "T2_ci": [T[i] - tcrit * se[2 * i + 1], T[i] + tcrit * se[2 * i + 1]],
            "A": A[i], "A_se": se[2 * i], "A_ci": [A[i] - tcrit * se[2 * i], A[i] + tcrit * se[2 * i]],
            "fraction": d0[2 + i], "fraction_se": d_se[2 + i],
        })
    t_dense = np.linspace(0, t[-1], 400)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # Denominator for the normalised residual: the fitted signal's range over the sampled echoes. Unlike the sum
    # of component amplitudes it cannot be inflated by degenerate (cancelling) components, so it is comparable
    # between models of different order.
    amp = float(np.ptp(model(t, p, n, offset)))
    return {
        "mse": rss / N,
        # amplitude- and N-independent measures of fit quality
        "signal_range": amp,
        "resid_sd_over_amp": float(np.sqrt(s2) / amp) if amp > 0 else None,
        "noise_sigma": sigma,
        "chi2_red": float(rss / (sigma ** 2 * dof)) if sigma else None,
        "n": n, "offset": offset, "ci_level": ci_level, "t_crit": tcrit, "dof": dof,
        "components": comps,
        "offset_value": p[2 * n] if offset else None,
        "offset_se": se[2 * n] if offset else None,
        "total_amplitude": d0[0], "total_amplitude_se": d_se[0],
        "mean_T2": d0[1], "mean_T2_se": d_se[1],
        "rss": rss, "rmse": float(np.sqrt(rss / N)), "resid_sd": float(np.sqrt(s2)),
        "r2": 1 - rss / ss_tot if ss_tot > 0 else None,
        "aic": N * np.log(rss / N) + 2 * k, "bic": N * np.log(rss / N) + k * np.log(N),
        "n_params": k,
        "fit": model(t, p, n, offset), "resid": y - model(t, p, n, offset),
        "t_dense": t_dense, "fit_dense": model(t_dense, p, n, offset),
        "warnings": warnings,
    }


def fit_with_comparison(t, y, n, offset=True, ci_level=0.95, compare_up_to=4, sigma=None):
    """Fit the chosen n and, for model selection, 1..max(n, compare_up_to) ranked by BIC.

    AIC/BIC only mean something as differences between models fitted to the same data, so each row
    carries d_aic / d_bic relative to the best model (0 for the best).
    """
    selected = fit(t, y, n, offset, ci_level, sigma=sigma)
    rows = []
    for m in range(1, max(n, compare_up_to) + 1):
        try:
            f = selected if m == n else fit(t, y, m, offset, ci_level, sigma=sigma)
        except (ValueError, RuntimeError):
            continue
        rows.append({"n": m, "n_params": f["n_params"], "rss": f["rss"], "mse": f["mse"],
                     "resid_sd_over_amp": f["resid_sd_over_amp"], "chi2_red": f["chi2_red"],
                     "aic": f["aic"], "bic": f["bic"]})
    if rows:
        a0, b0 = min(r["aic"] for r in rows), min(r["bic"] for r in rows)
        for r in rows:
            r["d_aic"], r["d_bic"] = r["aic"] - a0, r["bic"] - b0
    best = min(rows, key=lambda r: r["bic"])["n"] if rows else n
    return {"selected": selected, "comparison": rows, "recommended_n": best}
