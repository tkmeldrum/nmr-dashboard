"""A few-population model for T2-D data: S(t, b) = sum_k A_k exp(-t / T2_k) exp(-b D_k), A_k >= 0, no offset.

The regularised 2D map (ilt2d) is the exploratory view; this model gives each population a T2, a D and an amplitude with an interval.
Amplitudes are linear, so for fixed T2 and D they are solved exactly (non-negative least squares on a K x K system); only the 2K
log-T2 / log-D values are searched (variable projection), from starts taken from the map's regions.

Units: t in s, b as in the data (the solver's 1e-9 s/m^2 units), D in m^2/s (b * D * 1e9 is what is exponentiated, as in ilt2d).
"""
import numpy as np
from scipy import optimize, stats

from . import ilt2d

D_SCALE = ilt2d.D_SCALE
MAX_K = 4
EDGE_TOL = 1e-3            # in ln units: closer than this to a limit means "at the limit"


def basis(t, b, T, D):
    """Echo-time and b-value factors of each population: E (n_t x K), G (n_b x K)."""
    return np.exp(-np.asarray(t)[:, None] / np.asarray(T)[None, :]), np.exp(-(np.asarray(b)[:, None] * D_SCALE) * np.asarray(D)[None, :])


def predict(t, b, A, T, D):
    E, G = basis(t, b, T, D)
    return (E * np.asarray(A)[None, :]) @ G.T


def _solve_amplitudes(Y, E, G):
    """Non-negative least-squares amplitudes for fixed T2 and D, through the K x K normal equations."""
    M = (E.T @ E) * (G.T @ G)
    c = np.einsum("ek,eb,bk->k", E, Y, G)
    scale = np.sqrt(np.diag(M)) + 1e-300
    Ms, cs = M / np.outer(scale, scale), c / scale                  # unit diagonal: well conditioned whatever the amplitudes
    L = np.linalg.cholesky(Ms + 1e-12 * np.eye(len(c)))
    a, _ = optimize.nnls(L.T, np.linalg.solve(L, cs))
    return a / scale


def _residual(x, t, b, Y, K, sigma):
    T, D = np.exp(x[:K]), np.exp(x[K:])
    E, G = basis(t, b, T, D)
    A = _solve_amplitudes(Y, E, G)
    return ((Y - (E * A[None, :]) @ G.T) / sigma).ravel()


def _starts(regions, K, lo, hi, rng, n):
    """Starting (ln T2, ln D) vectors: the map's regions first, extra populations by splitting the strongest, then jittered copies."""
    lnlo, lnhi = np.log(lo), np.log(hi)
    base = [(np.log(np.clip(r["T2_mean"], lo[0], hi[0])), np.log(np.clip(r["D_mean"], lo[1], hi[1]))) for r in regions[:K]]
    while len(base) < K:                                                 # not enough regions: split the strongest, or start anywhere
        if base:
            t0, d0 = base[0]
            base.append((t0 + rng.choice([-1.0, 1.0]) * 0.7, d0 + rng.choice([-1.0, 1.0]) * 0.7))
        else:
            base.append((rng.uniform(lnlo[0], lnhi[0]), rng.uniform(lnlo[1], lnhi[1])))
    base = np.array(base)                                                # K x 2
    out = []
    for i in range(n):
        s = base + (0 if i == 0 else rng.normal(0, 0.3 + 0.1 * i, base.shape))
        s[:, 0] = np.clip(s[:, 0], lnlo[0] + 1e-3, lnhi[0] - 1e-3)
        s[:, 1] = np.clip(s[:, 1], lnlo[1] + 1e-3, lnhi[1] - 1e-3)
        out.append(np.concatenate([s[:, 0], s[:, 1]]))
    return out


def fit_params(t, b, Y, K, regions, t2_range, d_range, sigma=None, n_starts=8, seed=0):
    """The best-fit amplitudes and ln T2 / ln D of K populations (ordered by T2), without intervals: what a bootstrap replicate needs."""
    t, b, Y = np.asarray(t, float), np.asarray(b, float), np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    lo, hi = (t2_range[0], d_range[0]), (t2_range[1], d_range[1])
    lb = np.concatenate([np.full(K, np.log(lo[0])), np.full(K, np.log(lo[1]))])
    ub = np.concatenate([np.full(K, np.log(hi[0])), np.full(K, np.log(hi[1]))])
    sig = sigma if sigma else 1.0
    best = None
    for x0 in _starts(regions or [], K, lo, hi, rng, n_starts):
        try:
            r = optimize.least_squares(_residual, x0, bounds=(lb, ub), args=(t, b, Y, K, sig), method="trf", x_scale=1.0, max_nfev=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError(f"the {K}-population fit did not converge")
    order = np.argsort(best.x[:K])                                       # populations ordered by T2 (then D)
    lnT, lnD = best.x[:K][order], best.x[K:][order]
    E, G = basis(t, b, np.exp(lnT), np.exp(lnD))
    return _solve_amplitudes(Y, E, G), lnT, lnD, (lb, ub)


def fit_k(t, b, Y, K, regions, t2_range, d_range, sigma=None, n_starts=8, seed=0):
    """Least-squares fit of K populations; returns the best of several starts, with intervals from the covariance."""
    t, b, Y = np.asarray(t, float), np.asarray(b, float), np.asarray(Y, float)
    A, lnT, lnD, (lb, ub) = fit_params(t, b, Y, K, regions, t2_range, d_range, sigma, n_starts, seed)
    return _summarise(t, b, Y, A, lnT, lnD, lb[:K][0], ub[:K][0], (lb[K], ub[K]), sigma)


def _summarise(t, b, Y, A, lnT, lnD, lnT_lo, lnT_hi, lnD_lim, sigma):
    K, N = len(A), Y.size
    pred = predict(t, b, A, np.exp(lnT), np.exp(lnD))
    rss = float(((Y - pred) ** 2).sum())
    p = 3 * K
    s2 = sigma ** 2 if sigma else rss / max(N - p, 1)
    free = [(i, name) for i in range(K) for name in ("A", "T", "D")]
    # full-parameter Jacobian of the model at the solution (columns: A_k, ln T_k, ln D_k)
    J = np.zeros((N, p))
    E, G = basis(t, b, np.exp(lnT), np.exp(lnD))
    for k in range(K):
        e, g = E[:, k], G[:, k]
        base = np.outer(e, g)
        J[:, 3 * k] = base.ravel()
        J[:, 3 * k + 1] = (A[k] * np.outer(e * (t / np.exp(lnT[k])), g)).ravel()                          # d/d lnT of exp(-t/T)
        J[:, 3 * k + 2] = (A[k] * np.outer(e, g * (-(b * D_SCALE) * np.exp(lnD[k])))).ravel()             # d/d lnD of exp(-b D)
    at = np.zeros(p, bool)
    at[0::3] = A < 1e-9 * max(A.sum(), 1e-300)
    at[1::3] = (lnT - lnT_lo < EDGE_TOL) | (lnT_hi - lnT < EDGE_TOL)
    at[2::3] = (lnD - lnD_lim[0] < EDGE_TOL) | (lnD_lim[1] - lnD < EDGE_TOL)
    keep = np.flatnonzero(~at)                                           # parameters on a limit are not Gaussian: leave them out
    cov = np.full((p, p), np.nan)
    if len(keep):
        sub = J[:, keep]
        try:
            cov[np.ix_(keep, keep)] = np.linalg.pinv(sub.T @ sub) * s2
        except np.linalg.LinAlgError:
            pass
    dof = max(N - p, 1)
    tcrit = float(stats.t.ppf(0.975, dof))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    se[~np.isfinite(np.diag(cov))] = np.nan
    S = A.sum()
    pops = []
    for k in range(K):
        sA, sT, sD = se[3 * k], se[3 * k + 1], se[3 * k + 2]
        T, D = float(np.exp(lnT[k])), float(np.exp(lnD[k]))
        frac = A[k] / S if S > 0 else np.nan
        g = np.array([((k == j) * S - A[k]) / S ** 2 for j in range(K)]) if S > 0 else np.full(K, np.nan)
        cA = cov[0::3, 0::3]
        sf = float(np.sqrt(max(g @ cA @ g, 0))) if np.all(np.isfinite(cA)) and S > 0 else np.nan
        d_edge = "low" if lnD[k] - lnD_lim[0] < EDGE_TOL else "high" if lnD_lim[1] - lnD[k] < EDGE_TOL else None
        t_edge = "low" if lnT[k] - lnT_lo < EDGE_TOL else "high" if lnT_hi - lnT[k] < EDGE_TOL else None
        a_ok = np.isfinite(sA) and A[k] - tcrit * sA > 0
        pops.append({
            "A": float(A[k]), "A_se": float(sA), "fraction": float(frac), "fraction_se": sf,
            "T2": T, "T2_se": float(T * sT) if np.isfinite(sT) else np.nan, "T2_ci": [float(np.exp(lnT[k] - tcrit * sT)), float(np.exp(lnT[k] + tcrit * sT))] if np.isfinite(sT) else [np.nan, np.nan],
            "D": D, "D_se": float(D * sD) if np.isfinite(sD) else np.nan, "D_ci": [float(np.exp(lnD[k] - tcrit * sD)), float(np.exp(lnD[k] + tcrit * sD))] if np.isfinite(sD) else [np.nan, np.nan],
            "T2_at_limit": t_edge, "D_at_limit": d_edge,
            "supported": bool(a_ok and A[k] >= 1e-3 * S),
        })
    return {"K": K, "populations": pops, "rss": rss, "n_points": N, "n_params": p, "sigma": sigma,
            "chi2_per_point": float(rss / (sigma ** 2 * N)) if sigma else None,
            "aic": N * np.log(rss / N) + 2 * p, "bic": N * np.log(rss / N) + p * np.log(N), "t_crit": tcrit}


def fit(t, b, Y, regions, t2_range, d_range, sigma=None, k=None, max_k=None, seed=0):
    """Fit K = 1 .. max_k, rank by BIC, and return the chosen model with the comparison.

    k: use this many populations (the comparison is still computed); None = the one BIC prefers.
    d_range: the diffusion range the populations may occupy, in m^2/s (normally the resolvable window).
    """
    t, b, Y = np.asarray(t, float), np.asarray(b, float), np.asarray(Y, float)
    top = max(1, min(MAX_K, int(max_k or max(len(regions) + 1, 3))))
    if k:
        top = max(top, min(int(k), MAX_K))
    fits = {}
    for K in range(1, top + 1):
        fits[K] = fit_k(t, b, Y, K, regions, t2_range, d_range, sigma, seed=seed)
    rows = [{"k": K, "n_params": f["n_params"], "rss": f["rss"], "chi2_per_point": f["chi2_per_point"], "aic": f["aic"], "bic": f["bic"]} for K, f in fits.items()]
    a0, b0 = min(r["aic"] for r in rows), min(r["bic"] for r in rows)
    for r in rows:
        r["d_aic"], r["d_bic"] = r["aic"] - a0, r["bic"] - b0
    best = min(rows, key=lambda r: r["bic"])["k"]
    chosen = int(k) if k else best
    out = dict(fits[chosen])
    out.update(comparison=rows, recommended_k=best, d_window=[float(d_range[0]), float(d_range[1])], t2_range=[float(t2_range[0]), float(t2_range[1])])
    warnings = []
    for i, p in enumerate(out["populations"], 1):
        if p["D_at_limit"]:
            warnings.append(f"Population {i}: D is at the {p['D_at_limit']} limit of the allowed range ({d_range[0]:.1e}–{d_range[1]:.1e} m²/s), so the data give only a bound on it, not a value.")
        if p["T2_at_limit"]:
            warnings.append(f"Population {i}: T₂ is at the {p['T2_at_limit']} limit of the allowed range, so the data give only a bound on it.")
        if not p["supported"] and not p["D_at_limit"] and not p["T2_at_limit"]:
            warnings.append(f"Population {i} is not supported by the data (its amplitude is consistent with zero). Try fewer populations.")
    if best == top and top >= MAX_K:
        warnings.append(f"BIC was still improving at {top} populations, the most this tool fits: the data may hold more than {top}, or populations that are not simple exponentials.")
    if chosen != best:
        warnings.append(f"BIC prefers {best} population{'s' if best > 1 else ''} over the {chosen} chosen (see the model comparison).")
    if out["chi2_per_point"] is not None and out["chi2_per_point"] > 1.15:
        warnings.append(f"χ² per point is {out['chi2_per_point']:.2f}: the {chosen}-population model leaves structure in the data (more populations, a different D range, or non-exponential behaviour).")
    out["warnings"] = warnings
    return out
