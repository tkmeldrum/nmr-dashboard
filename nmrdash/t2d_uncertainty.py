"""MCMC and bootstrap for the T2-D few-population model (t2d_model.py; plan in docs/PLAN_T2D_UNCERTAINTY.md).

Same shape as the 1D versions in uncertainty.py: small chunked objects (`step(k)`, then `result()`), so the page can show progress and stop early.

MCMC (emcee): sampled A_k >= 0 (flat), ln T2_k and ln D_k (uniform inside the allowed ranges: the resolvable diffusion window by default),
populations ordered T2_1 < T2_2 < ... so labels cannot swap, and optionally ln sigma. No offset. The log-likelihood is evaluated exactly without
building the model matrices, from the data's overlaps with each population's two factors:
    RSS = |Y|^2 - 2 sum_k A_k <e_k, Y g_k> + sum_kl A_k A_l <e_k, e_l> <g_k, g_l>.
Bootstrap: surrogate data = the fitted model + resampled noise (the measured imaginary-channel values, the fit's residuals, or Gaussian noise),
each refitted from the least-squares solution with the same number of populations.
"""
import time

import numpy as np

from . import figures, physics, t2d_model as M
from .uncertainty import BOOT_METHODS, MAX_CORNER_SAMPLES, MAX_TRACE_POINTS, MAX_TRACE_WALKERS, TRIAL_STEPS, _interval, _summ

MAX_SCATTER = 3000


def param_labels(K):
    """(plain names, TeX names) for [A1, T2_1, D1, A2, ...]; for one population the numbers are dropped."""
    one = K == 1
    plain, tex = [], []
    for i in range(1, K + 1):
        plain += ["A" if one else f"A{i}", "T2" if one else f"T2,{i}", "D" if one else f"D{i}"]
        tex += [r"$A$" if one else rf"$A_{i}$", r"$\log_{10}T_2$" if one else rf"$\log_{{10}}T_{{2,{i}}}$", r"$\log_{10}D$" if one else rf"$\log_{{10}}D_{i}$"]
    return plain, tex


def derived_labels(K):
    return [] if K == 1 else ["total amplitude", "mean T2", "mean D"] + [f"fraction {i}" for i in range(1, K + 1)]


def derived(p, K):
    """[total amplitude, amplitude-weighted mean T2 and D, fractions...] for flat natural [A, T2, D] * K rows."""
    p = np.atleast_2d(p)
    A, T, D = p[:, 0:3 * K:3], p[:, 1:3 * K:3], p[:, 2:3 * K:3]
    tot = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([tot, (A * T).sum(axis=1) / tot, (A * D).sum(axis=1) / tot, A / tot[:, None]])


def _flat(f):
    """[A1, T1, D1, A2, ...] from a t2d_model fit result."""
    return np.array([v for p in f["populations"] for v in (p["A"], p["T2"], p["D"])], float)


def _ls_se(f):
    return [v for p in f["populations"] for v in (p["A_se"], p["T2_se"], p["D_se"])]


def _scatter(flat, K, rng):
    """Samples of every population's (T2, D), for drawing over the 2D map."""
    sub = flat[rng.choice(len(flat), size=min(MAX_SCATTER // K, len(flat)), replace=False)]
    return {"T2": [sub[:, 3 * k + 1] for k in range(K)], "D": [sub[:, 3 * k + 2] for k in range(K)]}


def _edge_warnings(f, what):
    out = []
    for i, p in enumerate(f["populations"], 1):
        for key, name in (("D_at_limit", "D"), ("T2_at_limit", "T2")):
            if p[key]:
                out.append(f"The least-squares fit puts {name} of population {i} at the {p[key]} edge of the range allowed, so the data give only a bound on it: "
                           f"its {what} interval is cut off by that range, not set by the data.")
    return out


class Mcmc2D:
    def __init__(self, t, b, Y, fit, t2_range, d_range, sigma=None, ci_level=0.95, nwalkers=32, steps=6000, burn=2000, seed=0):
        try:
            import emcee
        except ImportError as exc:
            raise RuntimeError("MCMC needs the 'emcee' package (pip install emcee).") from exc
        self.emcee = emcee
        self.t, self.b, self.Y = np.asarray(t, float), np.asarray(b, float), np.asarray(Y, float)
        self.fit, self.K, self.level = fit, int(fit["K"]), float(ci_level)
        self.sample_sigma = sigma is None
        self.sigma = None if sigma is None else float(sigma)
        self.steps, self.burn, self.seed, self.done = int(steps), int(burn), int(seed), 0
        self.ndim = 3 * self.K + int(self.sample_sigma)
        self.nwalkers = max(int(nwalkers), 2 * self.ndim, 8)
        self.nwalkers += self.nwalkers % 2
        self.lnT, self.lnD = np.log(t2_range), np.log(d_range)
        self.N, self.yy = self.Y.size, float((self.Y ** 2).sum())
        self.bs = self.b * M.D_SCALE
        self.p_ls = _flat(fit)
        self.sd0 = float(np.sqrt(fit["rss"] / max(self.N - fit["n_params"], 1)))
        self.lns_lo, self.lns_hi = np.log(self.sd0 / 20.0), np.log(self.sd0 * 20.0)
        self.rng = np.random.default_rng(self.seed)
        self.theta0 = self._theta(self.p_ls)
        self.p0 = self._init_walkers()
        self._new_sampler()
        self.est_seconds = self._time_trial()

    def _theta(self, p):
        th = []
        for k in range(self.K):
            th += [p[3 * k], np.log(p[3 * k + 1]), np.log(p[3 * k + 2])]
        if self.sample_sigma:
            th.append(np.log(self.sd0))
        return np.array(th, float)

    def _lnprob(self, th):
        th = np.atleast_2d(th)
        K = self.K
        A, lnT, lnD = th[:, 0:3 * K:3], th[:, 1:3 * K:3], th[:, 2:3 * K:3]
        ok = np.all(A >= 0, axis=1) & np.all((lnT > self.lnT[0]) & (lnT < self.lnT[1]), axis=1) & np.all((lnD > self.lnD[0]) & (lnD < self.lnD[1]), axis=1)
        if K > 1:
            ok &= np.all(np.diff(lnT, axis=1) > 0, axis=1)
        if self.sample_sigma:
            lns = th[:, -1]
            ok &= (lns > self.lns_lo) & (lns < self.lns_hi)
            sigma = np.exp(lns)
        else:
            sigma = self.sigma
        with np.errstate(over="ignore", invalid="ignore"):
            E = physics.exp_decay(self.t[None, None, :], np.exp(lnT)[:, :, None])               # walkers x K x echoes
            G = physics.diffusion_attenuation(self.bs[None, None, :], np.exp(lnD)[:, :, None])  # walkers x K x b-values
            c = ((E @ self.Y) * G).sum(axis=2)                                          # <e_k, Y g_k>
            quad = np.einsum("wk,wl,wkl->w", A, A, np.einsum("wke,wle->wkl", E, E) * np.einsum("wkb,wlb->wkl", G, G))
            rss = self.yy - 2.0 * (A * c).sum(axis=1) + quad
            lp = -0.5 * rss / sigma ** 2 - (self.N * np.log(sigma) if self.sample_sigma else 0.0)
        return np.where(ok & np.isfinite(lp), lp, -np.inf)

    def _init_walkers(self):
        K, base, scale = self.K, self.theta0, 0.02
        k = np.arange(K)[None, :]
        for _ in range(60):
            th = base[None, :] * (1 + scale * self.rng.standard_normal((self.nwalkers, self.ndim)))
            lnT = base[1:3 * K:3][None, :] + scale * self.rng.standard_normal((self.nwalkers, K))
            lnT.sort(axis=1)
            th[:, 1:3 * K:3] = np.clip(lnT, self.lnT[0] + 1e-4 * (k + 1), self.lnT[1] - 1e-4 * (K - k))     # strictly inside the prior, ordered
            lnD = base[2:3 * K:3][None, :] + scale * self.rng.standard_normal((self.nwalkers, K))
            th[:, 2:3 * K:3] = np.clip(lnD, self.lnD[0] + 1e-4, self.lnD[1] - 1e-4)
            th[:, 0:3 * K:3] = np.abs(th[:, 0:3 * K:3]) + 1e-9
            if np.all(np.isfinite(self._lnprob(th))):
                return th
            scale *= 1.4
        raise RuntimeError("Could not start the MCMC walkers in a region of non-zero posterior; the fit may be degenerate.")

    def _new_sampler(self):
        self.sampler = self.emcee.EnsembleSampler(self.nwalkers, self.ndim, self._lnprob, vectorize=True)
        self.sampler.random_state = np.random.RandomState(self.seed).get_state()
        self.state = self.emcee.State(self.p0.copy())
        self.done = 0

    def _time_trial(self):
        t0 = time.perf_counter()
        self.sampler.run_mcmc(self.state, TRIAL_STEPS, progress=False)
        dt = time.perf_counter() - t0
        self._new_sampler()
        return dt / TRIAL_STEPS * self.steps

    def step(self, k):
        k = int(min(k, self.steps - self.done))
        if k > 0:
            self.state = self.sampler.run_mcmc(self.state, k, progress=False)
            self.done += k
        return {"done": self.done, "total": self.steps}

    def _natural(self, th):
        out = th.copy()
        out[..., 1:3 * self.K:3] = np.exp(out[..., 1:3 * self.K:3])
        out[..., 2:3 * self.K:3] = np.exp(out[..., 2:3 * self.K:3])
        if self.sample_sigma:
            out[..., -1] = np.exp(out[..., -1])
        return out

    def result(self, theme=None, max_samples=0):
        if self.done < 10:
            raise RuntimeError("Too few MCMC steps to summarise yet.")
        theme = theme or {}
        K, level = self.K, self.level
        chain = self.sampler.get_chain()                                     # steps x walkers x ndim
        burn = min(self.burn, self.done // 2)
        post = chain[burn:]
        flat = self._natural(post).reshape(-1, self.ndim)
        plain, tex = param_labels(K)
        names = plain + (["sigma"] if self.sample_sigma else [])
        tex_names = tex + ([r"$\sigma$"] if self.sample_sigma else [])
        dn = derived_labels(K)
        dflat = derived(flat[:, :3 * K], K)
        se = _ls_se(self.fit)
        params = [dict(_summ(names[j], flat[:, j], level), ls=float(self.p_ls[j]) if j < 3 * K else None, ls_se=se[j] if j < 3 * K else None) for j in range(self.ndim)]
        derived_rows = [_summ(dn[j], dflat[:, j], level) for j in range(len(dn))]

        tau = self.emcee.autocorr.integrated_time(post, tol=0, quiet=True)
        tau_max = float(np.max(tau))
        acc = float(np.mean(self.sampler.acceptance_fraction))
        warnings = _edge_warnings(self.fit, "posterior")
        if len(post) < 50 * tau_max:
            warnings.append(f"The chain is only {len(post) / tau_max:.0f} autocorrelation times long (50+ recommended): run more steps before trusting the tails of the intervals.")
        if acc < 0.1 or acc > 0.8:
            warnings.append(f"Acceptance fraction {acc:.2f} is unusual (0.2–0.5 is typical); the posterior may be badly conditioned.")
        if self.done < self.steps:
            warnings.append(f"Stopped early: {self.done} of {self.steps} steps.")
        if burn < self.burn:
            warnings.append(f"Burn-in was cut to {burn} steps because only {self.done} have run.")

        wsel = np.linspace(0, self.nwalkers - 1, min(MAX_TRACE_WALKERS, self.nwalkers)).astype(int)
        tstep = max(1, chain.shape[0] // MAX_TRACE_POINTS)
        traces = self._natural(chain[::tstep][:, wsel])
        rng = np.random.default_rng(self.seed + 1)
        sub = flat[rng.choice(len(flat), size=min(MAX_CORNER_SAMPLES, len(flat)), replace=False)].copy()
        truths = np.concatenate([self.p_ls, [self.sd0] if self.sample_sigma else []])
        for arr in (sub, truths[None, :]):                                    # the corner plot is in log10 T2 and log10 D, which span decades
            arr[:, 1:3 * K:3] = np.log10(arr[:, 1:3 * K:3])
            arr[:, 2:3 * K:3] = np.log10(arr[:, 2:3 * K:3])
        png = figures.corner_png(sub, tex_names, truths, fg=theme.get("fg", "#1a1a1a"), grid=theme.get("grid", "#999999"))
        return {
            "kind": "mcmc_t2d", "level": level, "K": K, "sample_sigma": self.sample_sigma, "sigma": self.sigma, "nwalkers": self.nwalkers,
            "steps_run": self.done, "steps_planned": self.steps, "burn": burn, "params": params, "derived": derived_rows,
            "diagnostics": {"acceptance": acc, "tau": [float(x) for x in tau], "tau_max": tau_max,
                            "n_eff": float(self.nwalkers * len(post) / max(tau_max, 1.0)), "steps_per_tau": len(post) / tau_max},
            "warnings": warnings,
            "trace": {"names": names, "step": (np.arange(chain.shape[0])[::tstep]).tolist(), "values": traces, "burn": burn},
            "scatter": _scatter(flat, K, rng), "corner_png_b64": png,
            "samples": {"names": names, "values": flat[rng.choice(len(flat), size=min(max_samples, len(flat)), replace=False)]} if max_samples else None,
        }


class Bootstrap2D:
    def __init__(self, t, b, Y, fit, t2_range, d_range, noise=None, sigma=None, method=None, replicates=200, ci_level=0.95, seed=0):
        self.t, self.b, self.Y = np.asarray(t, float), np.asarray(b, float), np.asarray(Y, float)
        self.fit, self.K = fit, int(fit["K"])
        self.noise = None if noise is None else (lambda n: (n - n.mean()).ravel())(np.asarray(noise, float))
        self.sigma = sigma
        self.method = method or ("noise" if self.noise is not None else "residual")
        if self.method not in BOOT_METHODS:
            raise ValueError(f"Unknown bootstrap method {self.method!r}")
        if self.method == "noise" and self.noise is None:
            raise ValueError("The noise bootstrap needs the imaginary channel (this data is real-valued); use residual resampling.")
        if self.method == "parametric" and not sigma:
            raise ValueError("The parametric bootstrap needs a noise estimate from the imaginary channel.")
        self.t2_range, self.d_range = t2_range, d_range
        self.B, self.level, self.done, self.failures = int(replicates), float(ci_level), 0, 0
        self.rng = np.random.default_rng(seed)
        self.p_ls = _flat(fit)
        pops = fit["populations"]
        self.base = M.predict(self.t, self.b, [p["A"] for p in pops], [p["T2"] for p in pops], [p["D"] for p in pops])
        self.regions = [{"T2_mean": p["T2"], "D_mean": p["D"]} for p in pops]        # start every refit from the least-squares solution
        self.rows = []
        t0 = time.perf_counter()
        self.step(3)                                                             # timing trial; these replicates are kept
        self.est_seconds = (time.perf_counter() - t0) / 3 * self.B

    def _noise(self):
        shape = self.Y.shape
        if self.method == "noise":
            return self.rng.choice(self.noise, size=self.Y.size).reshape(shape)
        if self.method == "parametric":
            return self.rng.normal(0.0, self.sigma, size=shape)
        r = (self.Y - self.base).ravel()
        r = (r - r.mean()) * np.sqrt(self.Y.size / max(self.Y.size - self.fit["n_params"], 1))
        return self.rng.choice(r, size=self.Y.size).reshape(shape)

    def step(self, k):
        k = int(min(k, self.B - self.done))
        for _ in range(k):
            Ys = self.base + self._noise()
            try:
                A, lnT, lnD, _ = M.fit_params(self.t, self.b, Ys, self.K, self.regions, self.t2_range, self.d_range, self.sigma, n_starts=1)
                self.rows.append([v for j in range(self.K) for v in (A[j], np.exp(lnT[j]), np.exp(lnD[j]))])
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                self.failures += 1
            self.done += 1
        return {"done": self.done, "total": self.B}

    def result(self):
        if len(self.rows) < 5:
            raise RuntimeError("Too few bootstrap replicates to summarise yet.")
        K, level = self.K, self.level
        P = np.array(self.rows)
        plain, _ = param_labels(K)
        dn = derived_labels(K)
        dflat = derived(P, K)
        warnings = _edge_warnings(self.fit, "bootstrap")
        if self.failures > 0.05 * self.done:
            warnings.append(f"{self.failures} of {self.done} replicates failed to fit; intervals may be biased.")
        if self.done < self.B:
            warnings.append(f"Stopped early: {self.done} of {self.B} replicates.")
        if self.method == "residual":
            warnings.append("Residual resampling reuses the fit's own residuals, so it inherits any model misfit.")
        return {"kind": "bootstrap_t2d", "level": level, "method": self.method, "replicates": self.done, "planned": self.B, "failures": self.failures, "K": K,
                "n_ok": len(P), "params": [dict(_summ(plain[j], P[:, j], level), ls=float(self.p_ls[j]), ls_se=_ls_se(self.fit)[j]) for j in range(P.shape[1])],
                "derived": [_summ(dn[j], dflat[:, j], level) for j in range(len(dn))],
                "samples": {"names": plain, "values": P}, "scatter": _scatter(P, K, np.random.default_rng(1)), "warnings": warnings}
