"""MCMC and bootstrap uncertainties for the multi-exponential fit and the ILT.

Both are written as small chunked objects (`step(k)` then `result()`), so a caller can show progress between
chunks, stop early and still get results from what has run so far.

MCMC (emcee, affine-invariant ensemble sampler)
    y(t) = sum_i A_i exp(-t/T_i) [+ c]  with Gaussian noise of SD sigma.
    Sampled parameters: A_i (flat prior, A >= 0), ln T_i (uniform = log-uniform T within the fit's bounds,
    ordered T_1 < T_2 < ... so components cannot swap labels), c (flat), and optionally ln sigma (uniform).
    Fixed sigma comes from the imaginary channel (see echoes.noise_sigma).

Bootstrap
    Surrogate decays = fitted curve + resampled noise, each refitted (multi-exponential, warm-started from the
    least-squares solution) and/or re-run through the ILT.
    noise source: "noise"      resample the measured imaginary-channel values (model-free; preferred)
                  "residual"   resample the fit's residuals (rescaled for degrees of freedom)
                  "parametric" Gaussian with the imaginary-channel sigma
"""
import time

import numpy as np

from . import figures, ilt_tools, multiexp
from .ilt import run_1dilt

BOOT_METHODS = ("noise", "residual", "parametric")
TRIAL_STEPS = 20
MAX_TRACE_WALKERS, MAX_TRACE_POINTS, PREDICTIVE_DRAWS, MAX_CORNER_SAMPLES = 8, 300, 300, 6000


# ── shared helpers ──────────────────────────────────────────────────────────

def params_from_fit(f):
    """Flat [A1, T1, A2, T2, ..., (c)] from a multiexp.fit result (components are sorted fast -> slow)."""
    p = []
    for c in f["components"]:
        p += [c["A"], c["T2"]]
    if f["offset"]:
        p.append(f["offset_value"])
    return np.array(p, float)


def param_labels(n, offset, symbol="T2"):
    """(plain names, TeX names) for [A1, T2_1, A2, T2_2, ..., (c)] (symbol "T1" for recovery data)."""
    one = n == 1
    d = symbol[1:]                                      # "2" or "1"
    plain, tex = [], []
    for i in range(1, n + 1):
        plain += ["A" if one else f"A{i}", symbol if one else f"{symbol},{i}"]
        tex += [r"$A$" if one else rf"$A_{i}$", rf"$T_{d}$ (s)" if one else rf"$T_{{{d},{i}}}$ (s)"]
    if offset:
        plain.append("offset")
        tex.append(r"$c$")
    return plain, tex


def derived(p, n):
    """[total amplitude, amplitude-weighted mean T2, fractions...] for a flat parameter vector (or rows of them)."""
    p = np.atleast_2d(p)
    A, T = p[:, 0:2 * n:2], p[:, 1:2 * n:2]
    tot = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (A * T).sum(axis=1) / tot
        frac = A / tot[:, None]
    return np.column_stack([tot, mean, frac])


def derived_labels(n, symbol="T2"):
    """Derived quantities worth reporting; for one component they would only repeat A and T2."""
    return [] if n == 1 else ["total amplitude", f"mean {symbol}"] + [f"fraction {i}" for i in range(1, n + 1)]


def _interval(x, level):
    lo, med, hi = np.nanpercentile(x, [50 * (1 - level), 50, 50 * (1 + level)], axis=0)
    return lo, med, hi


def _summ(name, x, level):
    lo, med, hi = _interval(x, level)
    return {"name": name, "median": float(med), "lo": float(lo), "hi": float(hi),
            "mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)) if len(x) > 1 else None}


# ── MCMC ────────────────────────────────────────────────────────────────────

class Mcmc:
    def __init__(self, t, y, n, offset, ls_fit, sigma=None, ci_level=0.95, nwalkers=32, steps=3000, burn=1000, seed=0):
        try:
            import emcee
        except ImportError as exc:
            raise RuntimeError("MCMC needs the 'emcee' package (pip install emcee).") from exc
        self.emcee = emcee
        self.t, self.y = np.asarray(t, float), np.asarray(y, float)
        self.n, self.offset, self.level = int(n), bool(offset), float(ci_level)
        self.sample_sigma = sigma is None
        self.sigma = None if sigma is None else float(sigma)
        self.steps, self.burn, self.seed, self.done = int(steps), int(burn), int(seed), 0
        self.ls = ls_fit
        self.kind = ls_fit.get("kind", "decay")
        self.symbol = multiexp.SYMBOL[self.kind]
        self.p_ls = params_from_fit(ls_fit)
        self.ndim = 2 * self.n + int(self.offset) + int(self.sample_sigma)
        self.nwalkers = max(int(nwalkers), 2 * self.ndim, 8)
        self.nwalkers += self.nwalkers % 2
        self.lnlo, self.lnhi = np.log(self.t[0] / 10.0), np.log(self.t[-1] * 10.0)
        rss0 = float(np.sum((self.y - multiexp.model(self.t, self.p_ls, self.n, self.offset, self.kind)) ** 2))
        self.sd0 = np.sqrt(rss0 / max(len(self.t) - len(self.p_ls), 1))
        self.lns_lo, self.lns_hi = np.log(self.sd0 / 20.0), np.log(self.sd0 * 20.0)
        self.rng = np.random.default_rng(self.seed)
        self.p0 = self._init_walkers()
        self._new_sampler()
        self.est_seconds = self._time_trial()

    # θ = [A1, lnT1, ..., An, lnTn, (c), (ln sigma)]
    def _theta_from_params(self, p):
        th = []
        for i in range(self.n):
            th += [p[2 * i], np.log(p[2 * i + 1])]
        if self.offset:
            th.append(p[2 * self.n])
        if self.sample_sigma:
            th.append(np.log(self.sd0))
        return np.array(th, float)

    def _lnprob(self, th):
        th = np.atleast_2d(th)
        n = self.n
        A, lnT = th[:, 0:2 * n:2], th[:, 1:2 * n:2]
        ok = np.all(A >= 0, axis=1) & np.all((lnT > self.lnlo) & (lnT < self.lnhi), axis=1)
        if n > 1:
            ok &= np.all(np.diff(lnT, axis=1) > 0, axis=1)
        k = 2 * n
        c = 0.0
        if self.offset:
            c, k = th[:, k][:, None], k + 1
        if self.sample_sigma:
            lns = th[:, k]
            ok &= (lns > self.lns_lo) & (lns < self.lns_hi)
            sigma = np.exp(lns)
        else:
            sigma = self.sigma
        with np.errstate(over="ignore", invalid="ignore"):
            decay = multiexp.KERNELS[self.kind](self.t[None, None, :] / np.exp(lnT)[:, :, None])
            model = np.einsum("wi,wit->wt", A, decay) + c
            rss = np.sum((self.y[None, :] - model) ** 2, axis=1)
            lp = -0.5 * rss / sigma ** 2 - (len(self.t) * np.log(sigma) if self.sample_sigma else 0.0)
        return np.where(ok & np.isfinite(lp), lp, -np.inf)

    def _init_walkers(self):
        base = self._theta_from_params(self.p_ls)
        scale = 0.02
        for _ in range(60):
            th = base[None, :] * (1 + scale * self.rng.standard_normal((self.nwalkers, self.ndim)))
            lnT = th[:, 1:2 * self.n:2]
            lnT[:] = np.clip(base[1:2 * self.n:2][None, :] + scale * self.rng.standard_normal(lnT.shape), self.lnlo + 1e-6, self.lnhi - 1e-6)
            lnT.sort(axis=1)
            if self.n > 1:                               # strictly ordered, even if the least-squares T2s coincide
                lnT += 1e-4 * np.arange(self.n)[None, :]
            th[:, 0:2 * self.n:2] = np.abs(th[:, 0:2 * self.n:2]) + 1e-9
            if np.all(np.isfinite(self._lnprob(th))):
                return th
            scale *= 1.4
        raise RuntimeError("Could not start MCMC walkers in a region of non-zero posterior; the fit may be degenerate.")

    def _new_sampler(self):
        self.sampler = self.emcee.EnsembleSampler(self.nwalkers, self.ndim, self._lnprob, vectorize=True)
        self.sampler.random_state = np.random.RandomState(self.seed).get_state()
        self.state = self.emcee.State(self.p0.copy())
        self.done = 0

    def _time_trial(self):
        t0 = time.perf_counter()
        self.sampler.run_mcmc(self.state, TRIAL_STEPS, progress=False)
        dt = time.perf_counter() - t0
        self._new_sampler()                              # discard the trial so the real run is reproducible
        return dt / TRIAL_STEPS * self.steps

    def step(self, k):
        k = int(min(k, self.steps - self.done))
        if k > 0:
            self.state = self.sampler.run_mcmc(self.state, k, progress=False)
            self.done += k
        return {"done": self.done, "total": self.steps}

    def _natural(self, th):
        out = th.copy()
        out[..., 1:2 * self.n:2] = np.exp(out[..., 1:2 * self.n:2])
        if self.sample_sigma:
            out[..., -1] = np.exp(out[..., -1])
        return out

    def result(self, theme=None, max_samples=0):
        if self.done < 10:
            raise RuntimeError("Too few MCMC steps to summarise yet.")
        theme = theme or {}
        n, level = self.n, self.level
        chain = self.sampler.get_chain()                                     # (steps, walkers, ndim)
        burn = min(self.burn, self.done // 2)
        post = chain[burn:]
        nat_post = self._natural(post)
        flat = nat_post.reshape(-1, self.ndim)
        plain, tex = param_labels(n, self.offset, self.symbol)
        core = len(plain)
        names = plain + (["sigma"] if self.sample_sigma else [])
        tex_names = tex + ([r"$\sigma$"] if self.sample_sigma else [])
        dflat = derived(flat[:, :2 * n], n)
        dn = derived_labels(n, self.symbol)

        params = [dict(_summ(names[j], flat[:, j], level), ls=float(self.p_ls[j]) if j < core else None,
                       ls_se=self._ls_se(j) if j < core else None) for j in range(self.ndim)]
        derived_rows = [_summ(dn[j], dflat[:, j], level) for j in range(len(dn))]

        tau = self.emcee.autocorr.integrated_time(post, tol=0, quiet=True)
        tau_max = float(np.max(tau))
        acc = float(np.mean(self.sampler.acceptance_fraction))
        n_eff = float(self.nwalkers * len(post) / max(tau_max, 1.0))
        warnings = []
        if len(post) < 50 * tau_max:
            warnings.append(f"The chain is only {len(post) / tau_max:.0f} autocorrelation times long (50+ recommended): "
                            "run more steps before trusting the tails of the intervals.")
        if acc < 0.1 or acc > 0.8:
            warnings.append(f"Acceptance fraction {acc:.2f} is unusual (0.2–0.5 is typical); the posterior may be badly conditioned.")
        if self.done < self.steps:
            warnings.append(f"Stopped early: {self.done} of {self.steps} steps.")
        if burn < self.burn:
            warnings.append(f"Burn-in was cut to {burn} steps because only {self.done} have run.")

        # trace plot data: a few walkers, thinned
        wsel = np.linspace(0, self.nwalkers - 1, min(MAX_TRACE_WALKERS, self.nwalkers)).astype(int)
        tstep = max(1, chain.shape[0] // MAX_TRACE_POINTS)
        traces = self._natural(chain[::tstep][:, wsel])                       # (points, walkers, ndim)

        # posterior-predictive band on a dense time axis
        rng = np.random.default_rng(self.seed + 1)
        draws = flat[rng.choice(len(flat), size=min(PREDICTIVE_DRAWS, len(flat)), replace=False)]
        t_dense = np.linspace(0, self.t[-1], 200)
        curves = np.array([multiexp.model(t_dense, d[:2 * n + int(self.offset)], n, self.offset, self.kind) for d in draws])
        plo, pmed, phi = _interval(curves, level)

        sub = flat[rng.choice(len(flat), size=min(MAX_CORNER_SAMPLES, len(flat)), replace=False)]
        truths = np.concatenate([self.p_ls, [self.sd0] if self.sample_sigma else []])
        png = figures.corner_png(sub, tex_names, truths, fg=theme.get("fg", "#1a1a1a"), grid=theme.get("grid", "#999999"))
        return {
            "kind": "mcmc", "symbol": self.symbol, "level": level, "n": n, "offset": self.offset, "sample_sigma": self.sample_sigma,
            "sigma": self.sigma, "nwalkers": self.nwalkers, "steps_run": self.done, "steps_planned": self.steps, "burn": burn,
            "params": params, "derived": derived_rows,
            "diagnostics": {"acceptance": acc, "tau": [float(x) for x in tau], "tau_max": tau_max,
                            "n_eff": n_eff, "steps_per_tau": len(post) / tau_max},
            "warnings": warnings,
            "trace": {"names": names, "step": (np.arange(chain.shape[0])[::tstep]).tolist(), "values": traces, "burn": burn},
            "predictive": {"t": t_dense, "lo": plo, "median": pmed, "hi": phi},
            "corner_png_b64": png,
            "samples": {"names": names, "values": flat[rng.choice(len(flat), size=min(max_samples, len(flat)), replace=False)]} if max_samples else None,
        }

    def _ls_se(self, j):
        n = self.n
        if j < 2 * n:
            c = self.ls["components"][j // 2]
            return c["T2_se"] if j % 2 else c["A_se"]
        return self.ls["offset_se"]


# ── bootstrap ───────────────────────────────────────────────────────────────

class Bootstrap:
    def __init__(self, t, y, noise_series=None, sigma=None, me=None, ls_fit=None, ilt=None,
                 method=None, replicates=200, ci_level=0.95, seed=0, peak_min_fraction=0.05):
        self.t, self.y = np.asarray(t, float), np.asarray(y, float)
        self.N = len(self.t)
        self.noise = None if noise_series is None else np.asarray(noise_series, float) - np.mean(noise_series)
        self.sigma = sigma
        self.method = method or ("noise" if self.noise is not None else "residual")
        if self.method not in BOOT_METHODS:
            raise ValueError(f"Unknown bootstrap method {self.method!r}")
        if self.method == "noise" and self.noise is None:
            raise ValueError("The noise bootstrap needs the imaginary channel (this data is real-valued); use residual resampling.")
        if self.method == "parametric" and not sigma:
            raise ValueError("The parametric bootstrap needs a noise estimate from the imaginary channel.")
        self.B, self.level, self.done, self.failures = int(replicates), float(ci_level), 0, 0
        self.rng = np.random.default_rng(seed)
        self.me, self.ilt_cfg = me, ilt
        self.peak_min_fraction = peak_min_fraction

        self.ls = ls_fit
        if me:
            self.n, self.offset = int(me["n"]), bool(me.get("offset", True))
            self.fit_kind = ls_fit.get("kind", "decay")
            self.symbol = multiexp.SYMBOL[self.fit_kind]
            self.p_ls = params_from_fit(ls_fit)
            self.me_base = np.asarray(ls_fit["fit"], float)
            self.me_rows, self.me_derived = [], []
        if ilt:
            self.ilt_params = {"ILTminmax": [float(ilt["t2_min"]), float(ilt["t2_max"])],
                               "nrILTSteps": min(int(ilt["n_steps"]), ilt_tools.STEPS_LIMIT)}
            spec, tau, pred, _, _ = run_1dilt(self.y, self.t, self.ilt_params, alpha=float(ilt["alpha"]))
            self.tau, self.spec0, self.ilt_base = tau, spec, pred
            self.bands = ilt_tools.band_edges(tau, spec, peak_min_fraction)
            self.il_specs, self.il_band_mean, self.il_band_frac = [], [], []
        t0 = time.perf_counter()
        self.step(3)                                                     # timing trial; these replicates are kept
        self.est_seconds = (time.perf_counter() - t0) / 3 * self.B

    def _noise(self, base, k):
        if self.method == "noise":
            return self.rng.choice(self.noise, size=self.N, replace=True)
        if self.method == "parametric":
            return self.rng.normal(0.0, self.sigma, size=self.N)
        r = self.y - base
        r = (r - r.mean()) * np.sqrt(self.N / max(self.N - k, 1))
        return self.rng.choice(r, size=self.N, replace=True)

    def step(self, k):
        k = int(min(k, self.B - self.done))
        for _ in range(k):
            ok = True
            if self.me:
                ys = self.me_base + self._noise(self.me_base, len(self.p_ls))
                try:
                    f = multiexp.fit(self.t, ys, self.n, self.offset, n_starts=3, p0=self.p_ls, kind=self.fit_kind)
                    p = params_from_fit(f)
                    self.me_rows.append(p)
                    self.me_derived.append(derived(p[:2 * self.n], self.n)[0])
                except (RuntimeError, ValueError):
                    ok = False
            if self.ilt_cfg:
                yi = self.ilt_base + self._noise(self.ilt_base, 0)
                try:
                    spec = run_1dilt(yi, self.t, self.ilt_params, alpha=float(self.ilt_cfg["alpha"]))[0]
                    self.il_specs.append(spec)
                    means, fracs = [], []
                    for _pk, lo, hi in self.bands:
                        seg = spec[lo:hi]
                        fracs.append(seg.sum() / spec.sum() if spec.sum() > 0 else np.nan)
                        means.append(float(np.exp(np.sum(seg * np.log(self.tau[lo:hi])) / seg.sum())) if seg.sum() > 0 else np.nan)
                    self.il_band_mean.append(means)
                    self.il_band_frac.append(fracs)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    ok = False
            self.done += 1
            self.failures += 0 if ok else 1
        return {"done": self.done, "total": self.B}

    def result(self):
        if self.done < 5:
            raise RuntimeError("Too few bootstrap replicates to summarise yet.")
        level = self.level
        out = {"kind": "bootstrap", "level": level, "method": self.method, "replicates": self.done,
               "planned": self.B, "failures": self.failures, "multiexp": None, "ilt": None, "warnings": []}
        if self.failures > 0.05 * self.done:
            out["warnings"].append(f"{self.failures} of {self.done} replicates failed to fit; intervals may be biased.")
        if self.done < self.B:
            out["warnings"].append(f"Stopped early: {self.done} of {self.B} replicates.")
        if self.method == "residual":
            out["warnings"].append("Residual resampling reuses the fit's own residuals, so it inherits any model misfit.")
        if self.me and len(self.me_rows) >= 5:
            P = np.array(self.me_rows)
            D = np.array(self.me_derived)
            plain, _ = param_labels(self.n, self.offset, self.symbol)
            dn = derived_labels(self.n, self.symbol)
            out["multiexp"] = {
                "n": self.n, "offset": self.offset, "n_ok": len(P),
                "params": [dict(_summ(plain[j], P[:, j], level), ls=float(self.p_ls[j])) for j in range(P.shape[1])],
                "derived": [_summ(dn[j], D[:, j], level) for j in range(len(dn))],
                "samples": {"names": plain, "values": P},
            }
        if self.ilt_cfg and len(self.il_specs) >= 5:
            S = np.array(self.il_specs)
            lo, med, hi = _interval(S, level)
            bands = []
            M, F = np.array(self.il_band_mean), np.array(self.il_band_frac)
            for b, (pk, a, z) in enumerate(self.bands):
                seg = self.spec0[a:z]
                bands.append({"t2_peak": float(self.tau[pk]),
                              "base_mean_T2": float(np.exp(np.sum(seg * np.log(self.tau[a:z])) / seg.sum())) if seg.sum() > 0 else None,
                              "base_fraction": float(seg.sum() / self.spec0.sum()),
                              "T2": _summ("T2", M[:, b], level), "fraction": _summ("fraction", F[:, b], level)})
            out["ilt"] = {"tau": self.tau, "spectrum": self.spec0, "lo": lo, "median": med, "hi": hi,
                          "bands": bands, "n_ok": len(S)}
        return out
