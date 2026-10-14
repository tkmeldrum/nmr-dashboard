# References to check

These references were written from memory when the models were documented and have **not been checked**. For each row: is this the right source for the relation, and are the journal, volume, page and year right? Correct anything that is wrong in `docs/references.json` (or tell me), set `"verified": true`, and run `python tools/make_references.py`. **Only verified references are shown on the About page**; until then the page shows the equation, the description and the code link, without a reference.

Verified so far: 0 of 20.

| Verified | Relation or software | Code | Draft reference |
|---|---|---|---|
| no | **Exponential T₂ decay** (`exp_decay`) | physics.py: `exp_decay`, `decay_factor` | Carr and Purcell, Phys. Rev. 94, 630 (1954); Meiboom and Gill, Rev. Sci. Instrum. 29, 688 (1958) |
| no | **Multi-exponential decay** (`multi_exp`) | physics.py: `multi_exp` | Whittall and MacKay, J. Magn. Reson. 84, 134 (1989) |
| no | **Saturation recovery (T₁)** (`saturation`) | physics.py: `saturation_recovery`, `multi_saturation`, `recovery_factor` | Fukushima and Roeder, Experimental Pulse NMR: A Nuts and Bolts Approach (Addison-Wesley, 1981) |
| no | **Diffusion attenuation** (`diffusion`) | physics.py: `diffusion_attenuation` | Stejskal and Tanner, J. Chem. Phys. 42, 288 (1965); Tanner, J. Chem. Phys. 52, 2523 (1970) |
| no | **Diffusion weighting b of the stimulated echo** (`b_value`) | physics.py: `b_value_stimulated_echo`, `geometric_grid`, `gradient_implied_by_kea` | Hahn, Phys. Rev. 80, 580 (1950); Tanner, J. Chem. Phys. 52, 2523 (1970); Callaghan, Principles of Nuclear Magnetic Resonance Microscopy (Oxford, 1991) |
| no | **T₂–D signal** (`t2d`) | physics.py: `t2d_signal`, `t2d_factors` | Combines Tanner (1970) and Carr and Purcell (1954); Hürlimann and Venkataramanan, J. Magn. Reson. 157, 31 (2002) |
| no | **Inversion kernels** (`kernels`) | physics.py: `t2_kernel`, `diffusion_kernel` | Venkataramanan, Song and Hürlimann, IEEE Trans. Signal Process. 50, 1017 (2002); Song et al., J. Magn. Reson. 154, 261 (2002) |
| no | **Range of D the b-values can resolve** (`resolvable`) | physics.py: `resolvable_diffusion_window` | Rule of thumb (exp(−5) < 1 %; exp(−1) = 37 %), not a sharp limit |
| no | **Position along the gradient** (`position`) | physics.py: `z_um_from_frequency`, `native_resolution_um` | Frequency encoding in a static gradient (Callaghan, 1991) |
| no | **Phasing and integrating the echoes** (`phasing`) | inference.py: `phase_and_integrate` | Standard zero-order phase correction |
| no | **Noise level from the imaginary channel** (`noise`) | inference.py: `noise_sigma_mad` | Hampel (1974); Rousseeuw and Croux, J. Am. Stat. Assoc. 88, 1273 (1993) |
| no | **Fit quality** (`fit_quality`) | inference.py: `sum_squared_residuals`, `reduced_chi2`, `chi2_per_point` | Standard least squares |
| no | **Choosing the number of components** (`criteria`) | inference.py: `information_criteria` | Akaike, IEEE Trans. Autom. Control 19, 716 (1974); Schwarz, Ann. Statist. 6, 461 (1978) |
| no | **Confidence intervals from least squares** (`interval`) | inference.py: `student_t_critical` | Bates and Watts, Nonlinear Regression Analysis and Its Applications (Wiley, 1988) |
| no | **Regularisation of the inversions** (`regularisation`) | inference.py: `second_derivative_matrix`, `tikhonov_weighting` | Venkataramanan, Song and Hürlimann, IEEE Trans. Signal Process. 50, 1017 (2002); matches the Matlab upnnlsmooth1D / nnlsmooth1Dsvd codes |
| no | **The regularised inversion (1D and 2D) as implemented here** (`ilt_method`) | ilt.py, ilt2d.py: `run_1dilt`, `run_2dilt` | Venkataramanan, Song and Hürlimann, IEEE Trans. Signal Process. 50, 1017 (2002). (The code is the author's own, iterated from the Matlab codes to Python; this is the method it follows.) |
| no | **MCMC sampler** (`emcee`) | uncertainty.py, t2d_uncertainty.py: `Mcmc`, `Mcmc2D` | Foreman-Mackey, Hogg, Lang and Goodman, Publ. Astron. Soc. Pac. 125, 306 (2013) (emcee) |
| no | **NumPy** (`numpy`) | (software): – | Harris et al., Nature 585, 357 (2020) |
| no | **SciPy** (`scipy`) | (software): – | Virtanen et al., Nature Methods 17, 261 (2020) |
| no | **Matplotlib** (`matplotlib`) | (software): – | Hunter, Comput. Sci. Eng. 9, 90 (2007) |
