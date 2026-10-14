# Third-party software

nmr-dashboard is released under the Apache License 2.0 (see `LICENSE`). It is used together with the following software, each
under its own licence. Versions are the ones the site is tested with.

| Software | Version | Licence | Used for |
|---|---|---|---|
| [Pyodide](https://pyodide.org) (with CPython) | 0.27.7 | MPL-2.0 (CPython: PSF-2.0) | Runs the Python analysis inside the browser |
| [NumPy](https://numpy.org) | 2.0.2 | BSD-3-Clause | Arrays and linear algebra |
| [SciPy](https://scipy.org) | 1.14.1 | BSD-3-Clause | Fitting, optimisation, signal processing |
| [Matplotlib](https://matplotlib.org) | 3.8.4 | Matplotlib licence (PSF-based, BSD-compatible) | Saved figures |
| [emcee](https://emcee.readthedocs.io) | 3.1.6 | MIT | MCMC sampling |
| [Plotly.js](https://plotly.com/javascript/) | 2.35.2 | MIT | Interactive plots on the page |
| Pillow, kiwisolver, fonttools, contourpy, cycler, pyparsing, python-dateutil, packaging, six, OpenBLAS, micropip and others | as shipped with Pyodide 0.27.7 | BSD, MIT, Apache-2.0, HPND or MPL-2.0 | Dependencies of the above |

Each package carries its own licence text inside its distribution. Where the site serves a package unchanged (for example from
Pyodide's package set), that notice travels with it.

## Where methods come from

- Regularised inverse Laplace transform (1D and the 2D tensor-product form): L. Venkataramanan, Y.-Q. Song and M. D. Hürlimann,
  *IEEE Trans. Signal Process.* **50**, 1017 (2002).
- Affine-invariant ensemble sampler (emcee): D. Foreman-Mackey, D. W. Hogg, D. Lang and J. Goodman, *Publ. Astron. Soc. Pac.* **125**, 306 (2013).
- NumPy: C. R. Harris et al., *Nature* **585**, 357 (2020). SciPy: P. Virtanen et al., *Nature Methods* **17**, 261 (2020).
  Matplotlib: J. D. Hunter, *Comput. Sci. Eng.* **9**, 90 (2007).
