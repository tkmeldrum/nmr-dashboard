# test_data

Synthetic sample datasets for development, the tests and the page's "Explore test data" button. **No real measurements are kept in this repository** (it is public).

Kea datasets are folders; Tecmag datasets are single `.tnt` files (stored conjugated, as `nmrdash.tecmag` reads them). Everything here is made by `tools/make_synthetic_data.py` (seeded, so it is reproducible: run it to regenerate; do not edit the files by hand). Each kind comes as a
noise-free copy and a noisy copy; the values the data were made from are in `TRUTH` in that script and the tests check that the analyses recover them.

| Folder | Sequence | What is in it | True values |
|---|---|---|---|
| `kea_cpmg_clean`, `kea_cpmg_noisy` | CPMG | 256 echoes × 64 points, echo time 200 µs, ¹H slab: a short-T₂ and a long-T₂ layer overlapping around z = 0 (so the echo sum sees both, and the ends of the profile one each) | T₂ 5 ms (40%) and 20 ms (60%); layers −120…30 µm and −30…120 µm; noisy: σ = 12 per point per channel |
| `kea_t1_clean`, `kea_t1_noisy` | T1Sat | 20 delays, 10 ms – 5 s, `data.dat` (delay, amplitude) | T₁ 0.8 s, offset 3% of the plateau; noisy: σ = 60 on 20 000 |
| `kea_t2d_clean`, `kea_t2d_noisy` | SSE-T2 | 600 echoes (100 µs) × 25 diffusion delays (τ 0.178 – 3.2 ms, Δ 5 ms, G = 23.67 T/m); `data.dat` carries a spectrometer-style b table for 23.49 T/m (gradient = 1000 kHz/mm) | T₂ 4 ms / D 1.5e-10 m²/s (40%) and T₂ 20 ms / D 2e-11 m²/s (60%); noisy: σ = 1 on amplitude 300 |
| `tecmag_cpmg_clean.tnt`, `tecmag_cpmg_noisy.tnt` | CPMG | 256 echoes × 32 points (dwell 2 µs, echo time 150 µs), the same two layers as the Kea set at G = 6.7 T/m (layers −424…106 µm and −106…424 µm) | T₂ 5 ms (40%) and 20 ms (60%); noisy: σ = 12 per point per channel |
| `tecmag_t1_clean.tnt`, `tecmag_t1_noisy.tnt` | T1 saturation recovery | 16 delays (10 ms – 5 s, a table in µs), each a 16-echo train | T₁ 0.8 s, offset 3% of the plateau; echo train T₂ 20 ms; noisy: σ = 3 per point on amplitude 300 |
| `tecmag_t2d_clean.tnt`, `tecmag_t2d_noisy.tnt` | stimulated-echo T₂–D | 16 delays (τ table 40 – 400 µs, linear), one continuous 300-echo record each (echo time 60 µs, window 40 µs, dwell 2 µs); Δ = 10 ms, G = 6.7 T/m | T₂ 3 ms / D 1.5e-9 m²/s (40%) and T₂ 12 ms / D 3e-10 m²/s (60%); noisy: σ = 0.25 per point |

Noise-free sets are almost exactly noise-free (float32 rounding only), so statistics that compare a fit with the noise level (χ²) are only meaningful on the noisy sets.

Tests that need a variation (a population too slow to resolve, a single exponential) generate it in a temporary folder with the `**override` arguments of the writers
(`tests/sample_data.py`).
