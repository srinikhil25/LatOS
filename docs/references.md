# References

Citations accumulated across all stages, keyed for use in
`docs/stages/*.md` and ready to lift into a BibTeX file at thesis time.

Per-entry shape:
- `<key>` — first author / year identifier
- Full reference (author, title, journal/publisher, year, DOI when
  available)
- Where Latos uses it

---

## Materials science methods

### `tauc1966`
Tauc, J., Grigorovici, R., & Vancu, A. (1966). *Optical properties and
electronic structure of amorphous germanium*. Physica Status Solidi (B),
**15**(2), 627–637. DOI: [10.1002/pssb.19660150224](https://doi.org/10.1002/pssb.19660150224)

Original paper introducing the (αhν)^n vs. hν plot for extracting the
optical band gap of amorphous semiconductors. Latos's `UvDrsTaucAnalyzer`
(Stage 3B) implements this method.

### `davis1970`
Davis, E. A., & Mott, N. F. (1970). *Conduction in non-crystalline
systems V. Conductivity, optical absorption and photoconductivity in
amorphous semiconductors*. Philosophical Magazine, **22**(179),
0903–0922. DOI: [10.1080/14786437008221061](https://doi.org/10.1080/14786437008221061)

Refines the Tauc analysis to direct (n=2) and indirect (n=1/2)
transitions. The exponent choice in `UvDrsTaucAnalyzer.default_params`
is from this work.

### `kubelka1931`
Kubelka, P., & Munk, F. (1931). *Ein Beitrag zur Optik der
Farbanstriche*. Zeitschrift für Technische Physik, **12**, 593–601.

The original two-flux theory of diffuse reflectance, source of the
Kubelka-Munk function `F(R) = (1-R)²/(2R)` used to map UV-DRS
reflectance to an absorption proxy.

### `goldsmid2010`
Goldsmid, H. J. (2010). *Introduction to Thermoelectricity*. Springer
Series in Materials Science 121. Springer.
DOI: [10.1007/978-3-642-00716-3](https://doi.org/10.1007/978-3-642-00716-3)

Standard textbook on thermoelectric figure of merit zT = S²σT/κ.
Background for the `thermoelectric_xlsx` parser and the future zT
analyzer (Stage 3D candidate).

### `cullity2001`
Cullity, B. D., & Stock, S. R. (2001). *Elements of X-ray
Diffraction* (3rd ed.). Prentice Hall.

Standard XRD reference covering Bragg's law, peak indexing, and
Rietveld refinement. Background for current XRD parsers and the
future XRD peak-fit analyzer.

### `ryan1988`
Ryan, C. G., Clayton, E., Griffin, W. L., Sie, S. H., & Cousens, D. R.
(1988). *SNIP, a statistics-sensitive background treatment for the
quantitative analysis of PIXE spectra in geoscience applications*.
Nuclear Instruments and Methods in Physics Research Section B,
**34**(3), 396–402. DOI: [10.1016/0168-583X(88)90063-8](https://doi.org/10.1016/0168-583X(88)90063-8)

Original Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP)
baseline algorithm. The de-facto standard for XRF / XRD background
estimation. Used in Latos via `pybaselines.smooth.snip` (Stage 3D).

### `savitzky1964`
Savitzky, A., & Golay, M. J. E. (1964). *Smoothing and differentiation
of data by simplified least squares procedures*. Analytical Chemistry,
**36**(8), 1627–1639. DOI: [10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047)

Polynomial-window low-pass filter used for derivative-free smoothing of
the corrected XRD trace before peak detection (Stage 3D). Preserves
peak height and width far better than a simple moving average.

### `thompson1987`
Thompson, P., Cox, D. E., & Hastings, J. B. (1987). *Rietveld refinement
of Debye-Scherrer synchrotron X-ray data from Al₂O₃*. Journal of
Applied Crystallography, **20**(2), 79–83.
DOI: [10.1107/S0021889887087090](https://doi.org/10.1107/S0021889887087090)

The pseudo-Voigt convention that lmfit's `PseudoVoigtModel` follows:
PV(x) = (1−η)·G(x) + η·L(x) with shared FWHM. The base profile for the
Stage 3D peak fit.

### `caglioti1958`
Caglioti, G., Paoletti, A., & Ricci, F. P. (1958). *Choice of collimators
for a crystal spectrometer for neutron diffraction*. Nuclear Instruments
and Methods, **3**(4), 223–228.
DOI: [10.1016/0369-643X(58)90029-X](https://doi.org/10.1016/0369-643X(58)90029-X)

U, V, W instrumental-broadening formula used in Rietveld refinement.
Not yet implemented in Latos; the Stage 3D peak-fit returns total
(instrumental + sample) FWHMs. Cited in the Stage 3D limitations as the
future cross-modal calibration step.

### `rachinger1948`
Rachinger, W. A. (1948). *A correction for the α₁ α₂ doublet in the
measurement of widths of X-ray diffraction lines*. Journal of Scientific
Instruments, **25**(7), 254–255.
DOI: [10.1088/0950-7671/25/7/125](https://doi.org/10.1088/0950-7671/25/7/125)

Rachinger Kα₂ stripping — the standard preprocessing step for non-
Rietveld peak-fit workflows on Cu-Kα data. Not yet implemented in Latos;
the Stage 3D analyzer treats Kα₁ and Kα₂ as independent peaks.

### `newville2014`
Newville, M., Stensitzki, T., Allen, D. B., & Ingargiola, A. (2014).
*lmfit: Non-Linear Least-Squares Minimization and Curve-Fitting for
Python*. Zenodo. DOI: [10.5281/zenodo.11813](https://doi.org/10.5281/zenodo.11813)

The non-linear least-squares optimizer used in Stage 3D peak fitting.
`PseudoVoigtModel` and `CompositeModel` come from this package.

### `daniels2020`
Daniels, P., & Connolley, T. (2020). *xrdfit: A Python package for
fitting XRD spectra*. Journal of Open Source Software, **5**(54), 2381.
DOI: [10.21105/joss.02381](https://doi.org/10.21105/joss.02381)

Prior-art Python XRD peak-fit package; the SNIP-then-find_peaks-then-
lmfit composite pipeline used in Stage 3D follows the same recipe.

### `virtanen2020`
Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). *SciPy 1.0:
fundamental algorithms for scientific computing in Python*. Nature
Methods, **17**(3), 261–272.
DOI: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

`scipy.signal.find_peaks`, `peak_widths`, and `savgol_filter` are used
throughout Stage 3D for peak detection and smoothing.

### `toby2013`
Toby, B. H., & Von Dreele, R. B. (2013). *GSAS-II: the genesis of a
modern open-source all purpose crystallography software package*.
Journal of Applied Crystallography, **46**(2), 544–549.
DOI: [10.1107/S0021889813003531](https://doi.org/10.1107/S0021889813003531)

Reference open-source XRD analysis stack; the Stage 3D pipeline aligns
with its profile-fitting conventions.

### `briggs2003`
Briggs, D., & Grant, J. T. (Eds.). (2003). *Surface Analysis by Auger
and X-ray Photoelectron Spectroscopy*. IM Publications.

Standard XPS reference covering peak fitting, satellite features, and
chemical-state identification. Background for CasaXPS parser; would
support a future XPS analyzer.

---

## Algorithms & software

### `winkler1990`
Winkler, W. E. (1990). *String Comparator Metrics and Enhanced Decision
Rules for the Fuhrer-Sunter Model of Record Linkage*. Proceedings of the
Section on Survey Research Methods, American Statistical Association,
354–359.

Jaro-Winkler similarity, one of three string-similarity metrics combined
in `cluster_samples()` (Stage 2C).

### `levenshtein1966`
Levenshtein, V. I. (1966). *Binary codes capable of correcting deletions,
insertions, and reversals*. Soviet Physics Doklady, **10**, 707–710.

Levenshtein edit distance, underlies `rapidfuzz.fuzz.ratio` used in
Stage 2C.

### `rapidfuzz`
Bachmann, M. (2020–). *rapidfuzz: rapid fuzzy string matching in Python
and C++*. [GitHub](https://github.com/maxbachmann/RapidFuzz)

Software implementation of the string-similarity metrics used in Stage 2C.

### `hagberg2008`
Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). *Exploring network
structure, dynamics, and function using NetworkX*. Proceedings of the
7th Python in Science Conference (SciPy2008), 11–15.

The graph library used in Stage 2C to extract connected components
from the pairwise similarity graph.

### `gpytorch`
Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson,
A. G. (2018). *GPyTorch: Blackbox Matrix-Matrix Gaussian Process
Inference with GPU Acceleration*. Advances in Neural Information
Processing Systems, **31**.
[arXiv:1809.11165](https://arxiv.org/abs/1809.11165)

To be used in Stage 6 for Bayesian-optimization surrogate models.

### `botorch`
Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson,
A. G., & Bakshy, E. (2020). *BoTorch: A Framework for Efficient
Monte-Carlo Bayesian Optimization*. Advances in Neural Information
Processing Systems, **33**, 21524–21538.
[arXiv:1910.06403](https://arxiv.org/abs/1910.06403)

To be used in Stage 6 for the Bayesian-optimization acquisition layer.

---

## Data formats & infrastructure

### `apache_parquet`
Apache Software Foundation. *Apache Parquet — columnar storage format*.
[https://parquet.apache.org](https://parquet.apache.org)

Latos stores per-measurement arrays in Parquet (Stage 1C).

### `sqlite`
Hipp, D. R., et al. (2000–). *SQLite — embedded SQL database engine*.
[https://www.sqlite.org](https://www.sqlite.org)

Latos's per-project metadata store (Stage 1B).

### `alembic`
Bayer, M. (2014–). *Alembic — database migrations for SQLAlchemy*.
[https://alembic.sqlalchemy.org](https://alembic.sqlalchemy.org)

Schema migrations between Latos versions (Stage 1B onward).

---

## To be added (placeholders)

References for future stages will be added as the methods land:

- Vision-language models for micrograph analysis (Stage 5) — VLM
  literature TBD
- Bayesian optimization with constraints (Stage 6) — Gardner et al.
  (2014), Gelbart et al. (2014)
- Constrained / multi-objective acquisition (Stage 6) — Daulton et al.
  (2020) for qEHVI
