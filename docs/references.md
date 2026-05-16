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
- Peak-fit / Voigt profiles for XRD (Stage 3 future analyzer) — likely
  Caglioti et al. (1958), Toby & Von Dreele (2013)
- Bayesian optimization with constraints (Stage 6) — Gardner et al.
  (2014), Gelbart et al. (2014)
- Constrained / multi-objective acquisition (Stage 6) — Daulton et al.
  (2020) for qEHVI
