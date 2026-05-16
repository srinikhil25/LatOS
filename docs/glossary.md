# Glossary

Cross-discipline terms used in Latos and the thesis. Materials-science
readers will find the software half useful; software readers will find
the materials half useful.

## Materials characterization techniques

- **XRD** (X-Ray Diffraction) — Identifies crystalline phases by
  measuring the angles at which X-rays diffract off a powdered or
  polycrystalline sample. Output: intensity vs. 2θ (the diffraction
  angle).

- **XPS** (X-ray Photoelectron Spectroscopy) — Probes a sample's
  surface composition and chemical states by measuring the binding
  energies of photoelectrons emitted under monochromatic X-ray
  irradiation. Output: intensity vs. binding energy (eV).

- **UV-DRS** (Ultraviolet-Visible Diffuse Reflectance Spectroscopy) —
  Measures the fraction of incident light a powdered sample reflects
  diffusely across UV-visible-near-IR wavelengths. Used to extract
  optical band gaps via the Tauc procedure. Output: reflectance vs.
  wavelength (nm).

- **Hall effect measurement** — Determines carrier type, density, and
  mobility in a semiconductor by measuring the transverse voltage that
  appears across a current-carrying sample in a perpendicular magnetic
  field.

- **Thermoelectric characterization** — Combined measurement of Seebeck
  coefficient, electrical resistivity, and thermal conductivity vs.
  temperature, used to compute the figure of merit zT = S²σT/κ.

- **EDS** (Energy-Dispersive X-ray Spectroscopy) — Identifies elements
  in a sample from the characteristic X-rays emitted under an electron
  beam (typically inside an SEM). Output: counts vs. energy (keV).

- **SEM / TEM / STEM** — Scanning / Transmission / Scanning-Transmission
  Electron Microscopy. Imaging modalities. Latos currently stores TEM/
  SEM/STEM TIFFs as metadata-only (image content deferred to Stage 5).

## Derived quantities

- **Band gap (Eg)** — Energy difference between the valence-band
  maximum and the conduction-band minimum of a semiconductor.
  Determines the longest-wavelength photon a material can absorb.

- **Direct vs. indirect band gap** — A direct gap allows photon
  absorption without phonon assistance (e.g. GaAs); an indirect gap
  requires phonon assistance (e.g. Si). Optical absorption rises with
  energy as `(E-Eg)^{1/2}` for direct and `(E-Eg)^2` for indirect.
  Drives the Tauc-plot exponent choice.

- **Kubelka-Munk function** F(R) = (1-R)² / 2R — Maps the diffuse
  reflectance R of a thick, weakly-absorbing scatterer to a quantity
  proportional to the absorption coefficient α. Lets UV-DRS data be
  interpreted as absorption.

- **Tauc plot** — Plot of (αhν)^n vs. hν (photon energy). For a
  semiconductor with a clean absorption edge, the linear region's
  x-intercept is Eg. n = 2 for direct gaps; n = 1/2 for indirect.

- **zT (thermoelectric figure of merit)** = S²σT/κ — Dimensionless
  efficiency metric for thermoelectric materials. S = Seebeck
  coefficient, σ = electrical conductivity, T = absolute temperature,
  κ = thermal conductivity. Higher = better.

- **Photon energy ⇄ wavelength** E[eV] = 1240 / λ[nm] — The conversion
  factor `hc/e`. Used everywhere UV-visible spectroscopy is involved.

## Materials in the Cs₃Bi₂I₉ system

- **Cs₃Bi₂I₉** — Cesium bismuth iodide, a lead-free perovskite-related
  semiconductor. Of interest as a Pb-free analogue of CsPbI₃ for
  photovoltaic and thermoelectric applications.

- **MXene** — A family of 2D transition-metal carbides/nitrides (e.g.
  Ti₃C₂Tx). Latos's test dataset includes MXene samples studied as
  thermoelectric materials.

- **Dhivya dataset** — The maintainer's predecessor's characterization
  data of mixed Cs-Bi-I phases, used as the real-data integration
  fixture (~161 files, ~590 MB).

## Software & tooling

- **Frozen dataclass** — A Python `@dataclass(frozen=True, slots=True)`
  whose instances are immutable after construction. Latos's domain
  model is built entirely from these for safety + hashability.

- **Repository pattern** — A persistence abstraction where domain code
  only sees domain objects, never ORM rows. The repository is the only
  module allowed to bridge the two. `ProjectRepository` is Latos's
  implementation.

- **Alembic** — SQLAlchemy's migration tool. Latos uses it for
  schema-versioned upgrade/downgrade between Latos versions.

- **Parquet** — A columnar, compressed binary format from the Apache
  Arrow project. Latos stores measurement arrays as one Parquet file
  per measurement, making them readable by pandas / DuckDB / Power
  Query without nested-type handling.

- **`ParsedData`** — The universal contract every Latos parser returns:
  technique, arrays, metadata, instrument, measured_at, issues,
  parser_name, parser_version. Differences between techniques live in
  `arrays` and `metadata` only.

- **`AnalysisResult`** — The universal contract every Latos analyzer
  produces: id, measurement_id, analyzer_name, analyzer_version,
  params, outputs, derived_arrays_path, issues, computed_at. Stored
  alongside the parent Measurement.

- **SHA-256 + parser_version cache key** — Latos's idempotent
  re-ingestion strategy: a file is re-parsed iff either its content
  hash or the parser's version has changed since the last ingest.

- **Confidence-pick dispatch** — Latos's parser-selection algorithm:
  every registered parser declares its confidence in [0, 1] for a
  given file; the highest scorer above a 0.5 threshold wins. Ties
  broken by registration order.

- **rapidfuzz** — A fast C++ implementation of fuzzy-string-matching
  algorithms (Levenshtein, Jaro-Winkler, token-sort, …). Powers
  Latos's sample-name clustering.

- **networkx** — Pure-Python graph library. Latos uses it to build a
  similarity graph from pairwise fuzzy scores and extract connected
  components as sample clusters.
