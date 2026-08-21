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

## Ionic thermoelectrics

- **Ionic thermoelectric (i-TE)** — A material where a temperature
  difference is converted to a voltage by *ions* rather than electrons.
  Seebeck coefficients reach millivolts per kelvin, a hundred times the
  electronic case, because an ion carries far more transported entropy
  than an electron.

- **Thermodiffusion (Soret effect)** — Ions migrate along a temperature
  gradient at different rates depending on species, so cations and anions
  separate and a voltage builds. Transient: it charges over hundreds to
  thousands of seconds and yields no steady current.

- **Thermogalvanic effect** — The other i-TE mechanism, driven by a redox
  couple whose equilibrium potential is temperature dependent. Gives a
  steady current, unlike thermodiffusion, but needs a redox species.

- **Electrode polarisation offset (ΔV_electrode)** — The part of a
  measured open-circuit voltage that comes from ion adsorption at the
  electrodes rather than from the thermal gradient. It does not scale with
  ΔT, so `ΔV = S·ΔT + ΔV_electrode`, and a single-point measurement folds
  it silently into the reported coefficient. Fitting across several ΔT
  values separates the two.

- **Hydrovoltaic voltage** — Voltage generated by a *water* concentration
  gradient rather than an ionic thermal one. In IL-cellulose systems it
  can raise the apparent Seebeck coefficient from 3 to 12.5 mV/K, which is
  why humidity is a Tier-1 recorded field rather than ambient context.

- **Heat of transport (Q\*)** — The energy carried by an ion as it moves,
  beyond its own enthalpy. Sets the thermodiffusive Seebeck coefficient:
  for a dual-ion electrolyte, `S = (w₊Q₊* − w₋Q₋*)/(eT)`.

## Optimization & experiment planning

- **Pre-registration** — A recommendation frozen to disk *before* the
  recommended sample is made, carrying the predicted value, its interval
  and the exact model configuration. It is the evidentiary basis for any
  closed-loop claim: a prediction written afterwards proves nothing.

- **Heteroscedastic** — Having non-constant measurement noise. A
  heteroscedastic Gaussian process is told how well *each* observation is
  known, so precise points pull the surface and vague ones are held
  loosely. Latos supplies this through `point_noise`.

- **Per-datapoint reliability** — Latos's alternative to deleting
  suspicious data: every value keeps a score and the reason for it, and
  that score reaches the optimizer as a variance rather than as a
  discard.

- **(ε, δ) stopping criterion** — "The best sample so far is within ε of
  the true optimum, with probability at least 1 − δ." Reported by Latos as
  `prob_within_epsilon`, and conditional on the model being right — which
  is why it is stated beside the data-sufficiency grade.

- **Campaign rehearsal** — Simulating a planned campaign against response
  curves whose optimum is known, before any sample is made, to size the
  budget and audition a prior. It tests the optimizer, never the material.

- **Prior audition** — Running a rehearsal twice, with and without a
  physics-informed prior mean, and reporting HELPS / NEUTRAL / HARMS.
  Three separate priors have now been measured to not help, so a prior is
  tested rather than assumed.

- **Discriminating shape** — In a rehearsal, a response curve whose
  optimum lies strictly inside the search range. Shapes with an endpoint
  optimum are solved by the seed design before any model runs and cannot
  separate one strategy from another, so headline numbers exclude them.
