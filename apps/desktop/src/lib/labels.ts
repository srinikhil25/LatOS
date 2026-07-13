/** Scientific axis labels for plots.
 *
 * The core exposes arrays and optimization variables under machine-safe
 * snake_case names (`two_theta`, `resistivity_uohm_m`, …). Those names must
 * never reach a chart axis verbatim — this is a scientific tool, and an axis
 * reading "two_theta" or "seebeck_uv_k" is wrong. This map turns each known
 * name into its correct symbol + unit, following thermoelectric-literature
 * convention (µ = micro, θ = theta, superscript exponents, · for products).
 *
 * Unknown names (e.g. a researcher's own synthesis variable) fall back to a
 * de-underscored version — never mangled, and already-formatted labels such
 * as "zT (derived)" pass through unchanged.
 */

const AXIS_LABELS: Record<string, string> = {
  // ─ XRD ─
  two_theta: "2θ (°)",
  intensity: "Intensity (a.u.)",
  counts: "Counts",
  // ─ XPS ─
  binding_energy: "Binding energy (eV)",
  // ─ EDS ─
  energy_kev: "Energy (keV)",
  // ─ UV-DRS ─
  wavelength: "Wavelength (nm)",
  reflectance: "Reflectance (%R)",
  photon_energy_ev: "Photon energy, hν (eV)",
  kubelka_munk: "F(R) (a.u.)",
  tauc_y: "(F(R)·hν)ⁿ (a.u.)",
  // ─ Transport (R&S + LFA) ─
  temperature_k: "Temperature (K)",
  temperature_c: "Temperature (°C)",
  resistivity_uohm_m: "Resistivity (µΩ·m)",
  resistivity_ohm_m: "Resistivity (Ω·m)",
  resistivity_ohm_cm: "Resistivity (Ω·cm)",
  seebeck_uv_k: "Seebeck coefficient (µV/K)",
  seebeck_uvk: "Seebeck coefficient (µV/K)",
  thermal_conductivity: "Thermal conductivity (W/m·K)",
  diffusivity_mm2_s: "Thermal diffusivity (mm²/s)",
  cp_j_gk: "Specific heat, cₚ (J/g·K)",
  // ─ Derived thermoelectric ─
  zt: "zT",
  power_factor: "Power factor (µW/m·K²)",
  power_factor_uw_mk2: "Power factor (µW/m·K²)",
  // ─ Hall / electronic ─
  carrier_concentration_cm3: "Carrier concentration (cm⁻³)",
  mobility_cm2_vs: "Mobility (cm²/V·s)",
  conductivity_s_cm: "Conductivity (S/cm)",
  hall_coefficient_cm3_c: "Hall coefficient (cm³/C)",
  sheet_resistance_ohm_sq: "Sheet resistance (Ω/□)",
  // ─ Optical / band structure ─
  band_gap_ev: "Band gap (eV)",
  // ─ Cross-sample features (Correlate page) ─
  crystallite_size_nm: "Crystallite size (nm)",
  peak_zt: "Peak zT",
  peak_seebeck_uv_k: "Peak Seebeck (µV/K)",
  peak_thermal_conductivity: "Peak thermal conductivity (W/m·K)",
  // ─ Synthesis variables (common defaults) ─
  doping_pct: "Doping (%)",
  // ─ Mechanical shock (drop test) ─
  time_s: "Time (s)",
  voltage_v: "Sensor voltage (V)",
  // ─ Plot fallbacks ─
  index: "Index",
};

/** The scientific label for a raw array / variable name.
 *
 * Exact-match against the known map first; otherwise de-underscore the name
 * (leaving already-formatted labels like "zT (derived)" untouched).
 */
export function axisLabel(name: string): string {
  const known = AXIS_LABELS[name];
  if (known) return known;
  return name.includes("_") ? name.replace(/_/g, " ").trim() : name;
}

/** Scientific labels for analyzer *output* keys (shown in the analysis panel).
 *
 * The analyzers report scalars under snake_case keys that carry units in the
 * name (`peak_centers_2theta`, `d_spacings_angstrom`, `kappa_range_w_mk`).
 * Rendered verbatim these read wrong on a scientific readout — 2theta must be
 * 2θ, angstrom must be Å, kappa must be κ, chi_square must be χ².
 */
const OUTPUT_KEY_LABELS: Record<string, string> = {
  // ─ XRD peak fit ─
  n_peaks: "Peak count",
  peak_centers_2theta: "Peak centers, 2θ (°)",
  peak_heights: "Peak heights",
  peak_fwhms_2theta: "Peak FWHM, 2θ (°)",
  peak_areas: "Peak areas",
  peak_fractions: "Peak Lorentzian fractions",
  d_spacings_angstrom: "d-spacings (Å)",
  scherrer_sizes_nm: "Scherrer sizes (nm)",
  mean_crystallite_size_nm: "Mean crystallite size (nm)",
  r_squared: "R²",
  fit_r_squared: "Fit R²",
  reduced_chi_square: "Reduced χ²",
  noise_sigma_estimate: "Noise σ estimate",
  // ─ UV-DRS Tauc ─
  band_gap_ev: "Band gap (eV)",
  band_gap_type: "Band-gap type",
  photon_energy_ev: "Photon energy, hν (eV)",
  // ─ EDS ─
  composition_rel_pct: "Composition (rel. at.%)",
  // ─ XPS ─
  region: "Region",
  peak_binding_energies_ev: "Peak binding energies (eV)",
  peak_fwhms_ev: "Peak FWHM (eV)",
  main_peak_be_ev: "Main peak, BE (eV)",
  be_range_ev: "BE range (eV)",
  charge_offset_vs_c1s_284p8_ev: "Charge offset vs C 1s (284.8 eV)",
  // ─ Hall ─
  carrier_type: "Carrier type",
  carrier_type_from_seebeck: "Carrier type (from Seebeck)",
  carrier_type_reliability: "Carrier-type reliability",
  carrier_concentration_cm3: "Carrier concentration (cm⁻³)",
  mobility_cm2_vs: "Mobility (cm²/V·s)",
  resistivity_ohm_cm: "Resistivity (Ω·cm)",
  conductivity_s_cm: "Conductivity (S/cm)",
  conductivity_from_n_mu_s_cm: "σ from q·n·µ (S/cm)",
  consistency_deviation_pct: "Consistency deviation (%)",
  hall_coefficient_cm3_c: "Hall coefficient (cm³/C)",
  sheet_resistance_ohm_sq: "Sheet resistance (Ω/□)",
  hall_cross_ratio: "Hall cross-config ratio",
  // ─ Transport summary ─
  kind: "Kind",
  temperature_range_k: "Temperature range (K)",
  kappa_range_w_mk: "κ range (W/m·K)",
  kappa_min_at_k: "κ minimum at (K)",
  seebeck_max_uv_k: "Seebeck max (µV/K)",
  seebeck_max_at_k: "Seebeck max at (K)",
  resistivity_range_uohm_m: "Resistivity range (µΩ·m)",
  peak_power_factor_uw_mk2: "Peak power factor (µW/m·K²)",
  peak_power_factor_at_k: "Peak power factor at (K)",
  // ─ Mechanical shock ─
  peak_voltage_v: "Peak voltage (V)",
  peak_time_ms: "Peak time (ms)",
};

/** The scientific label for an analyzer output key. */
export function outputKeyLabel(key: string): string {
  return OUTPUT_KEY_LABELS[key] ?? AXIS_LABELS[key] ?? key.replace(/_/g, " ").trim();
}
