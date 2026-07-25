/** Typed client for the latos-core sidecar (127.0.0.1 only).
 *
 * Shapes mirror `packages/core/src/latos/server/schemas.py` — when the
 * Python schema changes, change here too (codegen from OpenAPI is a
 * later nicety; the surface is small enough to mirror by hand for now).
 */

export const BASE = "http://127.0.0.1:8765";

export interface Health {
  status: string;
  version: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  root_path: string;
  samples: number;
  measurements: number;
  techniques: number;
  parsed: number;
  cached: number;
  failed: number;
  unclassified: number;
  review_status: "needs_review" | "confirmed";
}

export interface MeasurementSummary {
  id: string;
  technique: string;
  instrument: string | null;
  filename: string | null;
  features?: Record<string, number>;
}

export interface SampleSummary {
  id: string;
  name: string;
  aliases: string[];
  measurements: MeasurementSummary[];
}

export interface IngestProgress {
  index: number;
  total: number;
  name: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new Error(detail ?? `${init?.method ?? "GET"} ${path} → ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function openProject(root: string, projectName?: string): Promise<{ status: string }> {
  return request("/project/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ root, project_name: projectName ?? null }),
  });
}

export function getProject(): Promise<ProjectSummary> {
  return request<ProjectSummary>("/project");
}

export interface DeleteProjectResult {
  root: string;
  removed: boolean;
  recycled: boolean;
}

/** Recycle a project's derived `.latos/` store. Raw files are never touched. */
export function deleteProject(root: string): Promise<DeleteProjectResult> {
  return post<DeleteProjectResult>("/project/delete", { root });
}

export function getSamples(): Promise<SampleSummary[]> {
  return request<SampleSummary[]>("/samples");
}

/** Subscribe to the ingestion SSE stream.
 *
 * Calls `onProgress` per file tick, then exactly one of `onDone` /
 * `onError`, then closes itself. Returns a disposer for unmount.
 */
export function subscribeIngestEvents(handlers: {
  onProgress: (progress: IngestProgress) => void;
  onDone: () => void;
  onError: (message: string) => void;
}): () => void {
  const source = new EventSource(`${BASE}/ingest/events`);

  source.addEventListener("progress", (event) => {
    handlers.onProgress(JSON.parse((event as MessageEvent<string>).data) as IngestProgress);
  });
  source.addEventListener("done", () => {
    source.close();
    handlers.onDone();
  });
  source.addEventListener("error", (event) => {
    // Distinguish a server-sent `error` event (has data) from an
    // EventSource transport failure (no data).
    const data = (event as MessageEvent<string>).data;
    source.close();
    if (data) {
      const payload = JSON.parse(data) as { message?: string };
      handlers.onError(payload.message ?? "Ingestion failed");
    } else {
      handlers.onError("Lost connection to the Latos core");
    }
  });

  return () => source.close();
}

/** Poll /health until the sidecar answers (shell startup handshake). */
export async function waitForCore(timeoutMs = 15000, intervalMs = 250): Promise<Health> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown;
  while (Date.now() < deadline) {
    try {
      return await getHealth();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }
  throw new Error(`Latos core did not come up within ${timeoutMs}ms: ${String(lastError)}`);
}

export interface MeasurementArrays {
  measurement_id: string;
  names: string[];
  arrays: Record<string, (number | null)[]>;
}

export function getMeasurementArrays(id: string): Promise<MeasurementArrays> {
  return request<MeasurementArrays>(`/measurements/${id}/arrays`);
}

// ─── Fit engine (Stage 4) ──────────────────────────────────────────────
export type PeakShape =
  | "gaussian"
  | "lorentzian"
  | "voigt"
  | "pseudo_voigt"
  | "doniach"
  | "skewed_voigt";

export type BackgroundKind =
  | "none"
  | "constant"
  | "linear"
  | "polynomial"
  | "shirley"
  | "als";

export interface FitConstraint {
  type: "fixed_delta" | "fixed_ratio" | "shared_width";
  ref: number;
  target: number;
  delta?: number;
  ratio?: number;
}

export interface FitComponent {
  center: number;
  amplitude: number;
  sigma: number;
  fwhm: number | null;
  height: number | null;
}

export interface FitParam {
  name: string;
  value: number;
  stderr: number | null;
}

export interface FitResult {
  success: boolean;
  r_squared: number;
  chi_square: number;
  reduced_chi_square: number;
  components: FitComponent[];
  params: FitParam[];
  baseline: number[];
  best_fit: number[];
  residual: number[];
  markdown: string;
}

export interface FitRequest {
  x: number[];
  y: number[];
  peak_shape: PeakShape;
  peaks: number[];
  background: { kind: BackgroundKind; degree?: number; lam?: number; p?: number };
  constraints?: FitConstraint[];
}

export function detectPeaks(x: number[], y: number[], maxPeaks = 30): Promise<{ centers: number[] }> {
  return post<{ centers: number[] }>("/fit/detect-peaks", { x, y, max_peaks: maxPeaks });
}

export function runFit(req: FitRequest): Promise<FitResult> {
  return post<FitResult>("/fit", req);
}

export function getFitPresets(): Promise<{ doublets: Record<string, number[]> }> {
  return request<{ doublets: Record<string, number[]> }>("/fit/presets");
}

// ─── Cross-correlation + reporting (Stage 6) ───────────────────────────
export interface FeatureCell {
  value: number;
  unit: string;
  source: string;
  reliable: boolean;
}

export interface FeatureRow {
  sample_id: string;
  sample_name: string;
  features: Record<string, FeatureCell>;
}

export interface FeatureTable {
  properties: string[];
  rows: FeatureRow[];
}

export interface Correlation {
  property_a: string;
  property_b: string;
  pearson: number;
  spearman: number;
  n: number;
}

export interface Correlations {
  properties: string[];
  matrix: (number | null)[][];
  pairs: Correlation[];
}

/** The feature table can be slow to compute on first call (runs the analysis
 * layer across every sample); it is cached server-side thereafter. */
export function getFeatures(): Promise<FeatureTable> {
  return request<FeatureTable>("/features");
}

export function getCorrelations(reliableOnly = false): Promise<Correlations> {
  return request<Correlations>(`/correlations?reliable_only=${reliableOnly}`);
}

export function getReportStyles(): Promise<string[]> {
  return request<{ styles: string[] }>("/report/styles").then((r) => r.styles);
}

/** URL for a rendered publication figure (heatmap or scatter). */
export function figureUrl(opts: {
  kind: "heatmap" | "scatter";
  style: string;
  fmt: "svg" | "pdf" | "png";
  x?: string;
  y?: string;
}): string {
  const p = new URLSearchParams({ kind: opts.kind, style: opts.style, fmt: opts.fmt });
  if (opts.x) p.set("x", opts.x);
  if (opts.y) p.set("y", opts.y);
  return `${BASE}/report/figure?${p.toString()}`;
}

/** Fetch a figure as a blob and trigger a browser download. */
export async function downloadFigure(
  opts: Parameters<typeof figureUrl>[0],
  filename: string,
): Promise<void> {
  const resp = await fetch(figureUrl(opts));
  if (!resp.ok) throw new Error(`Figure export failed (${resp.status})`);
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export interface AnalyzerResult {
  analyzer: string;
  outputs: Record<string, unknown>;
  issues: string[];
}

export function getMeasurementAnalysis(id: string): Promise<AnalyzerResult[]> {
  return request<AnalyzerResult[]>(`/measurements/${id}/analysis`);
}

/** URL of the rendered PNG for an image measurement (TEM/SEM/STEM). */
export function measurementImageUrl(id: string): string {
  return `${BASE}/measurements/${id}/image`;
}

export const IMAGE_TECHNIQUES = new Set(["tem", "sem", "stem"]);

/** All technique values the core recognizes (for the override dropdown). */
export const TECHNIQUES = [
  "xrd",
  "xps",
  "uv_drs",
  "hall",
  "thermoelectric",
  "eds",
  "tem",
  "sem",
  "stem",
  "raman",
  "shock",
  "unknown",
] as const;

// ─── Review & Confirm edits (RB8 endpoints) ──────────────────────────
function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
}

export function confirmProject(): Promise<ProjectSummary> {
  return post<ProjectSummary>("/project/confirm");
}

export function reopenProject(): Promise<ProjectSummary> {
  return post<ProjectSummary>("/project/reopen");
}

export function renameSample(sampleId: string, name: string): Promise<ProjectSummary> {
  return post<ProjectSummary>(`/samples/${sampleId}/rename`, { name });
}

export function setMeasurementTechnique(
  measurementId: string,
  technique: string,
): Promise<ProjectSummary> {
  return post<ProjectSummary>(`/measurements/${measurementId}/technique`, { technique });
}

export function mergeSamples(
  sourceIds: string[],
  targetId: string,
): Promise<ProjectSummary> {
  return post<ProjectSummary>("/samples/merge", {
    source_ids: sourceIds,
    target_id: targetId,
  });
}

export function splitMeasurements(
  measurementIds: string[],
  newName: string,
): Promise<ProjectSummary> {
  return post<ProjectSummary>("/samples/split", {
    measurement_ids: measurementIds,
    new_name: newName,
  });
}

export interface MergeSuggestion {
  target_id: string;
  target_name: string;
  source_id: string;
  source_name: string;
  score: number;
  confidence: "high" | "medium";
  reason: string;
}

export function getMergeSuggestions(): Promise<MergeSuggestion[]> {
  return request<MergeSuggestion[]>("/samples/merge-suggestions");
}

export interface SampleAnomaly {
  sample_id: string;
  sample_name: string;
  kind: "mixed_samples" | "non_sample_name";
  message: string;
  related: string[];
}

export function getAnomalies(): Promise<SampleAnomaly[]> {
  return request<SampleAnomaly[]>("/samples/anomalies");
}

// ─── Optimization (BO) ───────────────────────────────────────────────
export type SampleParams = Record<string, Record<string, number>>;

export interface DatasetPoint {
  sample_id: string;
  sample_name: string;
  x: number;
  y: number;
  x2?: number | null; // companion (secondary-axis) value, when requested
}

export interface Recommendation {
  x: number;
  predicted_mean: number;
  ci95: number; // model (epistemic) 95% half-width
  ci95_predictive: number; // 95% half-width a new measurement should fall within
  predictive_interval_95: [number, number]; // [low, high] — for calibration checks
}

/** Self-assessed trustworthiness of the model's intervals. */
export type ReliabilityLevel = "exploratory" | "indicative" | "calibrated" | "unknown";

/** A dataset point whose target/axis value is flagged untrustworthy
 * (e.g. an unreliable Hall measurement, or a negative mobility). */
export interface QualityFlag {
  sample_name: string;
  variable: string;
  value: number;
  reason: string;
}

export interface OptimizeResult {
  input_variable: string;
  target_property: string; // display label (e.g. "|zT (derived) - 1|" in target mode)
  objective: Objective;
  reliability_level: ReliabilityLevel;
  reliability_note: string;
  quality_flags: QualityFlag[];
  grid_x: number[];
  grid_mean: number[];
  grid_ci95: number[];
  grid_lower: number[]; // explicit 95% band (physical units, clamped, log-aware)
  grid_upper: number[];
  grid_ei: number[];
  points: DatasetPoint[];
  best_x: number;
  best_y: number;
  recommendation: Recommendation;
  max_ei: number;
  noise_threshold: number;
  converged: boolean;
  /** "exploit" (highest expected improvement) or "explore" (least-sampled
   * region — chosen when the data is too thin to trust an "optimum" verdict). */
  recommendation_kind: "exploit" | "explore";
  verdict: string;
  /** Optional companion axis (e.g. vol% shown next to a wt% run). When
   * `secondary_variable` is set, `secondary_slope * x + secondary_intercept`
   * maps the primary axis to the companion units. */
  secondary_variable: string;
  secondary_slope: number | null;
  secondary_intercept: number | null;
}

export function getParameters(): Promise<SampleParams> {
  return request<SampleParams>("/parameters");
}

export function setSampleParameters(
  sampleId: string,
  parameters: Record<string, number>,
): Promise<{ status: string }> {
  return post<{ status: string }>(`/samples/${sampleId}/parameters`, { parameters });
}

export function getOptimizeTargets(): Promise<string[]> {
  return request<{ properties: string[] }>("/optimize/targets").then((r) => r.properties);
}

/** One sample's single-parabolic-band read (Seebeck vs zT at its zT peak).
 * When `applicable` is false the (Seebeck, zT) pair is inconsistent with a
 * single band — a multi-band / data flag, not a fabricated target. */
export interface SpbSample {
  sample_name: string;
  applicable: boolean;
  note: string;
  measured_seebeck_uv_k: number;
  measured_zt: number;
  beta: number | null;
  optimal_seebeck_uv_k: number | null;
  zt_ceiling: number | null;
  direction: "increase_seebeck" | "decrease_seebeck" | "at_optimum" | null;
}

export interface SpbCheckResult {
  best: SpbSample | null;
  samples: SpbSample[];
}

export function getSpbCheck(): Promise<SpbCheckResult> {
  return request<SpbCheckResult>("/optimize/spb");
}

/** One available BO input axis. `synthesis` values are researcher-entered
 * (editable); `measured` values come from instruments (read-only). */
export interface InputVariableInfo {
  name: string;
  source: "synthesis" | "measured";
  values: Record<string, number>; // sample_id -> value
}

export function getOptimizeInputs(): Promise<InputVariableInfo[]> {
  return request<InputVariableInfo[]>("/optimize/inputs");
}

/** What "best" means for a run: a direction, or reaching a target value. */
export type Objective = "maximize" | "minimize" | "target";

export interface OptimizeOptions {
  objective?: Objective;
  targetValue?: number; // required when objective === "target"
  atTemperatureK?: number; // derived-zT only: zT at this T instead of the peak
  secondaryVariable?: string; // companion axis to display alongside the primary
}

function optimizeBody(
  inputVariable: string,
  targetProperty: string,
  opts: OptimizeOptions,
): Record<string, unknown> {
  return {
    input_variable: inputVariable,
    target_property: targetProperty,
    objective: opts.objective ?? "maximize",
    target_value: opts.targetValue ?? null,
    at_temperature_k: opts.atTemperatureK ?? null,
    secondary_variable: opts.secondaryVariable ?? null,
  };
}

export function runOptimize(
  inputVariable: string,
  targetProperty: string,
  opts: OptimizeOptions = {},
): Promise<OptimizeResult> {
  return post<OptimizeResult>("/optimize/run", optimizeBody(inputVariable, targetProperty, opts));
}

export interface FreezeResult {
  path: string;
  recommendation: Recommendation;
  prior_best: number;
  robustness_stable: boolean;
  converged: boolean;
  reliability_level: ReliabilityLevel;
  reliability_note: string;
}

/** Freeze the current recommendation into an auditable pre-registration record. */
export function freezeRecommendation(
  inputVariable: string,
  targetProperty: string,
  opts: OptimizeOptions = {},
): Promise<FreezeResult> {
  return post<FreezeResult>(
    "/optimize/freeze",
    optimizeBody(inputVariable, targetProperty, opts),
  );
}

// ─── Loop-closer: validate a synthesized outcome vs the frozen prediction ──

/** The score of a measured outcome against a frozen prediction. */
export interface OutcomeVerdict {
  measured: number;
  predicted_mean: number;
  predictive_interval_95: [number, number];
  prior_best: number;
  direction: "maximize" | "minimize";
  within_interval: boolean; // calibration
  improved: boolean; // improvement over prior best, direction-aware
  signed_error: number;
  absolute_error: number;
  relative_error: number | null;
  summary: string;
  validated_at: string;
}

/** One frozen pre-registration and its recorded outcome (if validated). */
export interface PreregSummary {
  path: string;
  created_at: string;
  input_variable: string;
  property_name: string;
  direction: "maximize" | "minimize";
  recommended_x: number;
  predicted_mean: number;
  predictive_interval_95: [number, number];
  prior_best: number;
  reliability_level: ReliabilityLevel;
  outcome: OutcomeVerdict | null;
}

export function listPreregistrations(): Promise<PreregSummary[]> {
  return request<PreregSummary[]>("/optimize/prereg");
}

/** Score a measured value against a frozen record; persists the verdict. */
export function validateOutcome(
  preregPath: string,
  measuredValue: number,
): Promise<OutcomeVerdict> {
  return post<OutcomeVerdict>("/optimize/validate", {
    prereg_path: preregPath,
    measured_value: measuredValue,
  });
}
