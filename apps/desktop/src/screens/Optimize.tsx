/** Optimize — the closed loop, in the app (BO4).
 *
 * The researcher NAMES their synthesis variable (doping %, etching time,
 * annealing temperature, …), enters its value per sample, picks the property
 * to maximize, and clicks Run. Latos fits the model and shows: the curve, the
 * recommended next experiment (★), and a plain-language verdict. The optimizer
 * is variable-agnostic — nothing on this screen is specific to doping.
 *
 * Reachable only once the project is CONFIRMED (the gate).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  freezeRecommendation,
  getCampaignDrift,
  getDistrusted,
  getOptimizeInputs,
  getOptimizeTargets,
  getParameters,
  getSamples,
  getSpbCheck,
  listPreregistrations,
  runOptimize,
  runOptimizeNd,
  setDistrusted,
  setSampleParameters,
  validateOutcome,
  type CampaignDrift,
  type FreezeResult,
  type InputVariableInfo,
  type Objective,
  type OptimizeNdResult,
  type OptimizeOptions,
  type OptimizeResult,
  type PreregSummary,
  type SampleParams,
  type SampleSummary,
  type SpbCheckResult,
} from "../lib/api";
import { OptimizeChart } from "../components/OptimizeChart";
import { OptimizeSurfaceChart, type SurfaceMode } from "../components/OptimizeSurfaceChart";
import { ChartFrame } from "../components/ChartFrame";
import { AnalysisLoader } from "../components/AnalysisLoader";

/** "etching_time_h" → "etching time h" for on-screen labels. */
const humanize = (v: string) => v.replace(/_/g, " ").trim() || "variable";

/** Compact numeric formatting spanning doping (~1) to carrier density (~1e21). */
const fmtVal = (v: number): string => {
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-2)) return v.toExponential(2);
  return v.toFixed(2);
};

/** Chip styling for the model's self-assessed reliability level. */
function reliabilityChipClass(level: string): string {
  const base = "rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide ";
  if (level === "calibrated") {
    return (
      base +
      "bg-[color-mix(in_srgb,var(--latos-tech-eds)_18%,transparent)] text-[color:var(--latos-tech-eds)]"
    );
  }
  if (level === "indicative") {
    return base + "bg-[color-mix(in_srgb,var(--latos-accent)_15%,transparent)] text-accent";
  }
  // exploratory / unknown — the "handle with care" colour.
  return (
    base +
    "bg-[color-mix(in_srgb,var(--latos-severity-warning)_18%,transparent)] text-severity-warning"
  );
}

/** The real steps runOptimize runs through, shown while a run is in flight. */
const OPTIMIZE_STAGES = [
  "Standardizing inputs & assembling (X, y)…",
  "Fitting the Gaussian-process surrogate…",
  "Optimizing the acquisition function…",
  "Checking kernel robustness & reliability…",
];

export function Optimize({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [params, setParams] = useState<SampleParams>({});
  const [inputInfos, setInputInfos] = useState<InputVariableInfo[]>([]);
  const [inputVar, setInputVar] = useState<string>("");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  // Optional second axis. Empty means the one-variable run, which stays the
  // default: it is the only shape /optimize/freeze can pre-register.
  const [secondVar, setSecondVar] = useState<string>("");
  const [secondInputs, setSecondInputs] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [objective, setObjective] = useState<Objective>("maximize");
  const [targetValue, setTargetValue] = useState<string>("");
  const [atTempK, setAtTempK] = useState<string>("");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [ndResult, setNdResult] = useState<OptimizeNdResult | null>(null);
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>("mean");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [frozen, setFrozen] = useState<FreezeResult | null>(null);
  const [freezing, setFreezing] = useState(false);
  const [preregs, setPreregs] = useState<PreregSummary[]>([]);
  const [drift, setDrift] = useState<CampaignDrift[]>([]);
  const [measuredInputs, setMeasuredInputs] = useState<Record<string, string>>({});
  const [validatingPath, setValidatingPath] = useState<string | null>(null);
  const [spb, setSpb] = useState<SpbCheckResult | null>(null);
  const [distrusted, setDistrustedIds] = useState<Set<string>>(new Set());

  const loadPreregs = useCallback(() => {
    listPreregistrations()
      .then(setPreregs)
      .catch(() => setPreregs([]));
    // Drift reads the same frozen records, so it refreshes with them.
    getCampaignDrift()
      .then(setDrift)
      .catch(() => setDrift([]));
  }, []);

  /** Toggle the researcher's quality call, then re-run so the effect is visible. */
  const toggleDistrust = useCallback(
    (sampleId: string, next: boolean) => {
      setDistrusted(sampleId, next)
        .then((ids) => setDistrustedIds(new Set(ids)))
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    },
    [],
  );

  // Every available input axis: synthesis parameters + measured features.
  const knownVars = useMemo(
    () => inputInfos.map((v) => v.name),
    [inputInfos],
  );
  const selectedInfo = useMemo(
    () => inputInfos.find((v) => v.name === inputVar.trim()),
    [inputInfos, inputVar],
  );
  // Measured variables (e.g. Hall carrier concentration) come from an
  // instrument — shown read-only, never written back as synthesis params.
  const isMeasured = selectedInfo?.source === "measured";
  const secondInfo = useMemo(
    () => inputInfos.find((v) => v.name === secondVar.trim()),
    [inputInfos, secondVar],
  );
  const secondIsMeasured = secondInfo?.source === "measured";
  const twoAxis = secondVar.trim() !== "" && secondVar.trim() !== inputVar.trim();

  /** The stored values of `varName`, as editable strings, one per sample. */
  const valuesFor = useCallback(
    (
      varName: string,
      synth: SampleParams,
      infos: InputVariableInfo[],
      tree: SampleSummary[],
    ) => {
      const info = infos.find((v) => v.name === varName);
      const next: Record<string, string> = {};
      for (const s of tree) {
        const v =
          info?.source === "measured"
            ? info.values[s.id]
            : synth[s.id]?.[varName];
        next[s.id] = v === undefined ? "" : String(v);
      }
      return next;
    },
    [],
  );

  useEffect(() => {
    Promise.all([getSamples(), getParameters(), getOptimizeTargets(), getOptimizeInputs()])
      .then(([tree, p, props, infos]) => {
        setSamples(tree);
        setParams(p);
        setInputInfos(infos);
        // Backward-compatible default; first available axis for a new project.
        const first =
          infos.find((v) => v.name === "doping_pct")?.name ?? infos[0]?.name ?? "";
        setInputVar(first);
        setInputs(valuesFor(first, p, infos, tree));
        setTargets(props);
        const preferred =
          props.find((x) => x === "zT (derived)") ??
          (props.includes("zt") ? "zt" : props[0]);
        setTarget(preferred ?? "");
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    loadPreregs();
    // Physics read is independent of the run controls; fetch once. A failure
    // here (e.g. no thermoelectric data) simply hides the panel.
    getSpbCheck()
      .then(setSpb)
      .catch(() => setSpb(null));
    getDistrusted()
      .then((ids) => setDistrustedIds(new Set(ids)))
      .catch(() => setDistrustedIds(new Set()));
  }, [valuesFor, loadPreregs]);

  // Switching (or naming) the variable reloads its stored values into the table.
  const chooseVar = useCallback(
    (name: string) => {
      const v = name.trim();
      setInputVar(v);
      setInputs(valuesFor(v, params, inputInfos, samples));
      // The second-variable list excludes whatever the first one is, so naming
      // the second variable here would leave that select holding a value it no
      // longer offers. Drop back to the one-variable run instead.
      setSecondVar((s) => (s === v ? "" : s));
    },
    [params, inputInfos, samples, valuesFor],
  );

  const chooseSecondVar = useCallback(
    (name: string) => {
      const v = name.trim();
      setSecondVar(v);
      setSecondInputs(valuesFor(v, params, inputInfos, samples));
    },
    [params, inputInfos, samples, valuesFor],
  );

  // Persist one cell, preserving the sample's OTHER variables. Measured
  // variables are instrument-derived and never written back.
  const saveValue = useCallback(
    (sampleId: string, raw: string, varName: string, measured: boolean) => {
      if (!varName || measured) return;
      const rest = { ...(params[sampleId] ?? {}) };
      const value = Number.parseFloat(raw);
      if (Number.isFinite(value)) rest[varName] = value;
      else delete rest[varName];
      setParams((p) => ({ ...p, [sampleId]: rest }));
      void setSampleParameters(sampleId, rest);
    },
    [params],
  );
  const saveInput = useCallback(
    (sampleId: string, raw: string) => saveValue(sampleId, raw, inputVar, isMeasured),
    [saveValue, inputVar, isMeasured],
  );
  const saveSecond = useCallback(
    (sampleId: string, raw: string) =>
      saveValue(sampleId, raw, secondVar.trim(), Boolean(secondIsMeasured)),
    [saveValue, secondVar, secondIsMeasured],
  );

  // The objective options sent with run/freeze, parsed from the controls.
  const opts = useMemo<OptimizeOptions>(() => {
    const o: OptimizeOptions = { objective };
    if (objective === "target") {
      const tv = Number.parseFloat(targetValue);
      if (Number.isFinite(tv)) o.targetValue = tv;
    }
    if (target === "zT (derived)") {
      const tk = Number.parseFloat(atTempK);
      if (Number.isFinite(tk)) o.atTemperatureK = tk;
    }
    return o;
  }, [objective, targetValue, atTempK, target]);

  const targetValueMissing = objective === "target" && opts.targetValue === undefined;

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    setNdResult(null);
    setFrozen(null);
    try {
      // One result at a time: the two shapes answer the same question over
      // different numbers of axes, and showing both at once would invite
      // reading a curve as if it were a slice of the surface, which it is not.
      if (twoAxis) {
        setNdResult(await runOptimizeNd([inputVar.trim(), secondVar.trim()], target, opts));
      } else {
        setResult(await runOptimize(inputVar, target, opts));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [inputVar, secondVar, twoAxis, target, opts]);

  const freeze = useCallback(async () => {
    setFreezing(true);
    setError(null);
    try {
      setFrozen(await freezeRecommendation(inputVar, target, opts));
      loadPreregs();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setFreezing(false);
    }
  }, [inputVar, target, opts, loadPreregs]);

  // Loop-closer: score a synthesized sample's measured value against its
  // frozen prediction, then refresh so the verdict shows in place.
  const validate = useCallback(
    async (path: string) => {
      const measured = Number.parseFloat(measuredInputs[path] ?? "");
      if (!Number.isFinite(measured)) return;
      setValidatingPath(path);
      setError(null);
      try {
        await validateOutcome(path, measured);
        loadPreregs();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setValidatingPath(null);
      }
    },
    [measuredInputs, loadPreregs],
  );

  const filledCount = Object.values(inputs).filter((v) => v.trim() !== "").length;
  // On two axes the run is over the samples that have BOTH values, so that
  // intersection — not either column on its own — is what gates the button.
  const pairedCount = samples.filter(
    (s) => (inputs[s.id] ?? "").trim() !== "" && (secondInputs[s.id] ?? "").trim() !== "",
  ).length;
  const usableCount = twoAxis ? pairedCount : filledCount;
  const varLabel = humanize(inputVar);
  // How many down-weighted points came from the researcher rather than the
  // physics checks. Read defensively: the sidecar is a separate process and an
  // older one omits the field, which must not turn into "undefined" on screen.
  const distrustedCount = Number.isFinite(result?.n_distrusted) ? result!.n_distrusted : 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-edge px-6 py-3">
        <button
          type="button"
          onClick={onBack}
          className="text-sm text-secondary underline-offset-4 hover:underline"
        >
          ← Hub
        </button>
        <h1 className="text-sm font-semibold uppercase tracking-wide text-secondary">
          Optimize — next experiment
        </h1>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-6 px-8 py-6">
          {error && (
            <div className="rounded-md border border-edge bg-[color-mix(in_srgb,var(--latos-severity-warning)_10%,transparent)] px-4 py-3 text-sm" data-selectable>
              {error}
            </div>
          )}

          {/* Physics check — single parabolic band. Independent of a run:
              interprets the best sample's measured (Seebeck, zT) against
              thermoelectric physics, on the reliable Seebeck axis. */}
          {spb?.best && (
            <section
              className={`rounded-lg border px-5 py-4 text-sm ${
                spb.best.applicable
                  ? "border-edge bg-surface"
                  : "border-severity-warning bg-[color-mix(in_srgb,var(--latos-severity-warning)_12%,transparent)]"
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs font-semibold uppercase tracking-wide text-secondary">
                  Physics check — single parabolic band
                </span>
                <span className="text-xs text-secondary" data-selectable>
                  best: {spb.best.sample_name}
                </span>
              </div>
              {spb.best.applicable ? (
                <div className="mt-2 space-y-1">
                  <p className="text-primary" data-selectable>
                    Quality factor β ≈ {fmtVal(spb.best.beta ?? 0)}. Measured |S| ={" "}
                    {fmtVal(spb.best.measured_seebeck_uv_k)} µV/K; peak-zT optimum at |S| ≈{" "}
                    {fmtVal(spb.best.optimal_seebeck_uv_k ?? 0)} µV/K.
                  </p>
                  <p className="text-secondary" data-selectable>
                    {spb.best.direction === "at_optimum"
                      ? "This sample sits near its single-band zT optimum."
                      : spb.best.direction === "increase_seebeck"
                        ? "Under-doped for peak zT — lower the carrier concentration (raise |S|)."
                        : "Over-doped for peak zT — raise the carrier concentration (lower |S|)."}
                  </p>
                </div>
              ) : (
                <div className="mt-2 space-y-1">
                  <p className="font-medium text-severity-warning" data-selectable>
                    Measured zT ({fmtVal(spb.best.measured_zt)}) exceeds the single-band ceiling
                    {spb.best.zt_ceiling != null
                      ? ` (${fmtVal(spb.best.zt_ceiling)})`
                      : ""}{" "}
                    at |S| = {fmtVal(spb.best.measured_seebeck_uv_k)} µV/K.
                  </p>
                  <p className="text-secondary" data-selectable>
                    A single parabolic band cannot reach this zT at such a low Seebeck. Expect
                    multi-band transport, or check the Seebeck data/units before trusting a
                    physics-informed target.
                  </p>
                </div>
              )}
            </section>
          )}

          {running && (
            <AnalysisLoader title="Running the optimization" stages={OPTIMIZE_STAGES} />
          )}

          {/* Verdict + chart */}
          {result && (
            <section className="space-y-3">
              {result.quality_flags.length > 0 && (
                <div className="rounded-lg border border-severity-warning bg-[color-mix(in_srgb,var(--latos-severity-warning)_12%,transparent)] px-5 py-4 text-sm">
                  <div className="font-medium text-severity-warning">
                    ⚠ Data-quality warning — this run uses values flagged unreliable
                  </div>
                  <ul className="mt-2 space-y-1 text-secondary">
                    {result.quality_flags.map((f) => (
                      <li key={`${f.sample_name}:${f.variable}`} data-selectable>
                        <span className="font-medium text-primary">{f.sample_name}</span>{" "}
                        · {humanize(f.variable)} = {fmtVal(f.value)} — {f.reason}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-2 text-xs text-secondary">
                    The recommendation may be meaningless. Check the Hall reliability flag in the
                    sample's Analysis panel; fix or exclude the affected measurement, then re-run.
                  </div>
                </div>
              )}
              <div
                className={`rounded-lg border px-5 py-4 text-sm ${
                  result.converged ||
                  (result.max_ei < result.noise_threshold &&
                    result.reliability_level === "exploratory")
                    ? "border-[color:var(--latos-tech-eds)] bg-[color-mix(in_srgb,var(--latos-tech-eds)_10%,transparent)]"
                    : "border-accent bg-[color-mix(in_srgb,var(--latos-accent)_8%,transparent)]"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium">
                    {result.converged
                      ? "✓ Optimum reached"
                      : result.max_ei < result.noise_threshold &&
                          result.reliability_level === "exploratory"
                        ? "◑ Likely done — one optional check"
                        : "↑ Improvement still possible"}
                  </span>
                  {result.reliability_level !== "unknown" && (
                    <span
                      className={reliabilityChipClass(result.reliability_level)}
                      title={result.reliability_note}
                    >
                      {result.reliability_level}
                    </span>
                  )}
                </div>
                <div className="mt-1 text-secondary" data-selectable>
                  {result.verdict}
                </div>
                {/* How likely we are already done, as a number rather than a
                    yes/no. Stated as "under this model" on purpose: the
                    reliability grade beside it says how far to trust that.

                    Guarded because the sidecar is a separate process: in
                    development the UI hot-reloads while the Python server does
                    not, so a newer screen can be handed an older payload. A
                    missing field should cost this one line, not the screen. */}
                {Number.isFinite(result.epsilon) &&
                  Number.isFinite(result.prob_within_epsilon) && (
                    <div className="mt-1.5 text-secondary" data-selectable>
                      Under this model, the best measured point is within{" "}
                      <span className="font-medium text-primary">
                        {fmtVal(result.epsilon)}
                      </span>{" "}
                      of the optimum with probability{" "}
                      <span className="font-medium text-primary">
                        {(result.prob_within_epsilon * 100).toFixed(0)}%
                      </span>
                      {result.epsilon_delta_met
                        ? ` (meets the ${((1 - result.delta) * 100).toFixed(0)}% bar).`
                        : ` (below the ${((1 - result.delta) * 100).toFixed(0)}% bar).`}
                    </div>
                  )}
                {result.n_unreliable > 0 && (
                  <div className="mt-1 text-xs text-secondary" data-selectable>
                    {result.n_unreliable} measurement
                    {result.n_unreliable === 1 ? " was" : "s were"} down-weighted in the fit
                    rather than dropped
                    {/* Two independent sources feed this: the physics checks and
                        the researcher's own call. Naming which is which keeps a
                        ticked box from looking like the tool's own verdict.
                        Guarded like the epsilon line — an older sidecar omits
                        the field entirely. */}
                    {distrustedCount === 0
                      ? ` because a physics check rejected ${result.n_unreliable === 1 ? "it" : "them"}.`
                      : distrustedCount >= result.n_unreliable
                        ? ` because you marked ${result.n_unreliable === 1 ? "it" : "them"} untrusted.`
                        : `; ${distrustedCount} because you marked ${distrustedCount === 1 ? "it" : "them"} untrusted, the rest by a physics check.`}
                  </div>
                )}
                {result.reliability_note && (
                  <div className="mt-1.5 text-xs text-secondary" data-selectable>
                    {result.reliability_note}
                  </div>
                )}
              </div>
              <div className="rounded-lg border border-edge bg-surface p-3">
                <ChartFrame basename="latos-optimization" label="optimization figure">
                  <OptimizeChart result={result} />
                </ChartFrame>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={freezing}
                  onClick={() => void freeze()}
                  className="rounded-md border border-accent px-4 py-2 text-sm font-medium text-accent transition enabled:hover:bg-[color-mix(in_srgb,var(--latos-accent)_10%,transparent)] disabled:opacity-40"
                >
                  {freezing ? "Freezing…" : "Freeze recommendation (pre-register)"}
                </button>
                <span className="text-xs text-secondary">
                  Commits the frozen model config + predicted interval before you make the
                  sample — so the recommendation is prospective.
                </span>
              </div>

              {frozen && (
                <div className="rounded-lg border border-[color:var(--latos-tech-eds)] bg-[color-mix(in_srgb,var(--latos-tech-eds)_10%,transparent)] px-5 py-4 text-sm">
                  <div className="font-medium">✓ Pre-registration recorded</div>
                  <ul className="mt-2 space-y-1 text-secondary">
                    <li>
                      Recommended {varLabel}:{" "}
                      <span className="font-medium text-primary">
                        {frozen.recommendation.x.toFixed(3)}
                      </span>
                    </li>
                    <li>
                      Predicted {target}:{" "}
                      <span className="font-medium text-primary">
                        {frozen.recommendation.predicted_mean.toFixed(3)}
                      </span>{" "}
                      (95% predictive [
                      {frozen.recommendation.predictive_interval_95[0].toFixed(3)},{" "}
                      {frozen.recommendation.predictive_interval_95[1].toFixed(3)}])
                    </li>
                    <li>Prior best: {frozen.prior_best.toFixed(3)}</li>
                    <li>
                      Kernel robustness:{" "}
                      <span className="font-medium text-primary">
                        {frozen.robustness_stable
                          ? "stable"
                          : "UNSTABLE — data may be too sparse"}
                      </span>
                    </li>
                    <li>
                      Reliability:{" "}
                      <span className={reliabilityChipClass(frozen.reliability_level)}>
                        {frozen.reliability_level}
                      </span>{" "}
                      <span className="text-xs">(recorded in the pre-registration)</span>
                    </li>
                    <li className="text-xs" data-selectable>
                      Saved to {frozen.path}
                    </li>
                  </ul>
                </div>
              )}
            </section>
          )}

          {/* Two-variable verdict + surface. Separate from the one-variable
              block above because almost nothing transfers: there is no curve,
              the recommendation is a coordinate, and the reliability grade is
              carried by coverage of a plane rather than a count of points. */}
          {ndResult && (
            <section className="space-y-3">
              {ndResult.quality_flags.length > 0 && (
                <div className="rounded-lg border border-severity-warning bg-[color-mix(in_srgb,var(--latos-severity-warning)_12%,transparent)] px-5 py-4 text-sm">
                  <div className="font-medium text-severity-warning">
                    ⚠ Data-quality warning — this run uses values flagged unreliable
                  </div>
                  <ul className="mt-2 space-y-1 text-secondary">
                    {ndResult.quality_flags.map((f) => (
                      <li key={`${f.sample_name}:${f.variable}`} data-selectable>
                        {f.sample_name} — {f.reason}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="rounded-lg border border-edge bg-surface px-5 py-4 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className={reliabilityChipClass(ndResult.reliability_level)}>
                    {ndResult.reliability_level}
                  </span>
                  <span className="text-xs text-secondary">
                    {ndResult.points.length} samples over {ndResult.axes.length} variables
                  </span>
                </div>
                <p className="mt-2 text-primary" data-selectable>
                  {ndResult.verdict}
                </p>
                {ndResult.reliability_note && (
                  <p className="mt-1.5 text-xs text-secondary" data-selectable>
                    {ndResult.reliability_note}
                  </p>
                )}
                {/* Two axes make "how many points" the wrong question. This is
                    the right one: how big is the largest region nothing was
                    measured in, against the largest this grade tolerates. */}
                {ndResult.fill_limit > 0 && (
                  <p className="mt-1.5 text-xs text-secondary" data-selectable>
                    Largest unmeasured gap: {fmtVal(ndResult.fill_distance)} (a{" "}
                    {ndResult.reliability_level === "exploratory" ? "better" : "calibrated"} grade
                    needs {fmtVal(ndResult.fill_limit)} or less, in units where each variable&rsquo;s
                    range is 4). Adding a variable enlarges the space faster than it adds points.
                  </p>
                )}
                {ndResult.n_dropped_for_missing_axis > 0 && (
                  <p className="mt-1.5 text-xs text-severity-warning" data-selectable>
                    {ndResult.n_dropped_for_missing_axis} sample
                    {ndResult.n_dropped_for_missing_axis === 1 ? " was" : "s were"} left out: they
                    have {humanize(inputVar)} and the property, but no {humanize(secondVar)} value.
                  </p>
                )}
                {/* An axis whose values look like names rather than quantities.
                    Shown rather than blocked: the engine cannot tell a gas
                    written 0/1/2 from an anneal at 1, 2 and 3 hours, and only
                    the person who entered the number knows which it is. */}
                {(ndResult.axis_warnings ?? []).map((warning) => (
                  <p
                    key={warning}
                    className="mt-1.5 text-xs text-severity-warning"
                    data-selectable
                  >
                    {warning}
                  </p>
                ))}
              </div>

              <div className="rounded-lg border border-edge bg-surface p-3">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  {(
                    [
                      ["mean", "Predicted"],
                      ["sd", "Uncertainty"],
                      ["ei", "Opportunity"],
                    ] as [SurfaceMode, string][]
                  ).map(([m, label]) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setSurfaceMode(m)}
                      className={`rounded-md border px-3 py-1 text-xs transition ${
                        surfaceMode === m
                          ? "border-accent text-accent"
                          : "border-edge text-secondary hover:border-accent"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <ChartFrame basename="latos-optimization-surface" label="optimization surface">
                  <OptimizeSurfaceChart result={ndResult} mode={surfaceMode} />
                </ChartFrame>
              </div>

              {/* The ARD read. This is the answer a one-variable run cannot
                  give: which knob the property is actually sensitive to. */}
              <div className="rounded-lg border border-edge bg-surface px-5 py-4 text-sm">
                <div className="text-xs font-semibold uppercase tracking-wide text-secondary">
                  How much each variable matters
                </div>
                <ul className="mt-2 space-y-1 text-secondary">
                  {ndResult.axes.map((a) => (
                    <li key={a.name} data-selectable>
                      <span className="font-medium text-primary">{humanize(a.name)}</span> —
                      searched {fmtVal(a.low)} to {fmtVal(a.high)}; recommended{" "}
                      <span className="font-medium text-primary">
                        {fmtVal(
                          ndResult.recommendation.x[
                            ndResult.axes.findIndex((x) => x.name === a.name)
                          ],
                        )}
                      </span>
                      {a.pinned_at === "high"
                        ? " — the model found no structure along this axis over this range, so it is not driving the result."
                        : a.pinned_at === "low"
                          ? " — the model wants finer detail along this axis than it is allowed to fit, so treat the shape here as under-resolved."
                          : ""}
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-xs text-secondary">
                  Predicted {ndResult.target_property}:{" "}
                  <span className="font-medium text-primary">
                    {fmtVal(ndResult.recommendation.predicted_mean)}
                  </span>{" "}
                  (95% predictive [{fmtVal(ndResult.recommendation.predictive_interval_95[0])},{" "}
                  {fmtVal(ndResult.recommendation.predictive_interval_95[1])}]).
                </p>
              </div>

              <p className="text-xs text-secondary">
                Pre-registration records one variable at a time, so freezing is not offered for a
                two-variable run. To commit a prospective prediction, run the axis you intend to
                change on its own.
              </p>
            </section>
          )}

          {/* Controls */}
          <section className="flex flex-wrap items-end gap-4 rounded-lg border border-edge bg-surface px-5 py-4">
            <label className="text-sm">
              <span className="mb-1 block text-secondary">Synthesis variable</span>
              <input
                list="latos-known-vars"
                value={inputVar}
                onChange={(e) => chooseVar(e.target.value)}
                placeholder="e.g. doping_pct, etching_time_h"
                className="w-52 rounded-md border border-edge bg-surface px-2 py-1.5 outline-none focus:border-accent"
                data-selectable
              />
              <datalist id="latos-known-vars">
                {knownVars.map((v) => (
                  <option key={v} value={v} />
                ))}
              </datalist>
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-secondary">
                Second variable{" "}
                <span
                  className="cursor-help text-xs"
                  title="Optional. With two variables Latos fits a surface instead of a curve and can see how the axes interact — which is the part a one-variable run cannot show."
                >
                  (optional)
                </span>
              </span>
              <select
                value={secondVar}
                onChange={(e) => chooseSecondVar(e.target.value)}
                className="w-52 rounded-md border border-edge bg-surface px-2 py-1.5 outline-none focus:border-accent"
              >
                <option value="">— one variable —</option>
                {knownVars
                  .filter((v) => v !== inputVar.trim())
                  .map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-secondary">Objective</span>
              <select
                value={objective}
                onChange={(e) => setObjective(e.target.value as Objective)}
                className="rounded-md border border-edge bg-surface px-2 py-1.5 outline-none focus:border-accent"
              >
                <option value="maximize">Maximize</option>
                <option value="minimize">Minimize</option>
                <option value="target">Reach a value</option>
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block text-secondary">Property</span>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="rounded-md border border-edge bg-surface px-2 py-1.5 outline-none focus:border-accent"
              >
                {targets.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            {objective === "target" && (
              <label className="text-sm">
                <span className="mb-1 block text-secondary">Target value</span>
                <input
                  type="number"
                  step="any"
                  value={targetValue}
                  onChange={(e) => setTargetValue(e.target.value)}
                  placeholder="e.g. 1.4"
                  className="w-28 rounded-md border border-edge bg-surface px-2 py-1.5 outline-none focus:border-accent"
                  data-selectable
                />
              </label>
            )}
            {target === "zT (derived)" && (
              <label className="text-sm">
                <span className="mb-1 block text-secondary">At temperature (K, optional)</span>
                <input
                  type="number"
                  step="any"
                  value={atTempK}
                  onChange={(e) => setAtTempK(e.target.value)}
                  placeholder="peak"
                  className="w-32 rounded-md border border-edge bg-surface px-2 py-1.5 outline-none focus:border-accent"
                  data-selectable
                />
              </label>
            )}
            <button
              type="button"
              disabled={running || !target || !inputVar || usableCount < 3 || targetValueMissing}
              onClick={() => void run()}
              className="rounded-md bg-accent px-5 py-2 font-medium text-white transition enabled:hover:brightness-110 disabled:opacity-40"
            >
              {running ? "Running…" : twoAxis ? "Run optimization (2 variables)" : "Run optimization"}
            </button>
            {(!inputVar || usableCount < 3 || targetValueMissing) && (
              <span className="text-xs text-secondary">
                {!inputVar
                  ? "Name the variable you want to optimize over."
                  : targetValueMissing
                    ? "Enter the value to aim for."
                    : twoAxis
                      ? `Only ${pairedCount} sample${pairedCount === 1 ? " has" : "s have"} both ${varLabel} and ${humanize(secondVar)}. A second axis only counts samples that have both — at least 3 are needed.`
                      : `Enter ${varLabel} for at least 3 samples.`}
              </span>
            )}
          </section>

          {/* Input table */}
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-secondary">
              Synthesis inputs
            </h2>
            {isMeasured ? (
              <p className="text-xs text-secondary">
                <span className="font-medium">{varLabel}</span> is a measured quantity
                (from the instruments), used as a common physical axis — values are
                read-only. Useful when samples share no synthesis knob.
              </p>
            ) : (
              <p className="text-xs text-secondary">
                Enter the <span className="font-medium">{varLabel}</span> you used for each
                sample — the knob Latos optimizes. It can be any variable you tuned (doping,
                etching time, annealing temperature, …), or pick a measured axis such as the
                Hall carrier concentration from the list. Tip: drop a{" "}
                <span className="font-medium">synthesis.csv</span> (sample, variable columns)
                next to your raw files and these values fill in automatically at ingest.
              </p>
            )}
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-edge text-left text-secondary">
                  <th className="py-2 font-medium">Sample</th>
                  <th className="py-2 font-medium">Measurements</th>
                  <th className="py-2 font-medium">{varLabel}</th>
                  {twoAxis && (
                    <th className="py-2 font-medium">{humanize(secondVar)}</th>
                  )}
                  <th
                    className="py-2 font-medium"
                    title="Tick a sample you have reason to doubt. It stays in the dataset, fitted with larger assumed noise."
                  >
                    Don&rsquo;t trust
                  </th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s) => (
                  <tr key={s.id} className="border-b border-edge">
                    <td className="py-2 font-medium" data-selectable>
                      {s.name}
                    </td>
                    <td className="py-2 text-secondary">{s.measurements.length}</td>
                    <td className="py-2">
                      <input
                        type="number"
                        step="any"
                        disabled={!inputVar || isMeasured}
                        value={inputs[s.id] ?? ""}
                        onChange={(e) =>
                          setInputs((d) => ({ ...d, [s.id]: e.target.value }))
                        }
                        onBlur={(e) => saveInput(s.id, e.target.value)}
                        placeholder="—"
                        title={isMeasured ? "Measured value (read-only)" : undefined}
                        className="w-28 rounded-md border border-edge bg-surface px-2 py-1 outline-none focus:border-accent disabled:opacity-60"
                      />
                    </td>
                    {twoAxis && (
                      <td className="py-2">
                        <input
                          type="number"
                          step="any"
                          disabled={secondIsMeasured}
                          value={secondInputs[s.id] ?? ""}
                          onChange={(e) =>
                            setSecondInputs((d) => ({ ...d, [s.id]: e.target.value }))
                          }
                          onBlur={(e) => saveSecond(s.id, e.target.value)}
                          placeholder="—"
                          title={secondIsMeasured ? "Measured value (read-only)" : undefined}
                          className="w-28 rounded-md border border-edge bg-surface px-2 py-1 outline-none focus:border-accent disabled:opacity-60"
                        />
                      </td>
                    )}
                    <td className="py-2">
                      <input
                        type="checkbox"
                        checked={distrusted.has(s.id)}
                        onChange={(e) => toggleDistrust(s.id, e.target.checked)}
                        title={`Down-weight ${s.name} in the fit — it is not removed`}
                        className="h-4 w-4 accent-[color:var(--latos-severity-warning)]"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-secondary">
              Tick <span className="font-medium">Don&rsquo;t trust</span> for a sample you
              have reason to doubt — the powder clumped, the sonicator was skipped, the
              sensor was knocked. Latos&rsquo;s physics checks only see the exported
              numbers, so they cannot know this. A ticked sample is{" "}
              <span className="font-medium">not deleted</span>: it is fitted with larger
              assumed noise, so the model still sees it but stops chasing it. Re-run to
              apply.
            </p>
          </section>

          {/* Loop-closer: frozen predictions and their validated outcomes */}
          {preregs.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-secondary">
                Pre-registrations & outcomes
              </h2>
              <p className="text-xs text-secondary">
                Each frozen prediction, and — once you have synthesized and measured the
                recommended sample — how the real result scored against it. Entering an
                outcome writes it beside the prediction, so the closed loop stays auditable.
              </p>

              {/* Convergence read from OUTSIDE the model. Everything in the
                  verdict above is the model judging its own fit, so it all
                  fails together when the fit is wrong. This compares the
                  frozen records on disk, which no later run can retune. */}
              {drift
                .filter((d) => d.n_freezes > 1)
                .map((d) => (
                  <div
                    key={`${d.property_name}:${d.input_variable}:${d.direction}`}
                    className={`rounded-lg border px-5 py-4 text-sm ${
                      d.settled
                        ? "border-[color:var(--latos-tech-eds)] bg-[color-mix(in_srgb,var(--latos-tech-eds)_10%,transparent)]"
                        : "border-edge bg-surface"
                    }`}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium">
                        {d.settled === true
                          ? "✓ Recommendation has stopped moving"
                          : "↔ Recommendation is still moving"}
                      </span>
                      <span className="text-secondary">
                        · {humanize(d.input_variable)} → {d.property_name} ·{" "}
                        {d.n_freezes} freezes
                      </span>
                    </div>
                    <div className="mt-1 text-secondary" data-selectable>
                      {d.note}
                    </div>
                    <div className="mt-1 text-xs text-secondary" data-selectable>
                      Moves:{" "}
                      {d.steps
                        .map(
                          (s) =>
                            `${fmtVal(s.from_x)} → ${fmtVal(s.to_x)} (${fmtVal(s.distance)})`,
                        )
                        .join(" · ")}
                    </div>
                  </div>
                ))}
              {preregs.map((pr) => (
                <div
                  key={pr.path}
                  className="rounded-lg border border-edge bg-surface px-5 py-4 text-sm"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">
                      {pr.direction === "minimize" ? "Minimize" : "Maximize"}{" "}
                      {pr.property_name}
                    </span>
                    <span className="text-secondary">
                      · recommended {humanize(pr.input_variable)} ={" "}
                      <span className="font-medium text-primary">
                        {pr.recommended_x.toFixed(3)}
                      </span>
                    </span>
                    {pr.reliability_level !== "unknown" && (
                      <span className={reliabilityChipClass(pr.reliability_level)}>
                        {pr.reliability_level}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-secondary">
                    Predicted{" "}
                    <span className="font-medium text-primary">
                      {pr.predicted_mean.toFixed(3)}
                    </span>{" "}
                    (95% predictive [{pr.predictive_interval_95[0].toFixed(3)},{" "}
                    {pr.predictive_interval_95[1].toFixed(3)}]) · prior best{" "}
                    {pr.prior_best.toFixed(3)} ·{" "}
                    <span className="text-xs">
                      {new Date(pr.created_at).toLocaleString()}
                    </span>
                  </div>

                  {pr.outcome ? (
                    <div
                      className={`mt-3 rounded-md border px-4 py-3 ${
                        pr.outcome.within_interval
                          ? "border-[color:var(--latos-tech-eds)] bg-[color-mix(in_srgb,var(--latos-tech-eds)_10%,transparent)]"
                          : "border-severity-warning bg-[color-mix(in_srgb,var(--latos-severity-warning)_10%,transparent)]"
                      }`}
                    >
                      <div className="flex flex-wrap gap-4 font-medium">
                        <span>
                          Measured{" "}
                          <span className="text-primary">
                            {pr.outcome.measured.toFixed(3)}
                          </span>
                        </span>
                        <span>
                          {pr.outcome.within_interval
                            ? "✓ within predicted interval (calibrated)"
                            : "✗ outside predicted interval (over-confident)"}
                        </span>
                        <span>
                          {pr.outcome.improved
                            ? "✓ improved on prior best"
                            : "— no improvement"}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-secondary" data-selectable>
                        {pr.outcome.summary}
                      </div>
                    </div>
                  ) : (
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <label className="text-xs text-secondary">
                        Measured {pr.property_name}:
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={measuredInputs[pr.path] ?? ""}
                        onChange={(e) =>
                          setMeasuredInputs((d) => ({ ...d, [pr.path]: e.target.value }))
                        }
                        placeholder="synthesized result"
                        className="w-40 rounded-md border border-edge bg-surface px-2 py-1 outline-none focus:border-accent"
                        data-selectable
                      />
                      <button
                        type="button"
                        disabled={
                          validatingPath === pr.path ||
                          !Number.isFinite(Number.parseFloat(measuredInputs[pr.path] ?? ""))
                        }
                        onClick={() => void validate(pr.path)}
                        className="rounded-md border border-accent px-3 py-1 text-xs font-medium text-accent transition enabled:hover:bg-[color-mix(in_srgb,var(--latos-accent)_10%,transparent)] disabled:opacity-40"
                      >
                        {validatingPath === pr.path ? "Validating…" : "Validate outcome"}
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
