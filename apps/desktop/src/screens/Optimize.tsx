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
  getOptimizeInputs,
  getOptimizeTargets,
  getParameters,
  getSamples,
  runOptimize,
  setSampleParameters,
  type FreezeResult,
  type InputVariableInfo,
  type Objective,
  type OptimizeOptions,
  type OptimizeResult,
  type SampleParams,
  type SampleSummary,
} from "../lib/api";
import { OptimizeChart } from "../components/OptimizeChart";

/** "etching_time_h" → "etching time h" for on-screen labels. */
const humanize = (v: string) => v.replace(/_/g, " ").trim() || "variable";

export function Optimize({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [params, setParams] = useState<SampleParams>({});
  const [inputInfos, setInputInfos] = useState<InputVariableInfo[]>([]);
  const [inputVar, setInputVar] = useState<string>("");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [objective, setObjective] = useState<Objective>("maximize");
  const [targetValue, setTargetValue] = useState<string>("");
  const [atTempK, setAtTempK] = useState<string>("");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [frozen, setFrozen] = useState<FreezeResult | null>(null);
  const [freezing, setFreezing] = useState(false);

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
  }, [valuesFor]);

  // Switching (or naming) the variable reloads its stored values into the table.
  const chooseVar = useCallback(
    (name: string) => {
      const v = name.trim();
      setInputVar(v);
      setInputs(valuesFor(v, params, inputInfos, samples));
    },
    [params, inputInfos, samples, valuesFor],
  );

  // Persist one cell, preserving the sample's OTHER variables. Measured
  // variables are instrument-derived and never written back.
  const saveInput = useCallback(
    (sampleId: string, raw: string) => {
      if (!inputVar || isMeasured) return;
      const rest = { ...(params[sampleId] ?? {}) };
      const value = Number.parseFloat(raw);
      if (Number.isFinite(value)) rest[inputVar] = value;
      else delete rest[inputVar];
      setParams((p) => ({ ...p, [sampleId]: rest }));
      void setSampleParameters(sampleId, rest);
    },
    [inputVar, isMeasured, params],
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
    setFrozen(null);
    try {
      setResult(await runOptimize(inputVar, target, opts));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [inputVar, target, opts]);

  const freeze = useCallback(async () => {
    setFreezing(true);
    setError(null);
    try {
      setFrozen(await freezeRecommendation(inputVar, target, opts));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setFreezing(false);
    }
  }, [inputVar, target, opts]);

  const filledCount = Object.values(inputs).filter((v) => v.trim() !== "").length;
  const varLabel = humanize(inputVar);

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

          {/* Verdict + chart */}
          {result && (
            <section className="space-y-3">
              <div
                className={`rounded-lg border px-5 py-4 text-sm ${
                  result.converged
                    ? "border-[color:var(--latos-tech-eds)] bg-[color-mix(in_srgb,var(--latos-tech-eds)_10%,transparent)]"
                    : "border-accent bg-[color-mix(in_srgb,var(--latos-accent)_8%,transparent)]"
                }`}
              >
                <div className="font-medium">
                  {result.converged ? "✓ Optimum reached" : "↑ Improvement still possible"}
                </div>
                <div className="mt-1 text-secondary" data-selectable>
                  {result.verdict}
                </div>
              </div>
              <div className="rounded-lg border border-edge bg-surface p-3">
                <OptimizeChart result={result} />
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
                    <li className="text-xs" data-selectable>
                      Saved to {frozen.path}
                    </li>
                  </ul>
                </div>
              )}
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
              disabled={running || !target || !inputVar || filledCount < 3 || targetValueMissing}
              onClick={() => void run()}
              className="rounded-md bg-accent px-5 py-2 font-medium text-white transition enabled:hover:brightness-110 disabled:opacity-40"
            >
              {running ? "Running…" : "Run optimization"}
            </button>
            {(!inputVar || filledCount < 3 || targetValueMissing) && (
              <span className="text-xs text-secondary">
                {!inputVar
                  ? "Name the variable you want to optimize over."
                  : targetValueMissing
                    ? "Enter the value to aim for."
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
                Hall carrier concentration from the list.
              </p>
            )}
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-edge text-left text-secondary">
                  <th className="py-2 font-medium">Sample</th>
                  <th className="py-2 font-medium">Measurements</th>
                  <th className="py-2 font-medium">{varLabel}</th>
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
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      </div>
    </div>
  );
}
