/** Optimize — the closed loop, in the app (BO4).
 *
 * The researcher enters their synthesis knob (doping %) per sample,
 * picks the property to maximize, and clicks Run. Latos fits the model
 * and shows: the curve, the recommended next experiment (★), and a
 * plain-language verdict. No CS jargon on screen.
 *
 * Reachable only once the project is CONFIRMED (the gate).
 */

import { useCallback, useEffect, useState } from "react";
import {
  freezeRecommendation,
  getOptimizeTargets,
  getParameters,
  getSamples,
  runOptimize,
  setSampleParameters,
  type FreezeResult,
  type OptimizeResult,
  type SampleSummary,
} from "../lib/api";
import { OptimizeChart } from "../components/OptimizeChart";

const INPUT_VAR = "doping_pct";

export function Optimize({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [doping, setDoping] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [frozen, setFrozen] = useState<FreezeResult | null>(null);
  const [freezing, setFreezing] = useState(false);

  useEffect(() => {
    Promise.all([getSamples(), getParameters(), getOptimizeTargets()])
      .then(([tree, params, props]) => {
        setSamples(tree);
        const init: Record<string, string> = {};
        for (const s of tree) {
          const v = params[s.id]?.[INPUT_VAR];
          init[s.id] = v === undefined ? "" : String(v);
        }
        setDoping(init);
        setTargets(props);
        // Prefer the provenance-tracked derived zT, then a raw zt column.
        const preferred =
          props.find((p) => p === "zT (derived)") ??
          (props.includes("zt") ? "zt" : props[0]);
        setTarget(preferred ?? "");
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const saveDoping = useCallback((sampleId: string, raw: string) => {
    const value = Number.parseFloat(raw);
    if (Number.isFinite(value)) {
      void setSampleParameters(sampleId, { [INPUT_VAR]: value });
    } else {
      void setSampleParameters(sampleId, {}); // clear
    }
  }, []);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    setFrozen(null);
    try {
      setResult(await runOptimize(INPUT_VAR, target));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [target]);

  const freeze = useCallback(async () => {
    setFreezing(true);
    setError(null);
    try {
      setFrozen(await freezeRecommendation(INPUT_VAR, target));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setFreezing(false);
    }
  }, [target]);

  const filledCount = Object.values(doping).filter((v) => v.trim() !== "").length;

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
                      Recommended {INPUT_VAR.replace("_", " ")}:{" "}
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
              <span className="mb-1 block text-secondary">Maximize</span>
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
            <button
              type="button"
              disabled={running || !target || filledCount < 3}
              onClick={() => void run()}
              className="rounded-md bg-accent px-5 py-2 font-medium text-white transition enabled:hover:brightness-110 disabled:opacity-40"
            >
              {running ? "Running…" : "Run optimization"}
            </button>
            {filledCount < 3 && (
              <span className="text-xs text-secondary">
                Enter the synthesis value for at least 3 samples.
              </span>
            )}
          </section>

          {/* Input table */}
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-secondary">
              Synthesis inputs
            </h2>
            <p className="text-xs text-secondary">
              Enter the {INPUT_VAR.replace("_", " ")} you used for each sample. This is the
              knob Latos optimizes.
            </p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-edge text-left text-secondary">
                  <th className="py-2 font-medium">Sample</th>
                  <th className="py-2 font-medium">Measurements</th>
                  <th className="py-2 font-medium">{INPUT_VAR.replace("_", " ")}</th>
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
                        value={doping[s.id] ?? ""}
                        onChange={(e) =>
                          setDoping((d) => ({ ...d, [s.id]: e.target.value }))
                        }
                        onBlur={(e) => saveDoping(s.id, e.target.value)}
                        placeholder="—"
                        className="w-28 rounded-md border border-edge bg-surface px-2 py-1 outline-none focus:border-accent"
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
