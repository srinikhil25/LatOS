/** Review & Confirm — the human-verification gate (RB9).
 *
 * After ingestion, Latos has *guessed* every file's sample and
 * technique. Those guesses can be wrong (folder messiness, over-merged
 * doped samples, STEM-as-TEM). Nothing downstream runs until a person
 * verifies and confirms here. Edits available: rename a sample, change
 * a measurement's technique, merge samples, split measurements into a
 * new sample. Any edit returns the project to NEEDS_REVIEW.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  confirmProject,
  getAnomalies,
  getMergeSuggestions,
  getProject,
  getSamples,
  mergeSamples,
  renameSample,
  reopenProject,
  setMeasurementTechnique,
  splitMeasurements,
  TECHNIQUES,
  type MergeSuggestion,
  type SampleAnomaly,
  type SampleSummary,
} from "../lib/api";
import { TechniqueChip, techniqueLabel } from "../components/TechniqueChip";

type Status = "needs_review" | "confirmed";

export function Review({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [status, setStatus] = useState<Status>("needs_review");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Suggested merges (suggest-only) + the ones the user has dismissed
  // this session (keyed by `${target}|${source}`).
  const [suggestions, setSuggestions] = useState<MergeSuggestion[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  // Flagged samples (date/generic names, session buckets) + dismissed ids.
  const [anomalies, setAnomalies] = useState<SampleAnomaly[]>([]);
  const [dismissedAnomalies, setDismissedAnomalies] = useState<Set<string>>(
    new Set(),
  );

  // Selection state (two independent sets).
  const [selSamples, setSelSamples] = useState<Set<string>>(new Set());
  const [selMeas, setSelMeas] = useState<Set<string>>(new Set());

  // Inline rename + split-name drafts.
  const [editingSample, setEditingSample] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [splitName, setSplitName] = useState("");

  const reload = useCallback(async () => {
    const [project, tree, sugg, anoms] = await Promise.all([
      getProject(),
      getSamples(),
      getMergeSuggestions(),
      getAnomalies(),
    ]);
    setStatus(project.review_status);
    setSamples(tree);
    setSuggestions(sugg);
    setAnomalies(anoms);
    setSelSamples(new Set());
    setSelMeas(new Set());
    setSplitName("");
  }, []);

  const suggestionKey = (s: MergeSuggestion) => `${s.target_id}|${s.source_id}`;

  useEffect(() => {
    reload().catch((e: unknown) =>
      setError(e instanceof Error ? e.message : String(e)),
    );
  }, [reload]);

  /** Run an edit, then reload; surface any error. */
  const run = useCallback(
    async (action: () => Promise<unknown>) => {
      setBusy(true);
      setError(null);
      try {
        await action();
        await reload();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [reload],
  );

  const orderedSelectedSamples = useMemo(
    () => samples.filter((s) => selSamples.has(s.id)),
    [samples, selSamples],
  );

  const visibleSuggestions = useMemo(
    () => suggestions.filter((s) => !dismissed.has(suggestionKey(s))),
    [suggestions, dismissed],
  );

  const visibleAnomalies = useMemo(
    () => anomalies.filter((a) => !dismissedAnomalies.has(a.sample_id)),
    [anomalies, dismissedAnomalies],
  );

  const toggle = (set: Set<string>, id: string): Set<string> => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  };

  const commitRename = (sampleId: string) => {
    const name = draftName.trim();
    setEditingSample(null);
    const current = samples.find((s) => s.id === sampleId)?.name;
    if (name && name !== current) void run(() => renameSample(sampleId, name));
  };

  const doMerge = () => {
    const [target, ...sources] = orderedSelectedSamples;
    void run(() => mergeSamples(sources.map((s) => s.id), target.id));
  };

  const acceptSuggestion = (s: MergeSuggestion) => {
    // Fold the noisier `source` spelling into the cleaner `target`.
    void run(() => mergeSamples([s.source_id], s.target_id));
  };

  const dismissSuggestion = (s: MergeSuggestion) => {
    setDismissed((prev) => new Set(prev).add(suggestionKey(s)));
  };

  const dismissAnomaly = (a: SampleAnomaly) => {
    setDismissedAnomalies((prev) => new Set(prev).add(a.sample_id));
  };

  const doSplit = () => {
    const name = splitName.trim();
    if (!name) return;
    void run(() => splitMeasurements([...selMeas], name));
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-edge px-6 py-3">
        <button
          type="button"
          onClick={onBack}
          className="text-sm text-secondary underline-offset-4 hover:underline"
        >
          ← Hub
        </button>
        <h1 className="text-sm font-semibold uppercase tracking-wide text-secondary">
          Review &amp; Confirm
        </h1>
      </div>

      {/* Status banner */}
      {status === "needs_review" ? (
        <div className="flex items-center gap-2 border-b border-edge bg-[color-mix(in_srgb,var(--latos-severity-warning)_12%,transparent)] px-6 py-2.5 text-sm">
          <span className="text-severity-warning">⚠</span>
          <span>
            Verify that every file is assigned to the right sample and technique.
            Analysis is locked until you confirm.
          </span>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2 border-b border-edge bg-[color-mix(in_srgb,var(--latos-tech-eds)_12%,transparent)] px-6 py-2.5 text-sm">
          <span>
            <span className="text-[color:var(--latos-tech-eds)]">✓</span> Confirmed —
            analysis is unlocked.
          </span>
          <button
            type="button"
            onClick={() => void run(reopenProject)}
            className="rounded-md border border-edge px-2.5 py-1 text-xs hover:border-accent"
          >
            Reopen for editing
          </button>
        </div>
      )}

      {error && (
        <div className="border-b border-edge px-6 py-2 text-sm text-severity-error" data-selectable>
          {error}
        </div>
      )}

      {/* Action bar */}
      <div className="flex flex-wrap items-center gap-3 border-b border-edge px-6 py-2">
        <button
          type="button"
          disabled={orderedSelectedSamples.length < 2 || busy}
          onClick={doMerge}
          className="rounded-md border border-edge px-3 py-1.5 text-sm enabled:hover:border-accent disabled:opacity-40"
        >
          Merge {orderedSelectedSamples.length || ""} samples
          {orderedSelectedSamples.length >= 2 &&
            ` → ${orderedSelectedSamples[0].name}`}
        </button>

        <div className="flex items-center gap-2">
          <input
            value={splitName}
            onChange={(e) => setSplitName(e.target.value)}
            placeholder="New sample name"
            className="w-40 rounded-md border border-edge bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent"
            data-selectable
          />
          <button
            type="button"
            disabled={selMeas.size === 0 || !splitName.trim() || busy}
            onClick={doSplit}
            className="rounded-md border border-edge px-3 py-1.5 text-sm enabled:hover:border-accent disabled:opacity-40"
          >
            Move {selMeas.size || ""} to new sample
          </button>
        </div>
      </div>

      {/* Suggested merges (suggest-only — Latos never merges these on its own) */}
      {status === "needs_review" && visibleSuggestions.length > 0 && (
        <div className="border-b border-edge bg-[color-mix(in_srgb,var(--latos-accent)_6%,transparent)] px-6 py-3">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-secondary">
            Possible duplicate samples — {visibleSuggestions.length} to review
          </h2>
          <ul className="space-y-1.5">
            {visibleSuggestions.map((s) => (
              <li
                key={suggestionKey(s)}
                className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-edge bg-surface px-3 py-2 text-sm"
              >
                <span className="font-medium" data-selectable>
                  {s.source_name}
                </span>
                <span className="text-secondary">→</span>
                <span className="font-medium" data-selectable>
                  {s.target_name}
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-xs ${
                    s.confidence === "high"
                      ? "bg-[color-mix(in_srgb,var(--latos-tech-eds)_18%,transparent)] text-[color:var(--latos-tech-eds)]"
                      : "bg-[color-mix(in_srgb,var(--latos-severity-warning)_18%,transparent)] text-severity-warning"
                  }`}
                >
                  {Math.round(s.score)}% · {s.confidence}
                </span>
                <span className="text-xs text-secondary">{s.reason}</span>
                <span className="ml-auto flex gap-2">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => acceptSuggestion(s)}
                    className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-white transition enabled:hover:brightness-110 disabled:opacity-40"
                  >
                    Merge
                  </button>
                  <button
                    type="button"
                    onClick={() => dismissSuggestion(s)}
                    className="rounded-md border border-edge px-3 py-1 text-xs hover:border-accent"
                  >
                    Dismiss
                  </button>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Flagged samples (date/generic names, session buckets) */}
      {status === "needs_review" && visibleAnomalies.length > 0 && (
        <div className="border-b border-edge bg-[color-mix(in_srgb,var(--latos-severity-warning)_8%,transparent)] px-6 py-3">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-secondary">
            Needs attention — {visibleAnomalies.length} sample
            {visibleAnomalies.length === 1 ? "" : "s"} may be mislabeled
          </h2>
          <ul className="space-y-1.5">
            {visibleAnomalies.map((a) => (
              <li
                key={a.sample_id}
                className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-edge bg-surface px-3 py-2 text-sm"
              >
                <span className="text-severity-warning">⚠</span>
                <span className="font-medium" data-selectable>
                  {a.sample_name}
                </span>
                <span className="rounded bg-[color-mix(in_srgb,var(--latos-severity-warning)_18%,transparent)] px-1.5 py-0.5 text-xs text-severity-warning">
                  {a.kind === "mixed_samples" ? "session folder?" : "not a sample?"}
                </span>
                <span className="text-xs text-secondary">{a.message}</span>
                <button
                  type="button"
                  onClick={() => dismissAnomaly(a)}
                  className="ml-auto rounded-md border border-edge px-3 py-1 text-xs hover:border-accent"
                >
                  Dismiss
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Tree */}
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {samples.map((sample) => (
          <div key={sample.id} className="mb-3 rounded-md border border-edge bg-surface">
            <div className="flex items-center gap-2 px-3 py-2">
              <input
                type="checkbox"
                checked={selSamples.has(sample.id)}
                onChange={() => setSelSamples((s) => toggle(s, sample.id))}
              />
              {editingSample === sample.id ? (
                <input
                  autoFocus
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  onBlur={() => commitRename(sample.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename(sample.id);
                    if (e.key === "Escape") setEditingSample(null);
                  }}
                  className="rounded border border-accent bg-surface px-1.5 py-0.5 font-medium outline-none"
                />
              ) : (
                <button
                  type="button"
                  onClick={() => {
                    setEditingSample(sample.id);
                    setDraftName(sample.name);
                  }}
                  className="font-medium hover:underline"
                  title="Click to rename"
                  data-selectable
                >
                  {sample.name}
                </button>
              )}
              <span className="text-xs text-secondary">
                {sample.measurements.length} measurement
                {sample.measurements.length === 1 ? "" : "s"}
              </span>
            </div>

            <ul className="border-t border-edge">
              {sample.measurements.map((m) => (
                <li
                  key={m.id}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm odd:bg-[color-mix(in_srgb,var(--latos-muted-surface)_50%,transparent)]"
                >
                  <input
                    type="checkbox"
                    checked={selMeas.has(m.id)}
                    onChange={() => setSelMeas((s) => toggle(s, m.id))}
                  />
                  <TechniqueChip technique={m.technique} />
                  <span className="min-w-0 flex-1 truncate" title={m.filename ?? m.id}>
                    {m.filename ?? m.instrument ?? m.id.slice(0, 8)}
                  </span>
                  <select
                    value={m.technique}
                    onChange={(e) =>
                      void run(() => setMeasurementTechnique(m.id, e.target.value))
                    }
                    className="rounded border border-edge bg-surface px-1.5 py-1 text-xs outline-none focus:border-accent"
                  >
                    {TECHNIQUES.map((t) => (
                      <option key={t} value={t}>
                        {techniqueLabel(t)}
                      </option>
                    ))}
                  </select>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Confirm footer */}
      <div className="flex items-center justify-between border-t border-edge px-6 py-3">
        <span className="text-sm text-secondary">
          {samples.length} samples ·{" "}
          {samples.reduce((n, s) => n + s.measurements.length, 0)} measurements
        </span>
        <button
          type="button"
          disabled={busy || status === "confirmed"}
          onClick={() => void run(async () => {
            await confirmProject();
            onBack();
          })}
          className="rounded-md bg-accent px-5 py-2 font-medium text-white transition enabled:hover:brightness-110 disabled:opacity-40"
        >
          {status === "confirmed" ? "Confirmed ✓" : "Confirm project"}
        </button>
      </div>
    </div>
  );
}
