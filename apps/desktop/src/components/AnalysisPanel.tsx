/** AnalysisPanel — runs the applicable analyzers on a measurement and
 * shows their results (band gap for UV-DRS, fitted peaks for XRD, …).
 *
 * Results are computed on demand by the sidecar; this panel just fetches
 * and renders them. Long arrays (e.g. 22 peak positions) are summarized
 * rather than dumped. Issues are shown with their severity colour so a
 * low-confidence or failed fit is obvious.
 */

import { useEffect, useState } from "react";
import { getMeasurementAnalysis, type AnalyzerResult } from "../lib/api";

const ARRAY_PREVIEW = 6;

function humanizeKey(key: string): string {
  return key.replace(/_/g, " ");
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  if (Array.isArray(value)) {
    const nums = value.map((v) => (typeof v === "number" ? v.toFixed(2) : String(v)));
    const head = nums.slice(0, ARRAY_PREVIEW).join(", ");
    return value.length > ARRAY_PREVIEW
      ? `${value.length} values · [${head}, …]`
      : `[${head}]`;
  }
  if (value === null) return "—";
  return String(value);
}

function issueClass(issue: string): string {
  if (issue.startsWith("error:")) return "text-severity-error";
  if (issue.startsWith("warning:")) return "text-severity-warning";
  return "text-secondary";
}

export function AnalysisPanel({ measurementId }: { measurementId: string }) {
  const [results, setResults] = useState<AnalyzerResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setResults(null);
    setError(null);
    getMeasurementAnalysis(measurementId)
      .then((r) => alive && setResults(r))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [measurementId]);

  if (error) {
    return (
      <section className="text-xs text-severity-error" data-selectable>
        Analysis failed: {error}
      </section>
    );
  }
  if (results === null) {
    return <section className="text-xs text-secondary">Running analysis…</section>;
  }
  // No analyzer applies to this technique (e.g. raw image, Hall).
  if (results.length === 0) return null;

  return (
    <section className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-secondary">
        Analysis
      </h3>
      {results.map((r) => (
        <div key={r.analyzer} className="rounded-lg border border-edge bg-surface p-4">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-accent">
            {r.analyzer}
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            {Object.entries(r.outputs).map(([key, value]) => (
              <div key={key} className="contents">
                <dt className="text-secondary">{humanizeKey(key)}</dt>
                <dd className="font-medium" data-selectable>
                  {formatValue(value)}
                </dd>
              </div>
            ))}
          </dl>
          {r.issues.length > 0 && (
            <ul className="mt-2 space-y-0.5 border-t border-edge pt-2 text-xs">
              {r.issues.map((issue) => (
                <li key={issue} className={issueClass(issue)} data-selectable>
                  {issue}
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </section>
  );
}
