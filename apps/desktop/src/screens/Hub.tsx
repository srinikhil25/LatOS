/** Hub screen — project summary after ingestion (RB4 skeleton).
 *
 * Proves the full loop: folder → sidecar ingestion → summary + samples
 * tree rendered from the API. The parity screens (Samples detail,
 * Analysis) build on exactly these endpoints.
 */

import { useEffect, useState } from "react";
import {
  getProject,
  getSamples,
  type ProjectSummary,
  type SampleSummary,
} from "../lib/api";
import { TechniqueChip } from "../components/TechniqueChip";

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-edge bg-surface px-5 py-4">
      <div className="text-2xl font-semibold">{value}</div>
      <div className="text-xs text-secondary">{label}</div>
    </div>
  );
}

export function Hub({
  onBack,
  onOpenSamples,
}: {
  onBack: () => void;
  onOpenSamples: () => void;
}) {
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getProject(), getSamples()])
      .then(([summary, tree]) => {
        setProject(summary);
        setSamples(tree);
      })
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  if (error) {
    return (
      <div className="p-10">
        <p className="text-severity-error" data-selectable>
          {error}
        </p>
      </div>
    );
  }
  if (!project) {
    return <div className="p-10 text-secondary">Loading project…</div>;
  }

  return (
    <div className="mx-auto max-w-5xl space-y-8 px-10 py-10">
      <header className="space-y-1">
        <h1 className="text-3xl font-semibold tracking-tight">{project.name}</h1>
        <p className="text-sm text-secondary">
          {project.parsed} parsed · {project.cached} cached · {project.failed}{" "}
          failed · {project.unclassified} unclassified
        </p>
      </header>

      <section className="grid grid-cols-3 gap-3">
        <Stat label="Samples" value={project.samples} />
        <Stat label="Measurements" value={project.measurements} />
        <Stat label="Techniques" value={project.techniques} />
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-secondary">
            Samples
          </h2>
          <button
            type="button"
            onClick={onOpenSamples}
            className="rounded-md border border-edge px-3 py-1.5 text-sm font-medium transition hover:border-accent"
          >
            Browse all →
          </button>
        </div>
        <ul className="space-y-2">
          {samples.map((sample) => (
            <li key={sample.id}>
              <button
                type="button"
                onClick={onOpenSamples}
                className="w-full rounded-md border border-edge bg-surface px-4 py-3 text-left transition hover:border-accent"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium" data-selectable>
                    {sample.name}
                  </span>
                  <span className="text-xs text-secondary">
                    {sample.measurements.length} measurement
                    {sample.measurements.length === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {[...new Set(sample.measurements.map((m) => m.technique))].map(
                    (technique) => (
                      <TechniqueChip key={technique} technique={technique} />
                    ),
                  )}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <button
        type="button"
        onClick={onBack}
        className="text-sm text-secondary underline-offset-4 hover:underline"
      >
        ← Open a different folder
      </button>
    </div>
  );
}
