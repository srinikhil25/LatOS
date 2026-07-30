/** Samples screen — grouped tree (left) + measurement detail with plot (right).
 *
 * Replaces the old Qt "20 identical rows" problem: measurements are
 * grouped by technique under each sample, collapsed by default with a
 * count. Click a measurement → its arrays plot on the right.
 */

import { useEffect, useMemo, useState } from "react";
import {
  getMeasurementArrays,
  getSamples,
  IMAGE_TECHNIQUES,
  measurementImageUrl,
  type MeasurementArrays,
  type MeasurementSummary,
  type SampleSummary,
} from "../lib/api";
import { LinePlot } from "../components/LinePlot";
import { ChartFrame } from "../components/ChartFrame";
import { axisLabel } from "../lib/labels";
import { ImageViewer } from "../components/ImageViewer";
import { AnalysisPanel } from "../components/AnalysisPanel";
import { TechniqueChip, techniqueLabel } from "../components/TechniqueChip";

interface TechniqueGroup {
  technique: string;
  measurements: MeasurementSummary[];
}

function groupByTechnique(measurements: MeasurementSummary[]): TechniqueGroup[] {
  const map = new Map<string, MeasurementSummary[]>();
  for (const m of measurements) {
    const list = map.get(m.technique) ?? [];
    list.push(m);
    map.set(m.technique, list);
  }
  return [...map.entries()].map(([technique, ms]) => ({
    technique,
    measurements: ms,
  }));
}

function fileLabel(m: MeasurementSummary): string {
  return m.filename ?? m.instrument ?? m.id.slice(0, 8);
}

function ArrayDetail({ measurement }: { measurement: MeasurementSummary }) {
  const [data, setData] = useState<MeasurementArrays | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setData(null);
    setError(null);
    setLoading(true);
    getMeasurementArrays(measurement.id)
      .then((arrays) => setData(arrays))
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoading(false));
  }, [measurement.id]);

  // X/Y heuristic mirrors the core: first column vs second; lone column vs index.
  const plot = useMemo(() => {
    if (!data || data.names.length === 0) return null;
    if (data.names.length >= 2) {
      const [xName, yName] = data.names;
      return { x: data.arrays[xName], y: data.arrays[yName], xName, yName };
    }
    const yName = data.names[0];
    const y = data.arrays[yName];
    return { x: y.map((_, i) => i), y, xName: "index", yName };
  }, [data]);

  if (loading) return <p className="text-secondary">Loading arrays…</p>;
  if (error || !plot) {
    return (
      <div className="rounded-md border border-edge bg-muted p-4 text-sm text-secondary">
        No plottable data for this measurement.
        {error && (
          <span className="mt-1 block text-xs opacity-70" data-selectable>
            {error}
          </span>
        )}
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-edge bg-surface p-4">
      <ChartFrame basename="latos-trace" label="measurement trace">
        <LinePlot
          x={plot.x}
          y={plot.y}
          xLabel={axisLabel(plot.xName)}
          yLabel={axisLabel(plot.yName)}
        />
      </ChartFrame>
    </div>
  );
}

function formatFeature(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e4 || abs < 1e-3)) return value.toExponential(3);
  return value.toPrecision(4);
}

function MeasuredProperties({ features }: { features?: Record<string, number> }) {
  if (!features || Object.keys(features).length === 0) return null;
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-secondary">
        Measured properties
      </h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 rounded-lg border border-edge bg-surface p-4 text-sm">
        {Object.entries(features).map(([key, value]) => (
          <div key={key} className="contents">
            <dt className="text-secondary">{key.replace(/_/g, " ")}</dt>
            <dd className="font-medium" data-selectable>
              {formatFeature(value)}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function DetailPane({ measurement }: { measurement: MeasurementSummary | null }) {
  if (!measurement) {
    return (
      <div className="flex h-full items-center justify-center text-secondary">
        Select a measurement to see its details.
      </div>
    );
  }

  const isImage = IMAGE_TECHNIQUES.has(measurement.technique);

  return (
    <div className="space-y-5 p-8">
      <header className="space-y-1">
        <div className="flex items-center gap-2">
          <TechniqueChip technique={measurement.technique} />
          <h2 className="text-lg font-semibold" data-selectable>
            {fileLabel(measurement)}
          </h2>
        </div>
        {measurement.instrument && (
          <p className="text-sm text-secondary" data-selectable>
            {measurement.instrument}
          </p>
        )}
      </header>

      <MeasuredProperties features={measurement.features} />

      {isImage ? (
        <ImageViewer
          key={measurement.id}
          src={measurementImageUrl(measurement.id)}
          alt={fileLabel(measurement)}
        />
      ) : (
        <ArrayDetail measurement={measurement} />
      )}

      <AnalysisPanel measurementId={measurement.id} />
    </div>
  );
}

export function Samples({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<MeasurementSummary | null>(null);

  useEffect(() => {
    getSamples()
      .then((tree) => {
        setSamples(tree);
        // Open the first sample by default so the tree isn't all-collapsed.
        if (tree[0]) setExpanded(new Set([tree[0].id]));
      })
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  if (error) {
    return (
      <div className="p-10 text-severity-error" data-selectable>
        {error}
      </div>
    );
  }

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
          Samples
        </h1>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Left: tree */}
        <nav className="w-80 shrink-0 overflow-y-auto border-r border-edge p-3">
          {samples.map((sample) => {
            const groups = groupByTechnique(sample.measurements);
            const open = expanded.has(sample.id);
            return (
              <div key={sample.id} className="mb-1">
                <button
                  type="button"
                  onClick={() => toggle(sample.id)}
                  className="flex w-full items-center justify-between rounded-md px-2 py-2 text-left hover:bg-muted"
                >
                  <span className="flex items-center gap-1.5 font-medium">
                    <span className="text-secondary">{open ? "▾" : "▸"}</span>
                    {sample.name}
                  </span>
                  <span className="text-xs text-secondary">
                    {sample.measurements.length}
                  </span>
                </button>

                {open && (
                  <div className="ml-3 border-l border-edge pl-2">
                    {groups.map((group) => (
                      <div key={group.technique} className="mb-1 mt-1">
                        <div className="flex items-center gap-1.5 px-2 py-1">
                          <TechniqueChip technique={group.technique} />
                          <span className="text-xs text-secondary">
                            {techniqueLabel(group.technique)} ({group.measurements.length})
                          </span>
                        </div>
                        <ul>
                          {group.measurements.map((m) => (
                            <li key={m.id}>
                              <button
                                type="button"
                                onClick={() => setSelected(m)}
                                className={`w-full truncate rounded px-2 py-1 text-left text-sm ${
                                  selected?.id === m.id
                                    ? "bg-accent text-white"
                                    : "text-primary hover:bg-muted"
                                }`}
                                title={fileLabel(m)}
                              >
                                {fileLabel(m)}
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* Right: detail */}
        <main className="min-w-0 flex-1 overflow-y-auto">
          <DetailPane measurement={selected} />
        </main>
      </div>
    </div>
  );
}
