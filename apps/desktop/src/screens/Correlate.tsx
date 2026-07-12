/** Correlate — cross-technique correlation + publication figure export (Stage 6).
 *
 * Surfaces the sample × property feature table (with provenance and
 * data-quality flags), an interactive Pearson heatmap, the strongest
 * property pairs, and one-click journal-styled figure export. Click a
 * heatmap cell or a ranked pair to preview its scatter and download it.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  downloadFigure,
  figureUrl,
  getCorrelations,
  getFeatures,
  getReportStyles,
  type Correlations,
  type FeatureTable,
} from "../lib/api";
import { axisLabel } from "../lib/labels";
import { AnalysisLoader } from "../components/AnalysisLoader";

const FORMATS = ["svg", "pdf", "png"] as const;
type Fmt = (typeof FORMATS)[number];

/** The real steps the /features build runs through, shown while it computes. */
const FEATURE_STAGES = [
  "Reading measurements for every sample…",
  "Fitting XRD peaks — Scherrer crystallite sizes…",
  "Computing UV-DRS Tauc band gaps…",
  "Summarizing transport — peak zT and power factor…",
  "Running physics & data-quality checks…",
  "Assembling the cross-sample feature table…",
];

/** Diverging red/blue fill for a Pearson r in [-1, 1] (null = blank). */
function cellColor(r: number | null): string {
  if (r === null || Number.isNaN(r)) return "transparent";
  const a = Math.min(1, Math.abs(r));
  return r >= 0
    ? `rgba(200, 60, 60, ${a.toFixed(2)})`
    : `rgba(60, 110, 200, ${a.toFixed(2)})`;
}

const pretty = (p: string) => axisLabel(p);
const fmtR = (v: number | null) => (v === null || Number.isNaN(v) ? "" : v.toFixed(2));

export function Correlate({ onBack }: { onBack: () => void }) {
  const [table, setTable] = useState<FeatureTable | null>(null);
  const [corr, setCorr] = useState<Correlations | null>(null);
  const [styles, setStyles] = useState<string[]>([]);
  const [style, setStyle] = useState<string>("nature");
  const [fmt, setFmt] = useState<Fmt>("svg");
  const [reliableOnly, setReliableOnly] = useState(false);
  const [selected, setSelected] = useState<{ x: string; y: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    // /features is the slow one (runs the analysis layer, then cached).
    Promise.all([getFeatures(), getReportStyles()])
      .then(([t, s]) => {
        setTable(t);
        setStyles(s);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const loadCorr = useCallback(() => {
    getCorrelations(reliableOnly)
      .then(setCorr)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [reliableOnly]);

  useEffect(() => {
    if (table) loadCorr();
  }, [table, loadCorr]);

  const props = corr?.properties ?? [];

  const exportName = useMemo(() => {
    if (selected) return `${selected.x}_vs_${selected.y}.${fmt}`;
    return `correlation_heatmap.${fmt}`;
  }, [selected, fmt]);

  const doExport = () => {
    const opts = selected
      ? ({ kind: "scatter", style, fmt, x: selected.x, y: selected.y } as const)
      : ({ kind: "heatmap", style, fmt } as const);
    void downloadFigure(opts, exportName).catch((e: unknown) =>
      setError(e instanceof Error ? e.message : String(e)),
    );
  };

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
          Correlate — cross-technique
        </h1>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-5 px-8 py-6">
          {error && (
            <div className="rounded-md border border-severity-warning bg-[color-mix(in_srgb,var(--latos-severity-warning)_10%,transparent)] px-4 py-3 text-sm" data-selectable>
              {error}
            </div>
          )}

          {loading && (
            <AnalysisLoader title="Building the feature table" stages={FEATURE_STAGES} />
          )}

          {table && corr && (
            <>
              {/* Controls */}
              <section className="flex flex-wrap items-center gap-4 rounded-lg border border-edge bg-surface px-5 py-3 text-sm">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={reliableOnly}
                    onChange={(e) => setReliableOnly(e.target.checked)}
                  />
                  Trustworthy data only
                </label>
                <span className="text-secondary">Style</span>
                <select
                  value={style}
                  onChange={(e) => setStyle(e.target.value)}
                  className="rounded-md border border-edge bg-surface px-2 py-1"
                >
                  {styles.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
                <span className="text-secondary">Format</span>
                <select
                  value={fmt}
                  onChange={(e) => setFmt(e.target.value as Fmt)}
                  className="rounded-md border border-edge bg-surface px-2 py-1"
                >
                  {FORMATS.map((f) => (
                    <option key={f} value={f}>
                      {f.toUpperCase()}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={doExport}
                  className="ml-auto rounded-md bg-accent px-3 py-1.5 font-medium text-white"
                >
                  Export {selected ? "scatter" : "heatmap"} ↓
                </button>
              </section>

              {/* Heatmap */}
              <section className="overflow-x-auto rounded-lg border border-edge bg-surface p-4">
                <div className="mb-2 text-xs uppercase tracking-wide text-secondary">
                  Pearson r — click a cell to preview its scatter
                </div>
                <table className="border-separate" style={{ borderSpacing: 2 }}>
                  <thead>
                    <tr>
                      <th />
                      {props.map((p) => (
                        <th key={p} className="h-24 align-bottom">
                          <div
                            className="text-xs text-secondary"
                            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
                            data-selectable
                          >
                            {pretty(p)}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {props.map((rp, i) => (
                      <tr key={rp}>
                        <th className="whitespace-nowrap pr-2 text-right text-xs text-secondary" data-selectable>
                          {pretty(rp)}
                        </th>
                        {props.map((cp, j) => {
                          const r = corr.matrix[i][j];
                          const active = i !== j && r !== null;
                          const isSel =
                            selected &&
                            ((selected.x === cp && selected.y === rp) ||
                              (selected.x === rp && selected.y === cp));
                          return (
                            <td
                              key={cp}
                              onClick={() => active && setSelected({ x: cp, y: rp })}
                              title={active ? `${pretty(rp)} ↔ ${pretty(cp)}: r = ${fmtR(r)}` : ""}
                              className={`h-9 w-9 text-center text-[10px] ${active ? "cursor-pointer" : ""} ${isSel ? "ring-2 ring-accent" : ""}`}
                              style={{
                                backgroundColor: cellColor(r),
                                color: r !== null && Math.abs(r) > 0.6 ? "white" : "inherit",
                              }}
                            >
                              {fmtR(r)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              {/* Selected scatter preview */}
              {selected && (
                <section className="rounded-lg border border-edge bg-surface p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs uppercase tracking-wide text-secondary">
                      {pretty(selected.x)} vs {pretty(selected.y)}
                    </span>
                    <button
                      type="button"
                      onClick={() => setSelected(null)}
                      className="text-xs text-secondary hover:underline"
                    >
                      clear
                    </button>
                  </div>
                  <img
                    src={figureUrl({ kind: "scatter", style, fmt: "png", x: selected.x, y: selected.y })}
                    alt={`${selected.x} vs ${selected.y}`}
                    className="mx-auto max-h-80 bg-white"
                  />
                </section>
              )}

              {/* Ranked pairs */}
              <section className="rounded-lg border border-edge bg-surface p-4">
                <div className="mb-2 text-xs uppercase tracking-wide text-secondary">
                  Strongest relationships
                </div>
                <ul className="space-y-1 text-sm">
                  {corr.pairs.slice(0, 12).map((p) => (
                    <li key={`${p.property_a}:${p.property_b}`}>
                      <button
                        type="button"
                        onClick={() => setSelected({ x: p.property_a, y: p.property_b })}
                        className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-[color-mix(in_srgb,var(--latos-accent)_8%,transparent)]"
                      >
                        <span className="flex-1" data-selectable>
                          {pretty(p.property_a)} ↔ {pretty(p.property_b)}
                        </span>
                        <span className={`font-mono ${Math.abs(p.pearson) > 0.7 ? "text-primary" : "text-secondary"}`}>
                          r={p.pearson.toFixed(2)}
                        </span>
                        <span className="font-mono text-secondary">ρ={p.spearman.toFixed(2)}</span>
                        <span className="text-xs text-secondary">n={p.n}</span>
                      </button>
                    </li>
                  ))}
                  {corr.pairs.length === 0 && (
                    <li className="text-secondary">No correlations (need ≥3 shared samples per pair).</li>
                  )}
                </ul>
              </section>

              {/* Feature table */}
              <section className="overflow-x-auto rounded-lg border border-edge bg-surface">
                <div className="px-4 py-2 text-xs uppercase tracking-wide text-secondary">
                  Feature table — ⚠ marks values flagged by a physics / data-quality check
                </div>
                <table className="w-full text-sm" data-selectable>
                  <thead className="text-xs uppercase tracking-wide text-secondary">
                    <tr className="border-t border-edge">
                      <th className="px-3 py-2 text-left">Sample</th>
                      {table.properties.map((p) => (
                        <th key={p} className="px-3 py-2 text-right">
                          {pretty(p)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.rows.map((row) => (
                      <tr key={row.sample_id} className="border-t border-edge">
                        <td className="px-3 py-2 font-medium">{row.sample_name}</td>
                        {table.properties.map((p) => {
                          const c = row.features[p];
                          return (
                            <td key={p} className="px-3 py-2 text-right" title={c ? c.source : ""}>
                              {c ? (
                                <span className={c.reliable ? "" : "text-severity-warning"}>
                                  {c.value.toPrecision(3)}
                                  {c.reliable ? "" : " ⚠"}
                                </span>
                              ) : (
                                <span className="text-secondary">—</span>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
