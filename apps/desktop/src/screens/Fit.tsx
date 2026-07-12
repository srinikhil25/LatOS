/** Fit — the "replace Origin" peak-fitting editor (Stage 4E).
 *
 * Pick a spectrum, auto-detect (or type) peaks, choose a line shape and
 * background, and fit. The result overlays the data with the fitted
 * envelope, its baseline, and a residual strip, plus a parameter table
 * with ±1σ uncertainties and a paste-ready Markdown report.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  detectPeaks,
  getMeasurementArrays,
  getSamples,
  runFit,
  type BackgroundKind,
  type FitResult,
  type PeakShape,
  type SampleSummary,
} from "../lib/api";
import { axisLabel } from "../lib/labels";
import { AnalysisLoader } from "../components/AnalysisLoader";

const SHAPES: { value: PeakShape; label: string }[] = [
  { value: "pseudo_voigt", label: "Pseudo-Voigt" },
  { value: "gaussian", label: "Gaussian" },
  { value: "lorentzian", label: "Lorentzian" },
  { value: "voigt", label: "Voigt" },
  { value: "doniach", label: "Doniach–Šunjić" },
  { value: "skewed_voigt", label: "Skewed Voigt" },
];

const BACKGROUNDS: { value: BackgroundKind; label: string }[] = [
  { value: "linear", label: "Linear" },
  { value: "shirley", label: "Shirley (XPS)" },
  { value: "als", label: "ALS (Raman)" },
  { value: "polynomial", label: "Polynomial" },
  { value: "constant", label: "Constant" },
  { value: "none", label: "None" },
];

// Techniques that produce a peak-fittable spectrum (excludes images like
// TEM/SEM and scalar/curve techniques like Hall and thermoelectric).
const FITTABLE = new Set(["xrd", "xps", "raman", "uv_drs", "eds"]);

/** The real steps runFit runs through, shown while a fit is in flight. */
const FIT_STAGES = [
  "Sorting the spectrum & seeding parameters…",
  "Building the composite peak + background model…",
  "Least-squares refinement (lmfit)…",
  "Computing ±1σ uncertainties and residuals…",
];

// Preferred x / y column names, most specific first.
const X_HINTS = ["two_theta", "binding_energy", "raman_shift", "wavelength_nm", "energy_ev"];
const Y_HINTS = ["intensity", "cps", "counts", "reflectance_pct", "absorbance"];

interface MeasOption {
  id: string;
  label: string;
}

function pickDefault(names: string[], hints: string[], fallbackIdx: number): string {
  for (const h of hints) {
    const hit = names.find((n) => n.toLowerCase().includes(h));
    if (hit) return hit;
  }
  return names[fallbackIdx] ?? names[0] ?? "";
}

/** Clean (x, y) numeric pairs, dropping any null/non-finite sample. */
function cleanXY(
  arrays: Record<string, (number | null)[]>,
  xName: string,
  yName: string,
): { x: number[]; y: number[] } {
  const xs = arrays[xName] ?? [];
  const ys = arrays[yName] ?? [];
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
    const xi = xs[i];
    const yi = ys[i];
    if (xi != null && yi != null && Number.isFinite(xi) && Number.isFinite(yi)) {
      x.push(xi);
      y.push(yi);
    }
  }
  return { x, y };
}

const fmt = (v: number | null): string =>
  v == null ? "—" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(3);

export function Fit({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [measId, setMeasId] = useState<string>("");
  const [names, setNames] = useState<string[]>([]);
  const [arrays, setArrays] = useState<Record<string, (number | null)[]>>({});
  const [xName, setXName] = useState<string>("");
  const [yName, setYName] = useState<string>("");
  const [peaks, setPeaks] = useState<number[]>([]);
  const [shape, setShape] = useState<PeakShape>("pseudo_voigt");
  const [background, setBackground] = useState<BackgroundKind>("linear");
  const [result, setResult] = useState<FitResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "detecting" | "fitting">("idle");
  const busy = phase !== "idle";

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const options = useMemo<MeasOption[]>(() => {
    const out: MeasOption[] = [];
    for (const s of samples) {
      for (const m of s.measurements) {
        if (!FITTABLE.has(m.technique)) continue;
        out.push({ id: m.id, label: `${s.name} · ${m.technique}${m.filename ? ` · ${m.filename}` : ""}` });
      }
    }
    return out;
  }, [samples]);

  const chooseMeasurement = useCallback((id: string) => {
    setMeasId(id);
    setResult(null);
    setPeaks([]);
    setError(null);
    if (!id) return;
    getMeasurementArrays(id)
      .then((a) => {
        const num = a.names.filter((n) => (a.arrays[n] ?? []).some((v) => v != null));
        setNames(num);
        setArrays(a.arrays);
        const x = pickDefault(num, X_HINTS, 0);
        const y = pickDefault(
          num.filter((n) => n !== x),
          Y_HINTS,
          0,
        );
        setXName(x);
        setYName(y);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const xy = useMemo(() => cleanXY(arrays, xName, yName), [arrays, xName, yName]);

  const autoDetect = useCallback(() => {
    if (xy.x.length < 5) return;
    setPhase("detecting");
    detectPeaks(xy.x, xy.y)
      .then((r) => setPeaks(r.centers.slice().sort((a, b) => a - b)))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setPhase("idle"));
  }, [xy]);

  const doFit = useCallback(() => {
    if (xy.x.length < 5 || peaks.length === 0) return;
    setPhase("fitting");
    setError(null);
    setResult(null);
    runFit({ x: xy.x, y: xy.y, peak_shape: shape, peaks, background: { kind: background } })
      .then(setResult)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setPhase("idle"));
  }, [xy, peaks, shape, background]);

  const removePeak = (i: number) => setPeaks((p) => p.filter((_, j) => j !== i));
  const addPeak = (v: string) => {
    const n = Number.parseFloat(v);
    if (Number.isFinite(n)) setPeaks((p) => [...p, n].sort((a, b) => a - b));
  };
  const copyReport = () => {
    if (result) void navigator.clipboard?.writeText(result.markdown);
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
          Fit — peak fitting
        </h1>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-5 px-8 py-6">
          {error && (
            <div className="rounded-md border border-severity-warning bg-[color-mix(in_srgb,var(--latos-severity-warning)_10%,transparent)] px-4 py-3 text-sm" data-selectable>
              {error}
            </div>
          )}

          {/* Controls */}
          <section className="space-y-4 rounded-lg border border-edge bg-surface px-5 py-4">
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wide text-secondary">Spectrum</span>
              <select
                value={measId}
                onChange={(e) => chooseMeasurement(e.target.value)}
                className="w-full rounded-md border border-edge bg-transparent px-3 py-2 text-sm"
              >
                <option value="">Select a measurement…</option>
                {options.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>

            {names.length > 0 && (
              <div className="grid grid-cols-2 gap-4">
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wide text-secondary">X axis</span>
                  <select
                    value={xName}
                    onChange={(e) => setXName(e.target.value)}
                    className="w-full rounded-md border border-edge bg-transparent px-3 py-2 text-sm"
                  >
                    {names.map((n) => (
                      <option key={n} value={n}>
                        {axisLabel(n)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wide text-secondary">Y axis</span>
                  <select
                    value={yName}
                    onChange={(e) => setYName(e.target.value)}
                    className="w-full rounded-md border border-edge bg-transparent px-3 py-2 text-sm"
                  >
                    {names.map((n) => (
                      <option key={n} value={n}>
                        {axisLabel(n)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}

            {names.length > 0 && (
              <div className="grid grid-cols-2 gap-4">
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wide text-secondary">Peak shape</span>
                  <select
                    value={shape}
                    onChange={(e) => setShape(e.target.value as PeakShape)}
                    className="w-full rounded-md border border-edge bg-transparent px-3 py-2 text-sm"
                  >
                    {SHAPES.map((s) => (
                      <option key={s.value} value={s.value}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wide text-secondary">Background</span>
                  <select
                    value={background}
                    onChange={(e) => setBackground(e.target.value as BackgroundKind)}
                    className="w-full rounded-md border border-edge bg-transparent px-3 py-2 text-sm"
                  >
                    {BACKGROUNDS.map((b) => (
                      <option key={b.value} value={b.value}>
                        {b.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}

            {names.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase tracking-wide text-secondary">
                    Peaks ({peaks.length})
                  </span>
                  <button
                    type="button"
                    onClick={autoDetect}
                    disabled={busy || xy.x.length < 5}
                    className="rounded-md border border-edge px-2 py-1 text-xs hover:bg-[color-mix(in_srgb,var(--latos-accent)_8%,transparent)] disabled:opacity-40"
                  >
                    Auto-detect
                  </button>
                  <input
                    type="number"
                    placeholder="add center…"
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        addPeak((e.target as HTMLInputElement).value);
                        (e.target as HTMLInputElement).value = "";
                      }
                    }}
                    className="w-28 rounded-md border border-edge bg-transparent px-2 py-1 text-xs"
                  />
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {peaks.map((p, i) => (
                    <button
                      key={`${p}-${i}`}
                      type="button"
                      onClick={() => removePeak(i)}
                      title="Remove"
                      className="rounded-full border border-edge px-2 py-0.5 text-xs text-secondary hover:border-severity-warning hover:text-severity-warning"
                    >
                      {fmt(p)} ✕
                    </button>
                  ))}
                  {peaks.length === 0 && (
                    <span className="text-xs text-secondary">None yet — auto-detect or add a center.</span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={doFit}
                  disabled={busy || peaks.length === 0}
                  className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
                >
                  {phase === "fitting" ? "Fitting…" : "Fit"}
                </button>
              </div>
            )}
          </section>

          {phase === "fitting" && (
            <AnalysisLoader title="Running the fit" stages={FIT_STAGES} />
          )}

          {/* Result */}
          {result && (
            <section className="space-y-4">
              <SpectrumPlot
                x={xy.x}
                y={xy.y}
                fit={result.best_fit}
                baseline={result.baseline}
                peaks={result.components.map((c) => c.center)}
                xLabel={axisLabel(xName)}
              />

              <div className="flex flex-wrap gap-4 rounded-lg border border-edge bg-surface px-5 py-3 text-sm" data-selectable>
                <span>R² = <strong>{result.r_squared.toFixed(4)}</strong></span>
                <span>χ² = {fmt(result.chi_square)}</span>
                <span>reduced χ² = {fmt(result.reduced_chi_square)}</span>
                <span className={result.success ? "text-[color:var(--latos-tech-eds)]" : "text-severity-warning"}>
                  {result.success ? "converged" : "did not converge"}
                </span>
              </div>

              <div className="overflow-x-auto rounded-lg border border-edge bg-surface">
                <table className="w-full text-sm" data-selectable>
                  <thead className="text-xs uppercase tracking-wide text-secondary">
                    <tr className="border-b border-edge">
                      <th className="px-4 py-2 text-left">Peak</th>
                      <th className="px-4 py-2 text-right">Center</th>
                      <th className="px-4 py-2 text-right">Area</th>
                      <th className="px-4 py-2 text-right">FWHM</th>
                      <th className="px-4 py-2 text-right">Height</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.components.map((c, i) => (
                      <tr key={c.center} className="border-b border-edge last:border-0">
                        <td className="px-4 py-2">{i + 1}</td>
                        <td className="px-4 py-2 text-right">{fmt(c.center)}</td>
                        <td className="px-4 py-2 text-right">{fmt(c.amplitude)}</td>
                        <td className="px-4 py-2 text-right">{fmt(c.fwhm)}</td>
                        <td className="px-4 py-2 text-right">{fmt(c.height)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="rounded-lg border border-edge bg-surface px-5 py-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wide text-secondary">Report (Markdown)</span>
                  <button
                    type="button"
                    onClick={copyReport}
                    className="rounded-md border border-edge px-2 py-1 text-xs hover:bg-[color-mix(in_srgb,var(--latos-accent)_8%,transparent)]"
                  >
                    Copy
                  </button>
                </div>
                <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-secondary" data-selectable>
                  {result.markdown}
                </pre>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

/** Data + fitted envelope + baseline overlay, with a residual strip. */
function SpectrumPlot({
  x,
  y,
  fit,
  baseline,
  peaks,
  xLabel,
}: {
  x: number[];
  y: number[];
  fit: number[];
  baseline: number[];
  peaks: number[];
  xLabel: string;
}) {
  const W = 720;
  const H = 300;
  const RH = 70;
  const pad = 34;
  const xmin = Math.min(...x);
  const xmax = Math.max(...x);
  const ymin = Math.min(...y, ...baseline);
  const ymax = Math.max(...y, ...fit);
  const sx = (v: number) => pad + ((v - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const syTop = (v: number) => pad + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
  const resid = y.map((yi, i) => yi - (fit[i] ?? yi));
  const rmax = Math.max(1e-9, ...resid.map(Math.abs));
  const syRes = (v: number) => H + RH / 2 - (v / rmax) * (RH / 2 - 6);

  const path = (arr: number[], sy: (v: number) => number) =>
    arr.map((v, i) => `${i === 0 ? "M" : "L"}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");

  return (
    <div className="overflow-x-auto rounded-lg border border-edge bg-surface p-3">
      <svg viewBox={`0 0 ${W} ${H + RH + 24}`} className="w-full" role="img" aria-label="fit overlay">
        {/* frames */}
        <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="none" stroke="var(--latos-edge)" />
        {/* data */}
        <path d={path(y, syTop)} fill="none" stroke="var(--latos-text-secondary)" strokeWidth={1} opacity={0.7} />
        {/* baseline */}
        <path d={path(baseline, syTop)} fill="none" stroke="var(--latos-edge)" strokeWidth={1} strokeDasharray="4 3" />
        {/* fit */}
        <path d={path(fit, syTop)} fill="none" stroke="var(--latos-accent)" strokeWidth={1.6} />
        {/* peak markers */}
        {peaks.map((p) => (
          <line
            key={p}
            x1={sx(p)}
            x2={sx(p)}
            y1={pad}
            y2={H - pad}
            stroke="var(--latos-accent)"
            strokeWidth={0.75}
            strokeDasharray="2 4"
            opacity={0.6}
          />
        ))}
        {/* residual strip */}
        <line x1={pad} x2={W - pad} y1={H + RH / 2} y2={H + RH / 2} stroke="var(--latos-edge)" strokeWidth={0.5} />
        <path d={path(resid, syRes)} fill="none" stroke="var(--latos-text-secondary)" strokeWidth={0.75} opacity={0.7} />
        <text x={pad} y={H + RH + 18} fontSize="11" fill="var(--latos-text-secondary)">
          residual
        </text>
        <text x={W - pad} y={H + RH + 18} fontSize="11" textAnchor="end" fill="var(--latos-text-secondary)">
          {xLabel}
        </text>
      </svg>
    </div>
  );
}
