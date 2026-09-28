/** AnalysisOverlayChart — draws the curves an analyzer fitted.
 *
 * The analysis panel lists an analyzer's scalars: peak centres, an R², a
 * reduced χ². Those numbers are not self-explaining. An R² of 0.74 with a
 * whole-scan R² of −0.50 says nothing about *which* part of the scan the
 * fit describes, and a researcher cannot accept or reject a fit they
 * cannot see. The analyzer already computes the overlay — observed curve,
 * baseline, fitted line, residual — so this component draws it.
 *
 * Deliberately convention-driven rather than XRD-specific. The analyzer
 * contract guarantees `derived_arrays` are equal-length 1-D columns, so
 * one column is the abscissa and the rest are traces over it. Today only
 * the XRD peak fit derives any; when another analyzer does, it plots here
 * without a change, provided it names its abscissa recognisably.
 *
 * Carries its own ChartFrame, so the figure gets the same save-as-PNG
 * action as every other chart in the app — and so that frame disappears
 * with the chart. Most analyzers derive no arrays, and an empty bordered
 * box with a save button under every one of them is worse than nothing.
 */

import { useEffect, useState } from "react";
import { getAnalysisArrays, type AnalyzerArrays } from "../lib/api";
import { axisLabel } from "../lib/labels";
import { ChartFrame } from "./ChartFrame";

/** Column names that mean "this is the x axis", cheapest match first. */
const ABSCISSA_NAMES = [
  "two_theta",
  "binding_energy_ev",
  "energy_ev",
  "raman_shift_cm1",
  "wavelength_nm",
  "temperature_k",
];

/** Drawn into the residual strip rather than the main panel. */
const RESIDUAL_NAME = "residual";

/** How each known trace is drawn. Anything unlisted falls back to `other`.
 *
 * Data is grey, the model is the accent: the baseline and the fitted line are
 * both things the analyzer produced, so they share a colour and differ by
 * weight. Nothing here uses the border token — a trace drawn in the colour of
 * chrome vanishes against a white background when the figure is exported.
 */
interface TraceStyle {
  stroke: string;
  width: number;
  dash?: string;
  /** Legend text. Not `axisLabel`: "Intensity (a.u.)" names an axis, and a
   *  key that repeats the axis label tells the reader nothing about which
   *  line is which. */
  label: string;
}

const TRACE_STYLE: Record<string, TraceStyle> = {
  intensity_observed: { stroke: "var(--latos-text-secondary)", width: 1, label: "observed" },
  baseline: { stroke: "var(--latos-accent)", width: 1, dash: "4 3", label: "baseline" },
  fit_line: { stroke: "var(--latos-accent)", width: 1.6, label: "fitted" },
};

/** An unlisted trace still gets drawn, named from its own column. */
function otherStyle(name: string): TraceStyle {
  return { stroke: "var(--latos-text-secondary)", width: 1, dash: "2 3", label: name.replace(/_/g, " ") };
}

/** Order traces so the fitted line paints last and stays legible. */
const PAINT_ORDER = ["baseline", "intensity_observed", "fit_line"];

const W = 720;
const H = 300;
const RH = 56;
const PAD = 40;
/** Top of the residual band. Sits just under the x tick labels: a difference
 *  curve is read against the panel above it, and a wide gap breaks that. */
const RES_TOP = H - PAD + 30;

/** Finite values only; `null` marks a gap the server could not encode. */
function finite(values: (number | null)[]): number[] {
  return values.filter((v): v is number => v !== null && Number.isFinite(v));
}

/** Round tick values at a 1/2/5 x 10^n interval, as a plotting library would.
 *
 * Ticks at the data's own min, midpoint and max look computed rather than
 * chosen: an axis reading 273 / 747 / 1221 tells the reader where the array
 * happened to start and stop, not where the round numbers are. Snapping to a
 * decimal interval also makes every label on an axis share a decimal count,
 * which taking the raw extremes does not (3.00 next to 15.0).
 */
function niceTicks(min: number, max: number, target = 5): { values: number[]; step: number } {
  if (!(max > min)) return { values: [min], step: 1 };
  const rough = (max - min) / target;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const step = (normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10) * magnitude;
  const values: number[] = [];
  // The 1e-9 slack keeps a tick that lands exactly on `max` from being lost
  // to floating-point drift after repeated addition.
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    values.push(Number(v.toFixed(10)));
  }
  return { values, step };
}

/** One formatter per axis, so every label on it carries the same decimals. */
function tickFormatter(step: number): (v: number) => string {
  const abs = Math.abs(step);
  if (abs >= 10000 || (abs > 0 && abs < 0.001)) return (v) => v.toExponential(1);
  const decimals = Math.max(0, -Math.floor(Math.log10(abs)));
  return (v) => v.toFixed(decimals);
}

export function AnalysisOverlayChart({
  measurementId,
  analyzer,
}: {
  measurementId: string;
  analyzer: string;
}) {
  const [data, setData] = useState<AnalyzerArrays | null>(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    void getAnalysisArrays(measurementId, analyzer).then((d) => {
      if (alive) setData(d);
    });
    return () => {
      alive = false;
    };
  }, [measurementId, analyzer]);

  if (data === null) return null;

  const xName =
    ABSCISSA_NAMES.find((n) => data.names.includes(n)) ??
    data.names.find((n) => n !== RESIDUAL_NAME);
  if (xName === undefined) return null;

  const x = finite(data.arrays[xName] ?? []);
  if (x.length < 2) return null;

  const traceNames = data.names.filter((n) => n !== xName && n !== RESIDUAL_NAME);
  if (traceNames.length === 0) return null;

  const traces = traceNames
    .map((name) => ({ name, values: finite(data.arrays[name] ?? []) }))
    .filter((t) => t.values.length === x.length);
  if (traces.length === 0) return null;

  const yName = traces.some((t) => t.name === "intensity_observed")
    ? "intensity_observed"
    : traces[0].name;
  const residual = finite(data.arrays[RESIDUAL_NAME] ?? []);
  const hasResidual = residual.length === x.length;
  const totalH = hasResidual ? RES_TOP + RH + 24 : H - PAD + 40;

  const xmin = Math.min(...x);
  const xmax = Math.max(...x);
  const ys = traces.flatMap((t) => t.values);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);

  const sx = (v: number) => PAD + ((v - xmin) / (xmax - xmin || 1)) * (W - 2 * PAD);
  const sy = (v: number) => PAD + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - 2 * PAD);
  const xTicks = niceTicks(xmin, xmax);
  const yTicks = niceTicks(ymin, ymax);
  const formatX = tickFormatter(xTicks.step);
  const formatY = tickFormatter(yTicks.step);
  const rmax = Math.max(1e-9, ...residual.map(Math.abs));
  const syRes = (v: number) => RES_TOP + RH / 2 - (v / rmax) * (RH / 2 - 6);

  const path = (arr: number[], scale: (v: number) => number) =>
    arr
      .map((v, i) => `${i === 0 ? "M" : "L"}${sx(x[i]).toFixed(1)},${scale(v).toFixed(1)}`)
      .join(" ");

  const ordered = [...traces].sort(
    (a, b) =>
      (PAINT_ORDER.indexOf(a.name) + 1 || PAINT_ORDER.length + 1) -
      (PAINT_ORDER.indexOf(b.name) + 1 || PAINT_ORDER.length + 1),
  );

  return (
    <div className="mt-3">
      <ChartFrame basename={`latos-${analyzer}`} label={`${analyzer} figure`} scale={4}>
        <svg
          viewBox={`0 0 ${W} ${totalH}`}
          className="w-full"
          role="img"
          aria-label={`${analyzer} fit overlay`}
        >
      {/* Left and bottom spines only. A full box in the border colour is
          near-invisible once the figure is exported onto white, and reads
          as a rendering fault rather than a frame. */}
      <line
        x1={PAD}
        x2={PAD}
        y1={PAD}
        y2={H - PAD}
        stroke="var(--latos-text-secondary)"
        strokeWidth={1}
      />
      <line
        x1={PAD}
        x2={W - PAD}
        y1={H - PAD}
        y2={H - PAD}
        stroke="var(--latos-text-secondary)"
        strokeWidth={1}
      />

      {yTicks.values.map((t) => (
        <g key={`y-${t}`}>
          <line
            x1={PAD - 4}
            x2={PAD}
            y1={sy(t)}
            y2={sy(t)}
            stroke="var(--latos-text-secondary)"
            strokeWidth={1}
          />
          <text
            x={PAD - 8}
            y={sy(t) + 3}
            fontSize="10"
            textAnchor="end"
            fill="var(--latos-text-secondary)"
          >
            {formatY(t)}
          </text>
        </g>
      ))}
      {xTicks.values.map((t) => (
        <g key={`x-${t}`}>
          <line
            x1={sx(t)}
            x2={sx(t)}
            y1={H - PAD}
            y2={H - PAD + 4}
            stroke="var(--latos-text-secondary)"
            strokeWidth={1}
          />
          <text
            x={sx(t)}
            y={H - PAD + 17}
            fontSize="10"
            textAnchor="middle"
            fill="var(--latos-text-secondary)"
          >
            {formatX(t)}
          </text>
        </g>
      ))}

      {ordered.map((t) => {
        const style = TRACE_STYLE[t.name] ?? otherStyle(t.name);
        return (
          <path
            key={t.name}
            d={path(t.values, sy)}
            fill="none"
            stroke={style.stroke}
            strokeWidth={style.width}
            strokeDasharray={style.dash}
          />
        );
      })}

      {ordered.map((t, i) => {
        const style = TRACE_STYLE[t.name] ?? otherStyle(t.name);
        return (
          <g key={`key-${t.name}`} transform={`translate(${PAD + 10}, ${PAD + 14 + i * 15})`}>
            <line
              x1={0}
              x2={18}
              y1={0}
              y2={0}
              stroke={style.stroke}
              strokeWidth={style.width}
              strokeDasharray={style.dash}
            />
            <text x={24} y={3} fontSize="11" fill="var(--latos-text-secondary)">
              {style.label}
            </text>
          </g>
        );
      })}

      {hasResidual && (
        <>
          <line
            x1={PAD}
            x2={W - PAD}
            y1={RES_TOP + RH / 2}
            y2={RES_TOP + RH / 2}
            stroke="var(--latos-text-secondary)"
            strokeWidth={0.5}
          />
          <path
            d={path(residual, syRes)}
            fill="none"
            stroke="var(--latos-text-secondary)"
            strokeWidth={0.75}
            opacity={0.7}
          />
          <text x={PAD} y={RES_TOP + RH + 16} fontSize="11" fill="var(--latos-text-secondary)">
            residual
          </text>
        </>
      )}

          <text
            x={W - PAD}
            y={totalH - 6}
            fontSize="11"
            textAnchor="end"
            fill="var(--latos-text-secondary)"
          >
            {axisLabel(xName)}
          </text>

          <text
            x={12}
            y={PAD + (H - 2 * PAD) / 2}
            fontSize="11"
            textAnchor="middle"
            fill="var(--latos-text-secondary)"
            transform={`rotate(-90, 12, ${PAD + (H - 2 * PAD) / 2})`}
          >
            {axisLabel(yName)}
          </text>
        </svg>
      </ChartFrame>
    </div>
  );
}
