/** OptimizeSurfaceChart — the two-variable closed-loop figure, drawn as SVG.
 *
 * The one-variable figure is a curve with a band around it. That does not
 * generalise: over two axes the model's answer is a surface, and the thing a
 * researcher needs to see is *where on the plane* the interesting region is —
 * which is a map, not a line.
 *
 * The map shows one of three quantities at a time, because they answer three
 * different questions and overlaying them would answer none of them clearly:
 *
 *   predicted   what the model thinks the property is worth everywhere
 *   uncertainty how sure it is — the unexplored regions light up
 *   opportunity expected improvement, which is what actually chose the ★
 *
 * Observed samples sit on top as labelled points, so the map is always read
 * against the measurements it was fitted to rather than on its own.
 */

import type { OptimizeNdResult } from "../lib/api";
import { axisLabel } from "../lib/labels";

export type SurfaceMode = "mean" | "sd" | "ei";

const W = 760;
const H = 480;
const ML = 62;
const PLOT_R = 640; // right edge of the map; the colour bar lives beyond it
const TOP = 18;
const BOT = 396;
const BAR_L = 664;
const BAR_W = 16;

const GRID = "#9aa0a6";
const POINT = "#D2542C";
const STAR = "#FFFFFF";

/** Viridis, sampled at six stops and interpolated between them.
 *
 * Perceptually uniform and readable in greyscale or with colour-vision
 * deficiency — which matters because this map may end up in a thesis figure,
 * where a rainbow ramp would invent structure that is not in the data. */
const RAMP = [
  [68, 1, 84],
  [65, 68, 135],
  [42, 120, 142],
  [34, 168, 132],
  [122, 209, 81],
  [253, 231, 37],
] as const;

function colour(t: number): string {
  const c = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(c), RAMP.length - 2);
  const f = c - i;
  const [r0, g0, b0] = RAMP[i];
  const [r1, g1, b1] = RAMP[i + 1];
  const mix = (a: number, b: number) => Math.round(a + (b - a) * f);
  return `rgb(${mix(r0, r1)},${mix(g0, g1)},${mix(b0, b1)})`;
}

function scale(v: number, d0: number, d1: number, r0: number, r1: number): number {
  if (d1 === d0) return (r0 + r1) / 2;
  return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

const fmt = (v: number): string => {
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(1);
  return a >= 100 ? v.toFixed(0) : v.toFixed(a >= 1 ? 2 : 3);
};

const MODE_LABEL: Record<SurfaceMode, string> = {
  mean: "predicted value",
  sd: "model uncertainty (higher = less explored)",
  ei: "expected improvement (what chose the ★)",
};

export function OptimizeSurfaceChart({
  result,
  mode = "mean",
}: {
  result: OptimizeNdResult;
  mode?: SurfaceMode;
}) {
  const s = result.surface;
  if (!s) {
    return (
      <div className="px-4 py-8 text-center text-sm text-secondary">
        A map needs exactly two axes. This run used {result.input_variables.length}.
      </div>
    );
  }

  const grid = mode === "mean" ? s.mean : mode === "sd" ? s.sd : s.ei;
  const flat = grid.flat();
  const vLo = Math.min(...flat);
  const vHi = Math.max(...flat);

  const nx = s.axis_x.length;
  const ny = s.axis_y.length;
  const xLo = s.axis_x[0];
  const xHi = s.axis_x[nx - 1];
  const yLo = s.axis_y[0];
  const yHi = s.axis_y[ny - 1];

  const X = (v: number) => scale(v, xLo, xHi, ML, PLOT_R);
  const Y = (v: number) => scale(v, yLo, yHi, BOT, TOP);

  // Cells are centred on their lattice point, so the map covers the whole box
  // rather than stopping half a cell short of each edge. The +0.6 overlap hides
  // the hairline seams the renderer leaves between abutting rects.
  const cw = (PLOT_R - ML) / (nx - 1);
  const ch = (BOT - TOP) / (ny - 1);

  const rec = result.recommendation.x;
  const best = result.best_x;
  const ticks = 5;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img">
      {/* the map */}
      <g shapeRendering="crispEdges">
        {grid.map((row, j) =>
          row.map((v, i) => (
            <rect
              key={`${i}-${j}`}
              x={X(s.axis_x[i]) - cw / 2}
              y={Y(s.axis_y[j]) - ch / 2}
              width={cw + 0.6}
              height={ch + 0.6}
              fill={colour(vHi === vLo ? 0.5 : (v - vLo) / (vHi - vLo))}
            />
          )),
        )}
      </g>
      <rect
        x={ML}
        y={TOP}
        width={PLOT_R - ML}
        height={BOT - TOP}
        fill="none"
        stroke={GRID}
        strokeOpacity={0.4}
      />

      {/* observed samples, drawn over the map they were fitted to */}
      {result.points.map((p) => (
        <g key={p.sample_id}>
          <circle
            cx={X(p.x[0])}
            cy={Y(p.x[1])}
            r={4.5}
            fill={POINT}
            stroke="#ffffff"
            strokeWidth={1.4}
          />
          <text
            x={X(p.x[0])}
            y={Y(p.x[1]) - 9}
            textAnchor="middle"
            fontSize="9.5"
            fill="#ffffff"
            stroke="#00000088"
            strokeWidth={2.2}
            paintOrder="stroke"
          >
            {p.sample_name}
          </text>
        </g>
      ))}

      {/* best measured point: a ring, so it reads as "already made" */}
      <circle
        cx={X(best[0])}
        cy={Y(best[1])}
        r={9}
        fill="none"
        stroke="#ffffff"
        strokeWidth={1.6}
        strokeDasharray="3 2"
      />

      {/* the recommendation */}
      <line
        x1={X(rec[0])}
        y1={TOP}
        x2={X(rec[0])}
        y2={BOT}
        stroke={STAR}
        strokeWidth={1}
        strokeOpacity={0.55}
        strokeDasharray="4 3"
      />
      <line
        x1={ML}
        y1={Y(rec[1])}
        x2={PLOT_R}
        y2={Y(rec[1])}
        stroke={STAR}
        strokeWidth={1}
        strokeOpacity={0.55}
        strokeDasharray="4 3"
      />
      <text
        x={X(rec[0])}
        y={Y(rec[1]) + 8}
        textAnchor="middle"
        fontSize="24"
        fill={STAR}
        stroke="#00000099"
        strokeWidth={2}
        paintOrder="stroke"
      >
        ★
      </text>

      {/* axes */}
      {Array.from({ length: ticks + 1 }, (_, i) => xLo + (i / ticks) * (xHi - xLo)).map((v) => (
        <text key={`x${v}`} x={X(v)} y={BOT + 17} textAnchor="middle" fontSize="11" fill={GRID}>
          {fmt(v)}
        </text>
      ))}
      {Array.from({ length: ticks + 1 }, (_, i) => yLo + (i / ticks) * (yHi - yLo)).map((v) => (
        <text key={`y${v}`} x={ML - 8} y={Y(v) + 4} textAnchor="end" fontSize="11" fill={GRID}>
          {fmt(v)}
        </text>
      ))}
      <text x={(ML + PLOT_R) / 2} y={BOT + 36} textAnchor="middle" fontSize="12" fill={GRID}>
        {axisLabel(s.axis_names[0])}
      </text>
      <text
        x={14}
        y={(TOP + BOT) / 2}
        textAnchor="middle"
        fontSize="12"
        fill={GRID}
        transform={`rotate(-90 14 ${(TOP + BOT) / 2})`}
      >
        {axisLabel(s.axis_names[1])}
      </text>

      {/* colour bar */}
      <defs>
        <linearGradient id="latos-surface-ramp" x1="0" y1="1" x2="0" y2="0">
          {RAMP.map((_c, i) => (
            <stop
              key={i}
              offset={`${(i / (RAMP.length - 1)) * 100}%`}
              stopColor={colour(i / (RAMP.length - 1))}
            />
          ))}
        </linearGradient>
      </defs>
      <rect
        x={BAR_L}
        y={TOP}
        width={BAR_W}
        height={BOT - TOP}
        fill="url(#latos-surface-ramp)"
        stroke={GRID}
        strokeOpacity={0.4}
      />
      <text x={BAR_L + BAR_W + 5} y={TOP + 10} fontSize="10" fill={GRID}>
        {fmt(vHi)}
      </text>
      <text x={BAR_L + BAR_W + 5} y={BOT} fontSize="10" fill={GRID}>
        {fmt(vLo)}
      </text>

      {/* what the colours mean, and what the marks mean */}
      <text x={ML} y={BOT + 56} fontSize="10.5" fill={GRID}>
        colour = {MODE_LABEL[mode]}
        {mode === "mean" ? ` · ${axisLabel(result.target_property)}` : ""}
      </text>
      <text x={ML} y={BOT + 71} fontSize="10.5" fill={GRID}>
        ● measured sample · ◌ best so far · ★ recommended next experiment
      </text>
    </svg>
  );
}
