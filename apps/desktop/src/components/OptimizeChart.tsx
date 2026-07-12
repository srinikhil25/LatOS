/** OptimizeChart — the closed-loop figure, drawn as SVG.
 *
 * Top panel: the GP model (mean curve + 95% band) over the synthesis
 * parameter, the researcher's measured points, and a ★ at the
 * recommended next experiment. Bottom strip: the "expected improvement"
 * — the mountain that shrinks as the search converges. Hand-drawn SVG
 * so every element is exactly where a materials scientist expects it,
 * with no chart-library jargon.
 */

import type { OptimizeResult } from "../lib/api";
import { axisLabel } from "../lib/labels";

const W = 720;
const H = 470;
const ML = 58;
const MR = 18;
const MAIN_TOP = 16;
const MAIN_BOT = 320;
const EI_TOP = 356;
const EI_BOT = 426;

const ACCENT = "#3B7DD8";
const GREEN = "#107C10";
const POINT = "#D2542C";
const GRID = "#9aa0a6";

function scale(v: number, d0: number, d1: number, r0: number, r1: number): number {
  if (d1 === d0) return (r0 + r1) / 2;
  return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

export function OptimizeChart({ result }: { result: OptimizeResult }) {
  const gx = result.grid_x;
  const xMin = gx[0];
  const xMax = gx[gx.length - 1];

  const upper = result.grid_mean.map((m, i) => m + result.grid_ci95[i]);
  const lower = result.grid_mean.map((m, i) => m - result.grid_ci95[i]);
  const yLo = Math.min(...lower, ...result.points.map((p) => p.y));
  const yHi = Math.max(...upper, ...result.points.map((p) => p.y));
  const yPad = 0.08 * (yHi - yLo || 1);

  const X = (v: number) => scale(v, xMin, xMax, ML, W - MR);
  const Y = (v: number) => scale(v, yLo - yPad, yHi + yPad, MAIN_BOT, MAIN_TOP);

  const eiMax = Math.max(...result.grid_ei, result.noise_threshold, 1e-9);
  const EY = (v: number) => scale(v, 0, eiMax, EI_BOT, EI_TOP);

  const meanPath = gx.map((x, i) => `${i ? "L" : "M"}${X(x)},${Y(result.grid_mean[i])}`).join("");
  const bandPath =
    gx.map((x, i) => `${i ? "L" : "M"}${X(x)},${Y(upper[i])}`).join("") +
    gx
      .map((_x, i) => `L${X(gx[gx.length - 1 - i])},${Y(lower[gx.length - 1 - i])}`)
      .join("") +
    "Z";
  const eiPath =
    `M${X(xMin)},${EY(0)}` +
    gx.map((x, i) => `L${X(x)},${EY(result.grid_ei[i])}`).join("") +
    `L${X(xMax)},${EY(0)}Z`;

  const rx = X(result.recommendation.x);
  const ry = Y(result.recommendation.predicted_mean);

  // Y-axis ticks (main panel).
  const yticks = 4;
  const tickVals = Array.from({ length: yticks + 1 }, (_, i) => yLo - yPad + (i / yticks) * (yHi + yPad - (yLo - yPad)));
  // X-axis ticks.
  const xticks = 5;
  const xtickVals = Array.from({ length: xticks + 1 }, (_, i) => xMin + (i / xticks) * (xMax - xMin));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img">
      {/* Y grid + labels */}
      {tickVals.map((v) => (
        <g key={`y${v}`}>
          <line x1={ML} y1={Y(v)} x2={W - MR} y2={Y(v)} stroke={GRID} strokeOpacity={0.15} />
          <text x={ML - 8} y={Y(v) + 4} textAnchor="end" fontSize="11" fill={GRID}>
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      {/* X labels */}
      {xtickVals.map((v) => (
        <text key={`x${v}`} x={X(v)} y={EI_BOT + 18} textAnchor="middle" fontSize="11" fill={GRID}>
          {v.toFixed(1)}
        </text>
      ))}

      {/* 95% band + mean */}
      <path d={bandPath} fill={ACCENT} fillOpacity={0.16} />
      <path d={meanPath} fill="none" stroke={ACCENT} strokeWidth={2} />

      {/* recommendation marker */}
      <line x1={rx} y1={MAIN_TOP} x2={rx} y2={EI_BOT} stroke={GREEN} strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={rx} y={ry - 14} textAnchor="middle" fontSize="20" fill={GREEN}>
        ★
      </text>

      {/* observed points */}
      {result.points.map((p) => (
        <g key={p.sample_id}>
          <circle cx={X(p.x)} cy={Y(p.y)} r={5} fill={POINT} />
          <text x={X(p.x)} y={Y(p.y) - 9} textAnchor="middle" fontSize="10" fill={POINT}>
            {p.sample_name}
          </text>
        </g>
      ))}

      {/* axis titles */}
      <text x={(ML + W - MR) / 2} y={H - 4} textAnchor="middle" fontSize="12" fill={GRID}>
        {axisLabel(result.input_variable)}
      </text>
      <text
        x={14}
        y={(MAIN_TOP + MAIN_BOT) / 2}
        textAnchor="middle"
        fontSize="12"
        fill={GRID}
        transform={`rotate(-90 14 ${(MAIN_TOP + MAIN_BOT) / 2})`}
      >
        {axisLabel(result.target_property)}
      </text>

      {/* EI strip */}
      <path d={eiPath} fill={GREEN} fillOpacity={0.28} />
      <line
        x1={ML}
        y1={EY(result.noise_threshold)}
        x2={W - MR}
        y2={EY(result.noise_threshold)}
        stroke={GREEN}
        strokeOpacity={0.5}
        strokeDasharray="2 3"
      />
      <text x={ML} y={EI_TOP - 4} fontSize="10" fill={GRID}>
        expected improvement (dashed = measurement-noise floor)
      </text>
    </svg>
  );
}
