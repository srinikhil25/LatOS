/** Theme-aware uPlot line chart.
 *
 * uPlot is ~50 kB and renders 100k+ points at 60 fps — the right tool
 * for instrument traces. This wrapper owns sizing (ResizeObserver) and
 * pulls colors from the Latos CSS tokens so plots match the theme.
 */

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export interface LinePlotProps {
  x: (number | null)[];
  y: (number | null)[];
  xLabel: string;
  yLabel: string;
  height?: number;
  /** Optional highlighted point (e.g. the extracted peak of a shock trace). */
  peak?: { x: number; y: number } | null;
}

export function LinePlot({ x, y, xLabel, yLabel, height = 320, peak = null }: LinePlotProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const textColor = cssVar("--latos-text-secondary");
    const gridColor = cssVar("--latos-border");
    const accent = cssVar("--latos-accent");
    const markColor = cssVar("--latos-severity-warning") || "#c0392b";

    const make = (width: number) => {
      plotRef.current?.destroy();
      plotRef.current = new uPlot(
        {
          width,
          height,
          // Instrument traces are XY data, not time series.
          scales: { x: { time: false } },
          series: [
            { label: xLabel },
            {
              label: yLabel,
              stroke: accent,
              width: 2,
              spanGaps: false, // NaN gaps stay visible as gaps
            },
          ],
          axes: [
            {
              label: xLabel,
              stroke: textColor,
              grid: { stroke: gridColor, width: 1 },
              ticks: { stroke: gridColor },
            },
            {
              label: yLabel,
              stroke: textColor,
              grid: { stroke: gridColor, width: 1 },
              ticks: { stroke: gridColor },
            },
          ],
          legend: { show: false },
          cursor: { drag: { x: true, y: false } }, // drag = zoom X
          // Draw a filled marker at the highlighted point (e.g. the peak),
          // in canvas space so it tracks zoom/pan.
          hooks: peak
            ? {
                draw: [
                  (u: uPlot) => {
                    const cx = u.valToPos(peak.x, "x", true);
                    const cy = u.valToPos(peak.y, "y", true);
                    const ctx = u.ctx;
                    ctx.save();
                    ctx.fillStyle = markColor;
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                  },
                ],
              }
            : {},
        },
        [x, y] as uPlot.AlignedData,
        host,
      );
    };

    make(host.clientWidth || 600);
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width && plotRef.current) {
        plotRef.current.setSize({ width, height });
      }
    });
    observer.observe(host);

    return () => {
      observer.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
  }, [x, y, xLabel, yLabel, height, peak?.x, peak?.y]);

  return <div ref={hostRef} className="w-full" />;
}
