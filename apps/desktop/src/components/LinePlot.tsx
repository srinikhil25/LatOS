/** Theme-aware uPlot line chart.
 *
 * uPlot is ~50 kB and renders 100k+ points at 60 fps — the right tool
 * for instrument traces. This wrapper owns sizing (ResizeObserver) and
 * pulls colors from the Latos CSS tokens so plots match the theme.
 *
 * Those colors are read once, when the canvas is built. The surrounding
 * CSS re-themes itself the moment the OS scheme changes, but an already
 * painted canvas does not, which leaves the dark grid drawn over a white
 * page — and, because exports copy the canvas as-is, bakes that into a
 * saved figure. So we watch the media query and rebuild on a change.
 */

import { useEffect, useRef, useState } from "react";
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
}

export function LinePlot({ x, y, xLabel, yLabel, height = 320 }: LinePlotProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [darkScheme, setDarkScheme] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => setDarkScheme(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const textColor = cssVar("--latos-text-secondary");
    const gridColor = cssVar("--latos-border");
    const accent = cssVar("--latos-accent");

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
    // `darkScheme` is not read in here: it is the signal to rebuild the
    // canvas so the token reads above pick up the new palette.
  }, [x, y, xLabel, yLabel, height, darkScheme]);

  return <div ref={hostRef} className="w-full" />;
}
