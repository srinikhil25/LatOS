/** ChartFrame — wraps any figure and gives it a "save as PNG" action.
 *
 * Researchers need these plots in reports, slides and emails, and
 * re-plotting by hand in Excel is how a figure stops matching the data it
 * came from. One button, on every chart, keeps the exported figure the
 * same object the tool computed.
 *
 * Place it *inside* the existing chart card so the surrounding card
 * styling stays where it is:
 *
 *   <div className="rounded-lg border border-edge bg-surface p-3">
 *     <ChartFrame basename="latos-optimize">
 *       <OptimizeChart result={result} />
 *     </ChartFrame>
 *   </div>
 */

import { useCallback, useRef, useState } from "react";
import { exportChartPng } from "../lib/exportChart";

type Phase = "idle" | "saving" | "saved" | "error";

export interface ChartFrameProps {
  /** File name stem; a timestamp and `.png` are appended on save. */
  basename: string;
  /** Accessible description of what is being saved. */
  label?: string;
  children: React.ReactNode;
}

export function ChartFrame({ basename, label = "figure", children }: ChartFrameProps) {
  const figureRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  const save = useCallback(async () => {
    const root = figureRef.current;
    if (!root) return;
    setPhase("saving");
    setError(null);
    try {
      await exportChartPng(root, basename);
      setPhase("saved");
      setTimeout(() => setPhase("idle"), 2000);
    } catch (err) {
      setPhase("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [basename]);

  return (
    <div>
      <div ref={figureRef}>{children}</div>
      <div className="mt-2 flex items-center justify-end gap-2">
        {phase === "error" && error && (
          <span className="text-xs text-severity-error" data-selectable>
            {error}
          </span>
        )}
        <button
          type="button"
          onClick={() => void save()}
          disabled={phase === "saving"}
          title={`Save this ${label} as a PNG`}
          aria-label={`Save this ${label} as a PNG`}
          className="rounded-md border border-edge px-2.5 py-1 text-xs text-secondary transition enabled:hover:border-accent enabled:hover:text-accent disabled:opacity-50"
        >
          {phase === "saving" ? "Saving…" : phase === "saved" ? "Saved ✓" : "↓ PNG"}
        </button>
      </div>
    </div>
  );
}
