/** AnalysisLoader — an honest "working" indicator for long computations.
 *
 * Some Latos operations (building the cross-sample feature table, batch
 * peak fits, an optimization run) legitimately take tens of seconds. A
 * static line of text reads as a frozen app; this component keeps the user
 * oriented instead, with three honest signals:
 *
 *   • motion — a spinner and an indeterminate sweep say "still alive";
 *   • a live elapsed-seconds counter — sets expectations and proves progress;
 *   • a rotating description of the real work in flight — tells the user what
 *     the compute is actually doing, not a fabricated percentage.
 *
 * The task duration is genuinely unknown ahead of time, so we deliberately
 * do NOT show a completion bar — a fake percentage would misrepresent the
 * work. `stages` should describe the actual steps of the computation.
 */

import { useEffect, useState } from "react";

export function AnalysisLoader({
  title = "Working…",
  stages = [],
}: {
  title?: string;
  stages?: string[];
}) {
  const [seconds, setSeconds] = useState(0);
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const clock = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(clock);
  }, []);

  useEffect(() => {
    if (stages.length <= 1) return;
    const rotate = setInterval(
      () => setStage((i) => (i + 1) % stages.length),
      2200,
    );
    return () => clearInterval(rotate);
  }, [stages.length]);

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-lg border border-edge bg-surface px-6 py-6"
    >
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-edge border-t-accent"
        />
        <span className="text-sm font-medium">{title}</span>
        <span className="ml-auto font-mono text-xs tabular-nums text-secondary">
          {seconds}s
        </span>
      </div>

      {/* Indeterminate sweep — motion only, never a false completion fraction. */}
      <div className="mt-4 h-1 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full w-1/3 rounded-full bg-accent"
          style={{ animation: "latos-indeterminate 1.4s ease-in-out infinite" }}
        />
      </div>

      {stages.length > 0 && (
        <div className="mt-3 text-xs text-secondary">{stages[stage]}</div>
      )}
    </div>
  );
}
