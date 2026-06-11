/** Start screen — hero, Open Folder, recent projects, ingest progress.
 *
 * The web twin of the Qt hub's start view. Recents persist in
 * localStorage (the project's own truth lives in `<root>/.latos/`;
 * this list is only a convenience).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { open as openFolderDialog } from "@tauri-apps/plugin-dialog";
import {
  openProject,
  subscribeIngestEvents,
  type IngestProgress,
} from "../lib/api";

const RECENTS_KEY = "latos.recentProjects";
const RECENTS_MAX = 8;

interface RecentEntry {
  name: string;
  path: string;
}

function loadRecents(): RecentEntry[] {
  try {
    return JSON.parse(localStorage.getItem(RECENTS_KEY) ?? "[]") as RecentEntry[];
  } catch {
    return [];
  }
}

function pushRecent(path: string): RecentEntry[] {
  const name = path.replaceAll("\\", "/").split("/").filter(Boolean).pop() ?? path;
  const next = [
    { name, path },
    ...loadRecents().filter((entry) => entry.path !== path),
  ].slice(0, RECENTS_MAX);
  localStorage.setItem(RECENTS_KEY, JSON.stringify(next));
  return next;
}

type Phase = "idle" | "ingesting" | "error";

export function Start({ onProjectReady }: { onProjectReady: () => void }) {
  const [recents, setRecents] = useState<RecentEntry[]>(loadRecents);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<IngestProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const disposeRef = useRef<(() => void) | null>(null);

  useEffect(() => () => disposeRef.current?.(), []);

  const ingest = useCallback(
    async (root: string) => {
      setPhase("ingesting");
      setProgress(null);
      setError(null);
      setRecents(pushRecent(root));
      try {
        await openProject(root);
      } catch (err) {
        setPhase("error");
        setError(err instanceof Error ? err.message : String(err));
        return;
      }
      disposeRef.current = subscribeIngestEvents({
        onProgress: setProgress,
        onDone: onProjectReady,
        onError: (message) => {
          setPhase("error");
          setError(message);
        },
      });
    },
    [onProjectReady],
  );

  const pickFolder = useCallback(async () => {
    const chosen = await openFolderDialog({ directory: true, multiple: false });
    if (typeof chosen === "string") void ingest(chosen);
  }, [ingest]);

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col justify-center gap-10 px-10">
      <header className="space-y-2">
        <h1 className="text-4xl font-semibold tracking-tight">Latos</h1>
        <p className="text-lg text-secondary">
          Multi-modal materials characterization
        </p>
        <p className="text-sm text-secondary">
          Parse → Validate → Fit → Correlate → Store → Suggest the next experiment.
        </p>
      </header>

      {phase === "ingesting" ? (
        <section className="space-y-3 rounded-lg border border-edge bg-surface p-6">
          <h2 className="font-medium">Ingesting…</h2>
          <p className="text-sm text-secondary" data-selectable>
            {progress
              ? `File ${progress.index + 1} of ${progress.total} — ${progress.name}`
              : "Scanning folder…"}
          </p>
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-200"
              style={{
                width: progress
                  ? `${Math.round(((progress.index + 1) / Math.max(progress.total, 1)) * 100)}%`
                  : "8%",
              }}
            />
          </div>
        </section>
      ) : (
        <section className="space-y-6">
          <button
            type="button"
            onClick={() => void pickFolder()}
            className="rounded-md bg-accent px-5 py-2.5 font-medium text-white transition hover:brightness-110 active:brightness-95"
          >
            Open Folder
          </button>

          {error && (
            <p className="text-sm text-severity-error" data-selectable>
              {error}
            </p>
          )}

          <div className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-secondary">
              Recent
            </h2>
            {recents.length === 0 ? (
              <p className="text-sm text-secondary">No recent projects yet.</p>
            ) : (
              <ul className="space-y-1.5">
                {recents.map((entry) => (
                  <li key={entry.path}>
                    <button
                      type="button"
                      onClick={() => void ingest(entry.path)}
                      className="w-full rounded-md border border-edge bg-surface px-4 py-3 text-left transition hover:border-accent"
                    >
                      <span className="block font-medium">{entry.name}</span>
                      <span className="block text-xs text-secondary">{entry.path}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
