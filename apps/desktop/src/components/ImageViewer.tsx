/** Zoom/pan viewer for a microscopy image.
 *
 * Wheel = zoom toward the cursor; drag = pan; double-click = reset.
 * The image is the rendered PNG served by the sidecar
 * (`/measurements/{id}/image`). Pure CSS transform — no canvas — so it
 * stays crisp at any zoom and is trivially fast.
 */

import { useCallback, useRef, useState } from "react";

const MIN_SCALE = 0.2;
const MAX_SCALE = 12;
const ZOOM_STEP = 1.0015; // per wheel delta unit

interface View {
  scale: number;
  x: number;
  y: number;
}

const INITIAL: View = { scale: 1, x: 0, y: 0 };

export function ImageViewer({ src, alt }: { src: string; alt: string }) {
  const [view, setView] = useState<View>(INITIAL);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const hostRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);

  const onWheel = useCallback((event: React.WheelEvent) => {
    event.preventDefault();
    const host = hostRef.current;
    if (!host) return;
    const rect = host.getBoundingClientRect();
    const cx = event.clientX - rect.left;
    const cy = event.clientY - rect.top;
    setView((v) => {
      const next = Math.min(
        MAX_SCALE,
        Math.max(MIN_SCALE, v.scale * ZOOM_STEP ** -event.deltaY),
      );
      const ratio = next / v.scale;
      // Keep the point under the cursor fixed while scaling.
      return {
        scale: next,
        x: cx - (cx - v.x) * ratio,
        y: cy - (cy - v.y) * ratio,
      };
    });
  }, []);

  const onPointerDown = useCallback(
    (event: React.PointerEvent) => {
      (event.target as HTMLElement).setPointerCapture(event.pointerId);
      dragRef.current = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
    },
    [view.x, view.y],
  );

  const onPointerMove = useCallback((event: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    setView((v) => ({
      ...v,
      x: drag.vx + (event.clientX - drag.x),
      y: drag.vy + (event.clientY - drag.y),
    }));
  }, []);

  const endDrag = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <div className="space-y-2">
      <div
        ref={hostRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onDoubleClick={() => setView(INITIAL)}
        className="relative h-[440px] cursor-grab touch-none overflow-hidden rounded-lg border border-edge bg-black active:cursor-grabbing"
      >
        {!loaded && !failed && (
          <div className="absolute inset-0 flex items-center justify-center text-secondary">
            Loading image…
          </div>
        )}
        {failed && (
          <div className="absolute inset-0 flex items-center justify-center text-secondary">
            Could not load image.
          </div>
        )}
        <img
          src={src}
          alt={alt}
          draggable={false}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            transformOrigin: "0 0",
          }}
          className="max-w-none select-none"
        />
      </div>
      <p className="text-center text-xs text-secondary">
        Scroll to zoom · drag to pan · double-click to reset
      </p>
    </div>
  );
}
