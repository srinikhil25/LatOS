/** Save any figure the app draws as a PNG the researcher can keep.
 *
 * A figure is worth nothing if it is trapped in the window. Every plot
 * here is either an inline `<svg>` (Optimize, Fit), a uPlot `<canvas>`
 * (sample traces), or an `<img>` the core rendered (Correlate), so this
 * module handles all three behind one call.
 *
 * Two details matter for fidelity:
 *
 *  - Our SVGs paint with theme tokens (`var(--latos-edge)`). Those are
 *    defined on `:root`, so a serialized SVG loses them and renders
 *    black-on-black. We therefore re-declare the resolved tokens inside
 *    the clone, which also means an export matches the theme on screen.
 *  - An SVG has no intrinsic background. Exporting onto transparency
 *    gives a figure that is unreadable in a light document if the app is
 *    in dark mode, so we paint the surface colour underneath.
 */

const TOKEN_PREFIX = "--latos-";

/** Fallback token list, used only if the stylesheet cannot be enumerated. */
const FALLBACK_TOKENS = [
  "--latos-accent",
  "--latos-surface",
  "--latos-muted-surface",
  "--latos-border",
  "--latos-edge",
  "--latos-text",
  "--latos-text-secondary",
];

/** Every `:root` rule we can read, cheapest-first (dev `<style>`, then bundle). */
function rootRules(): CSSStyleRule[] {
  const out: CSSStyleRule[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    let rules: CSSRuleList;
    try {
      rules = sheet.cssRules; // same-origin in dev and in the bundle
    } catch {
      continue; // a foreign sheet — nothing we need is in there
    }
    for (const rule of Array.from(rules)) {
      if (rule instanceof CSSStyleRule) out.push(rule);
    }
  }
  return out;
}

/** Theme custom-property names, read from the CSSOM (values come later). */
function tokenNames(): string[] {
  const found = new Set<string>();
  for (const rule of rootRules()) {
    for (const prop of Array.from(rule.style)) {
      if (prop.startsWith(TOKEN_PREFIX)) found.add(prop);
    }
  }
  if (found.size === 0) FALLBACK_TOKENS.forEach((t) => found.add(t));
  return Array.from(found);
}

/**
 * The light token values, taken from the bare `:root` rule.
 *
 * Dark mode lives in an `@media (prefers-color-scheme: dark)` block, so a
 * top-level `:root` rule (`parentRule === null`) always carries the light
 * palette regardless of what the OS is currently asking for.
 */
function lightTokens(): Map<string, string> {
  const values = new Map<string, string>();
  for (const rule of rootRules()) {
    if (rule.parentRule !== null) continue; // skip the dark-mode override
    if (!rule.selectorText.split(",").some((s) => s.trim() === ":root")) continue;
    for (const prop of Array.from(rule.style)) {
      if (!prop.startsWith(TOKEN_PREFIX)) continue;
      const value = rule.style.getPropertyValue(prop).trim();
      if (value !== "") values.set(prop, value);
    }
  }
  return values;
}

export type ExportTheme = "light" | "active";

/** `--token: value;` declarations for the requested export theme. */
function resolvedTokenCss(theme: ExportTheme): string {
  const light = theme === "light" ? lightTokens() : new Map<string, string>();
  const root = getComputedStyle(document.documentElement);
  return tokenNames()
    .map((name) => [name, light.get(name) ?? root.getPropertyValue(name).trim()] as const)
    .filter(([, value]) => value !== "")
    .map(([name, value]) => `${name}:${value};`)
    .join("");
}

function surfaceColor(theme: ExportTheme): string {
  if (theme === "light") {
    const light = lightTokens().get("--latos-surface");
    if (light) return light;
    return "#ffffff";
  }
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue("--latos-surface")
    .trim();
  return value || "#ffffff";
}

function fontStack(el: Element): string {
  const family = getComputedStyle(el).fontFamily;
  return family || "system-ui, sans-serif";
}

/** Pixel size of an SVG, from its viewBox (our SVGs are viewBox-sized). */
function svgSize(svg: SVGSVGElement): { width: number; height: number } {
  const box = svg.viewBox?.baseVal;
  if (box && box.width > 0 && box.height > 0) {
    return { width: box.width, height: box.height };
  }
  const rect = svg.getBoundingClientRect();
  return { width: rect.width || 720, height: rect.height || 470 };
}

function drawToBlob(
  draw: (ctx: CanvasRenderingContext2D) => void,
  width: number,
  height: number,
  background: string,
): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get a 2D context for the export.");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  draw(ctx);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("The browser refused to encode the PNG."));
    }, "image/png");
  });
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not rasterize the figure."));
    img.src = src;
  });
}

/** Rasterize an inline SVG, carrying the theme tokens and font across. */
async function svgToPngBlob(
  svg: SVGSVGElement,
  scale: number,
  background: string,
  theme: ExportTheme,
): Promise<Blob> {
  const { width, height } = svgSize(svg);
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));

  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = `svg{${resolvedTokenCss(theme)}font-family:${fontStack(svg)};}`;
  clone.insertBefore(style, clone.firstChild);

  const xml = new XMLSerializer().serializeToString(clone);
  const img = await loadImage(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(xml)}`);
  return drawToBlob(
    (ctx) => ctx.drawImage(img, 0, 0, width * scale, height * scale),
    width * scale,
    height * scale,
    background,
  );
}

/** Copy a uPlot canvas. Kept at its own resolution — upscaling only blurs. */
function canvasToPngBlob(canvas: HTMLCanvasElement, background: string): Promise<Blob> {
  return drawToBlob((ctx) => ctx.drawImage(canvas, 0, 0), canvas.width, canvas.height, background);
}

/** An `<img>` is already a bitmap from the core; keep it byte-for-byte. */
async function imgToPngBlob(img: HTMLImageElement): Promise<Blob> {
  const response = await fetch(img.src);
  if (!response.ok) throw new Error(`The core returned ${response.status} for that figure.`);
  return response.blob();
}

function timestamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke on the next tick — Chromium needs the URL alive for the click.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export interface ExportChartOptions {
  /** Resolution multiplier for vector figures. 2 keeps text crisp in slides. */
  scale?: number;
  /** Override the painted background (defaults to the export theme's surface). */
  background?: string;
  /**
   * Which palette the exported figure uses.
   *
   * Defaults to `"light"`: an exported figure is destined for a report,
   * a paper or a slide, and those are on white. A dark-mode figure pasted
   * into a white document looks like a mistake. Pass `"active"` to get
   * exactly what is on screen instead.
   *
   * Only vector figures can be re-themed. A uPlot canvas has already been
   * painted in the active theme, so it is always exported as-is.
   */
  theme?: ExportTheme;
}

/**
 * Find the figure inside `root` and save it as a PNG.
 *
 * `basename` gets a timestamp appended so repeated exports never silently
 * overwrite each other in the downloads folder.
 */
export async function exportChartPng(
  root: HTMLElement,
  basename: string,
  { scale = 2, background, theme = "light" }: ExportChartOptions = {},
): Promise<void> {
  const svg = root.querySelector("svg");
  const canvas = root.querySelector("canvas");
  const img = root.querySelector("img");

  let blob: Blob;
  if (svg) {
    blob = await svgToPngBlob(
      svg as SVGSVGElement,
      scale,
      background ?? surfaceColor(theme),
      theme,
    );
  } else if (canvas) {
    // Already rasterized in the active theme — re-theming would lie about it.
    blob = await canvasToPngBlob(
      canvas as HTMLCanvasElement,
      background ?? surfaceColor("active"),
    );
  } else if (img) {
    blob = await imgToPngBlob(img as HTMLImageElement);
  } else {
    throw new Error("There is no figure here to export.");
  }

  saveBlob(blob, `${basename}-${timestamp()}.png`);
}
