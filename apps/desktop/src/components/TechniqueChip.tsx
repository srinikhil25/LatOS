/** Technique identity badge — one hue per technique, app-wide.
 *
 * Extracted from Hub so Samples (and every future screen) renders the
 * identical chip.
 */

const TECH_VAR: Record<string, string> = {
  xrd: "--latos-tech-xrd",
  xps: "--latos-tech-xps",
  uv_drs: "--latos-tech-uv-drs",
  hall: "--latos-tech-hall",
  thermoelectric: "--latos-tech-thermoelectric",
  eds: "--latos-tech-eds",
  tem: "--latos-tech-tem",
  sem: "--latos-tech-sem",
  stem: "--latos-tech-stem",
  raman: "--latos-tech-raman",
};

export function techniqueLabel(technique: string): string {
  if (technique === "uv_drs") return "UV-DRS";
  return technique.toUpperCase();
}

export function TechniqueChip({ technique }: { technique: string }) {
  const cssVar = TECH_VAR[technique] ?? "--latos-tech-unknown";
  return (
    <span
      className="rounded-sm px-2 py-0.5 text-[11px] font-semibold"
      style={{
        color: `var(${cssVar})`,
        background: `color-mix(in srgb, var(${cssVar}) 18%, transparent)`,
      }}
    >
      {techniqueLabel(technique)}
    </span>
  );
}
