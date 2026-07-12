"""Fit reports — a `FitResult` into paste-ready text for a paper or notebook.

Three renderings of the same fit:

* **Markdown** — a human-readable report (goodness-of-fit + per-component
  table with ±1σ) to drop into a notebook or an issue.
* **CSV** — one row per fitted parameter (name, value, stderr) for a
  spreadsheet or supplementary data file.
* **LaTeX** — a ``tabular`` of the components ready for a manuscript.

Everything here is pure formatting — no fitting, no I/O.
"""

from __future__ import annotations

from latos.fitting.engine import FitResult

__all__ = ["csv_table", "latex_table", "markdown_report"]


def _fmt(value: float | None, stderr: float | None = None) -> str:
    if value is None:
        return "—"
    if stderr is None:
        return f"{value:.5g}"
    return f"{value:.5g} ± {stderr:.2g}"


def markdown_report(result: FitResult, *, title: str = "Fit report") -> str:
    """A Markdown report: goodness-of-fit plus a per-component table."""
    lines = [
        f"# {title}",
        "",
        f"- **Converged:** {'yes' if result.success else 'no'}",
        f"- **R²:** {result.r_squared:.5f}",
        f"- **χ²:** {result.chi_square:.5g}",
        f"- **Reduced χ²:** {result.reduced_chi_square:.5g}",
        f"- **Peaks:** {len(result.components)}",
        "",
        "| Peak | Center | Area (amplitude) | FWHM | Height |",
        "| --- | --- | --- | --- | --- |",
    ]
    stderr = {name: se for name, (_, se) in result.params.items()}
    for i, c in enumerate(result.components):
        lines.append(
            f"| {i + 1} "
            f"| {_fmt(c.center, stderr.get(f'{c.prefix}center'))} "
            f"| {_fmt(c.amplitude, stderr.get(f'{c.prefix}amplitude'))} "
            f"| {_fmt(c.fwhm, stderr.get(f'{c.prefix}fwhm'))} "
            f"| {_fmt(c.height, stderr.get(f'{c.prefix}height'))} |"
        )
    return "\n".join(lines)


def csv_table(result: FitResult) -> str:
    """CSV of every fitted parameter: ``parameter,value,stderr``."""
    rows = ["parameter,value,stderr"]
    for name, (value, stderr) in result.params.items():
        rows.append(f"{name},{value:.8g},{'' if stderr is None else f'{stderr:.8g}'}")
    return "\n".join(rows)


def latex_table(result: FitResult, *, caption: str = "Fit parameters") -> str:
    """A LaTeX ``tabular`` of the fitted components (center, area, FWHM)."""
    stderr = {name: se for name, (_, se) in result.params.items()}
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{cccc}",
        "\\hline",
        "Peak & Center & Area & FWHM \\\\",
        "\\hline",
    ]
    for i, c in enumerate(result.components):
        center = _fmt(c.center, stderr.get(f"{c.prefix}center")).replace("±", "$\\pm$")
        area = _fmt(c.amplitude, stderr.get(f"{c.prefix}amplitude")).replace("±", "$\\pm$")
        fwhm = _fmt(c.fwhm, stderr.get(f"{c.prefix}fwhm")).replace("±", "$\\pm$")
        lines.append(f"{i + 1} & {center} & {area} & {fwhm} \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)
