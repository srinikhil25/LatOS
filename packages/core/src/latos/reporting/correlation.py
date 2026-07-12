"""Cross-property correlation — the second Stage-6 layer.

Given a feature table (samples × properties), compute the pairwise
relationships between properties across samples: a Pearson *r* (linear) and
a Spearman *ρ* (monotonic), each over only the samples that have both
properties. Returns a full correlation matrix for a heatmap plus the
off-diagonal pairs ranked by |r|, so the strongest relationships surface
automatically — no technique pair is hard-coded.

Pure numpy/scipy: the caller supplies plain property names and per-sample
value dicts, so this layer knows nothing about the server or the feature
store.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["Correlation", "CorrelationResult", "correlate"]

# Fewer than this many shared samples and a correlation is meaningless.
_MIN_SAMPLES = 3


@dataclass(frozen=True)
class Correlation:
    """One property-pair relationship over the samples they share."""

    property_a: str
    property_b: str
    pearson: float
    spearman: float
    n: int  # shared, finite sample count


@dataclass(frozen=True)
class CorrelationResult:
    """A Pearson matrix (for a heatmap) plus the ranked off-diagonal pairs."""

    properties: list[str]
    matrix: list[list[float | None]]  # matrix[i][j] = Pearson(prop_i, prop_j)
    pairs: list[Correlation]  # |pearson| desc, strongest first


def _paired(a: list[float | None], b: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Rows where both values are present and finite."""
    xa: list[float] = []
    xb: list[float] = []
    for va, vb in zip(a, b, strict=True):
        if va is None or vb is None:
            continue
        if np.isfinite(va) and np.isfinite(vb):
            xa.append(va)
            xb.append(vb)
    return np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < _MIN_SAMPLES or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < _MIN_SAMPLES or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho = float(stats.spearmanr(x, y).statistic)
    return rho if np.isfinite(rho) else None


def correlate(
    properties: list[str],
    samples: list[dict[str, float | None]],
    *,
    min_samples: int = _MIN_SAMPLES,
) -> CorrelationResult:
    """Pairwise correlations across `properties` over `samples`.

    Args:
        properties: the column names to correlate.
        samples: one dict per sample, mapping property -> value (a property
            absent from a sample, or None/non-finite, is skipped for any
            pair involving it).
        min_samples: minimum shared samples for a pair to be reported.

    Returns a `CorrelationResult`; the matrix has `None` where a pair has
    too few shared samples or a constant column.
    """
    columns = {p: [s.get(p) for s in samples] for p in properties}
    n = len(properties)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    pairs: list[Correlation] = []

    for i, pa in enumerate(properties):
        matrix[i][i] = 1.0 if any(v is not None for v in columns[pa]) else None
        for j in range(i + 1, n):
            pb = properties[j]
            x, y = _paired(columns[pa], columns[pb])
            if x.size < min_samples:
                continue
            r = _safe_pearson(x, y)
            rho = _safe_spearman(x, y)
            matrix[i][j] = r
            matrix[j][i] = r
            if r is not None:
                pairs.append(
                    Correlation(
                        property_a=pa,
                        property_b=pb,
                        pearson=r,
                        spearman=rho if rho is not None else float("nan"),
                        n=int(x.size),
                    )
                )

    pairs.sort(key=lambda c: abs(c.pearson), reverse=True)
    return CorrelationResult(properties=properties, matrix=matrix, pairs=pairs)
