"""Tests for the cross-property correlation engine."""

from __future__ import annotations

import pytest

from latos.reporting.correlation import correlate


def _rows(**cols: list[float | None]) -> list[dict[str, float | None]]:
    """Column dict -> per-sample row dicts."""
    n = len(next(iter(cols.values())))
    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


class TestCorrelate:
    def test_perfect_positive_and_negative(self):
        rows = _rows(a=[1.0, 2.0, 3.0, 4.0], b=[2.0, 4.0, 6.0, 8.0], c=[8.0, 6.0, 4.0, 2.0])
        res = correlate(["a", "b", "c"], rows)
        by_pair = {(p.property_a, p.property_b): p for p in res.pairs}
        assert by_pair[("a", "b")].pearson == pytest.approx(1.0)
        assert by_pair[("a", "c")].pearson == pytest.approx(-1.0)
        # Matrix is symmetric with a unit diagonal.
        assert res.matrix[0][0] == 1.0
        assert res.matrix[0][1] == res.matrix[1][0]

    def test_pairs_ranked_by_absolute_strength(self):
        rows = _rows(
            a=[1.0, 2.0, 3.0, 4.0, 5.0],
            b=[1.0, 2.0, 3.0, 4.0, 5.0],  # r=+1 with a
            c=[5.0, 4.0, 2.9, 2.0, 1.2],  # strong negative, not perfect
        )
        res = correlate(["a", "b", "c"], rows)
        assert abs(res.pairs[0].pearson) >= abs(res.pairs[-1].pearson)
        assert (res.pairs[0].property_a, res.pairs[0].property_b) == ("a", "b")

    def test_spearman_catches_monotonic_nonlinear(self):
        rows = _rows(a=[1.0, 2.0, 3.0, 4.0], b=[1.0, 4.0, 9.0, 16.0])  # b = a²
        c = next(p for p in correlate(["a", "b"], rows).pairs)
        assert c.spearman == pytest.approx(1.0)  # perfectly monotonic

    def test_uses_only_shared_samples(self):
        rows = _rows(
            a=[1.0, 2.0, 3.0, 4.0, None],
            b=[2.0, 4.0, 6.0, 8.0, 99.0],  # last has no 'a'
        )
        c = next(p for p in correlate(["a", "b"], rows).pairs)
        assert c.n == 4  # the None-'a' row is dropped
        assert c.pearson == pytest.approx(1.0)

    def test_too_few_shared_samples_skipped(self):
        rows = _rows(a=[1.0, 2.0, None, None], b=[1.0, 2.0, 3.0, 4.0])
        res = correlate(["a", "b"], rows)
        assert res.pairs == []
        assert res.matrix[0][1] is None

    def test_constant_column_is_none(self):
        rows = _rows(a=[5.0, 5.0, 5.0, 5.0], b=[1.0, 2.0, 3.0, 4.0])
        res = correlate(["a", "b"], rows)
        assert res.matrix[0][1] is None
        assert res.pairs == []
