"""Tests for `latos.server.transport_data.sample_zt`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.analysis.transport import TransportError
from latos.core.enums import FileRole, Technique
from latos.core.models import FileRef, Measurement, Sample, new_id, utc_now
from latos.server.transport_data import sample_zt


def _meas(sample_id: str, technique: Technique = Technique.THERMOELECTRIC) -> Measurement:
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=technique,
        instrument=None,
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(
            FileRef(
                path=Path("D:/x.xlsx"),
                sha256="0" * 64,
                size_bytes=1,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            ),
        ),
        issues=(),
        parsed_data_path=None,
    )


_LFA = {
    "temperature_k": np.array([300.0, 400.0, 500.0, 600.0]),
    "thermal_conductivity": np.array([5.0, 4.5, 4.2, 4.0]),
}
_RS = {
    "temperature_k": np.array([316.0, 400.0, 500.0, 600.0]),
    "resistivity_uohm_m": np.array([0.12, 0.18, 0.24, 0.29]),
    "seebeck_uv_k": np.array([8.0, 14.0, 22.0, 32.0]),
}


def _sample_with() -> tuple[Sample, str, str]:
    """Return (sample, lfa_measurement_id, rs_measurement_id)."""
    sid = new_id()
    lfa = _meas(sid)
    rs = _meas(sid)
    sample = Sample(id=sid, project_id=new_id(), canonical_name="CS", measurements=(lfa, rs))
    return sample, lfa.id, rs.id


def _loader(mapping: dict):
    return lambda mid: mapping.get(mid, {})


def test_computes_zt_from_both_measurements():
    sample, lfa_id, rs_id = _sample_with()
    result = sample_zt(sample, _loader({lfa_id: _LFA, rs_id: _RS}))
    assert 0.0 < result.peak_zt < 1.5
    assert result.temperature_k.tolist() == _LFA["temperature_k"].tolist()


def test_missing_rs_raises_with_message():
    sample, lfa_id, _rs_id = _sample_with()
    with pytest.raises(TransportError, match="Resistivity/Seebeck"):
        sample_zt(sample, _loader({lfa_id: _LFA}))


def test_missing_lfa_raises_with_message():
    sample, _lfa_id, rs_id = _sample_with()
    with pytest.raises(TransportError, match="LFA"):
        sample_zt(sample, _loader({rs_id: _RS}))


def test_non_thermoelectric_measurements_ignored():
    sid = new_id()
    xrd = _meas(sid, technique=Technique.XRD)
    sample = Sample(id=sid, project_id=new_id(), canonical_name="CS", measurements=(xrd,))
    with pytest.raises(TransportError):
        sample_zt(sample, lambda _mid: {})
