"""Sample-level thermoelectric assembly: find R&S + LFA → run the kernel.

A sample's zT needs two measurements — a Resistivity/Seebeck run and an
LFA run. This module locates them among the sample's measurements (by
the arrays they carry, not by a fragile metadata flag), then hands the
raw arrays to `compute_zt`. Kept free of FastAPI/HTTP so it unit-tests
with a plain `load_arrays` callable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from latos.analysis.transport import ZtResult, compute_zt
from latos.core.enums import Technique

if TYPE_CHECKING:
    from latos.core.models import Sample

Arrays = dict[str, "NDArray[np.float64]"]
LoadArrays = Callable[[str], Arrays]

__all__ = ["sample_zt"]


def _is_lfa(arrays: Arrays) -> bool:
    return "thermal_conductivity" in arrays


def _is_resistivity_seebeck(arrays: Arrays) -> bool:
    return "resistivity_uohm_m" in arrays and "seebeck_uv_k" in arrays


def sample_zt(sample: Sample, load_arrays: LoadArrays) -> ZtResult:
    """Compute zT(T) for `sample` from its R&S + LFA measurements.

    Args:
        sample: The sample whose thermoelectric measurements to combine.
        load_arrays: Maps a measurement id to its stored arrays
            (``{name: ndarray}``); empty dict if none.

    Raises:
        TransportError: if the sample lacks a usable R&S or LFA
            measurement (the message names what's missing), or if the
            arrays are inconsistent (propagated from `compute_zt`).
    """
    rs: Arrays | None = None
    lfa: Arrays | None = None
    for m in sample.measurements:
        if m.technique is not Technique.THERMOELECTRIC:
            continue
        arrays = load_arrays(m.id)
        if not arrays:
            continue
        if lfa is None and _is_lfa(arrays):
            lfa = arrays
        elif rs is None and _is_resistivity_seebeck(arrays):
            rs = arrays

    if rs is None or lfa is None:
        from latos.analysis.transport import TransportError  # noqa: PLC0415

        missing = []
        if rs is None:
            missing.append("a Resistivity/Seebeck measurement")
        if lfa is None:
            missing.append("an LFA (thermal conductivity) measurement")
        raise TransportError(
            f"Sample {sample.canonical_name!r} cannot compute zT — missing "
            + " and ".join(missing)
            + "."
        )

    return compute_zt(
        rs_temperature_k=rs["temperature_k"],
        resistivity_uohm_m=rs["resistivity_uohm_m"],
        seebeck_uv_k=rs["seebeck_uv_k"],
        lfa_temperature_k=lfa["temperature_k"],
        thermal_conductivity_w_mk=lfa["thermal_conductivity"],
    )
