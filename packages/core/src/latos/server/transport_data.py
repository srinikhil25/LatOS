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

from latos.analysis.transport import TransportError, ZtResult, compute_zt
from latos.core.enums import Technique
from latos.optimization import spb

if TYPE_CHECKING:
    from latos.core.models import Sample

Arrays = dict[str, "NDArray[np.float64]"]
LoadArrays = Callable[[str], Arrays]

__all__ = [
    "rs_conductivity_s_cm",
    "sample_spb_guidance",
    "sample_zt",
    "seebeck_sign",
]

# 1 µΩ·m = 1e-4 Ω·cm, so σ[S/cm] = 1 / ρ[Ω·cm] = 1e4 / ρ[µΩ·m].
_UOHM_M_TO_S_CM = 1e4


def rs_conductivity_s_cm(sample: Sample, load_arrays: LoadArrays) -> float | None:
    """Electrical conductivity from the R&S resistivity, near room temperature.

    The Hall measurement is single-temperature (≈ room T), so we take the R&S
    resistivity at its lowest measured temperature (closest to the Hall point)
    and convert to S/cm. Returns None if the sample has no usable R&S
    resistivity. This is the *independent* conductivity the Hall σ is checked
    against (a cross-technique consistency test).
    """
    for m in sample.measurements:
        if m.technique is not Technique.THERMOELECTRIC:
            continue
        arrays = load_arrays(m.id)
        rho = arrays.get("resistivity_uohm_m")
        temp = arrays.get("temperature_k")
        if rho is None or len(rho) == 0:
            continue
        rho_arr = np.asarray(rho, dtype=float)
        if temp is not None and len(temp) == len(rho_arr):
            rho_room = float(rho_arr[int(np.argmin(np.asarray(temp, dtype=float)))])
        else:
            rho_room = float(np.nanmin(rho_arr))
        if rho_room > 0:
            return _UOHM_M_TO_S_CM / rho_room
    return None


def seebeck_sign(sample: Sample, load_arrays: LoadArrays) -> float | None:
    """Sign of the sample's Seebeck coefficient: +1 (p-type) / -1 (n-type).

    Read from the sample's Resistivity/Seebeck measurement (median over
    the temperature sweep — robust to a noisy endpoint). None when the
    sample has no R&S data or the median is exactly zero. Used as the
    independent carrier-type determination the Hall analyzer checks
    itself against.
    """
    for m in sample.measurements:
        if m.technique is not Technique.THERMOELECTRIC:
            continue
        arrays = load_arrays(m.id)
        s = arrays.get("seebeck_uv_k")
        if s is not None and len(s) > 0:
            median = float(np.median(s))
            if median != 0.0:
                return 1.0 if median > 0 else -1.0
    return None


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


def sample_spb_guidance(sample: Sample, load_arrays: LoadArrays) -> spb.SpbGuidance | None:
    """Single-parabolic-band read on `sample` from its (Seebeck, zT) at peak zT.

    Pairs the Seebeck coefficient measured *at the zT-peak temperature* with the
    peak zT, then asks the SPB physics model where the material sits relative to
    its own zT optimum (see `latos.optimization.spb.guidance`). Uses the Seebeck
    axis only — reliable even when the Hall carrier concentration is noise.

    Returns None when the sample cannot derive zT (missing R&S/LFA), so callers
    can skip it without special-casing.
    """
    try:
        zt = sample_zt(sample, load_arrays)
    except TransportError:
        return None
    temp = np.asarray(zt.temperature_k, dtype=float)
    seebeck_v_k = np.asarray(zt.seebeck_v_k, dtype=float)
    if temp.size == 0 or seebeck_v_k.size != temp.size:
        return None
    # Seebeck at the zT-peak temperature, converted V/K -> µV/K.
    seebeck_uv_k = float(np.interp(zt.peak_zt_temperature_k, temp, seebeck_v_k)) * 1e6
    return spb.guidance(seebeck_uv_k, zt.peak_zt)
