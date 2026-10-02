"""Thermovoltage analysis — the Seebeck coefficient as a fitted quantity.

Ionic thermoelectric cells report a voltage that mixes the thermovoltage with
an electrode-polarisation term. Fitting across several temperature differences
separates the two and attaches an uncertainty to each, which a single reading
cannot do.
"""

from latos.analysis.thermovoltage.slope import ThermovoltageSlopeAnalyzer, fit_seebeck_slope
from latos.analysis.thermovoltage.transient import (
    ApproachFit,
    ThermovoltageTransientAnalyzer,
    fit_exponential_approach,
)

__all__ = [
    "ApproachFit",
    "ThermovoltageSlopeAnalyzer",
    "ThermovoltageTransientAnalyzer",
    "fit_exponential_approach",
    "fit_seebeck_slope",
]
