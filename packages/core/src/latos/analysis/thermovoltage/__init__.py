"""Thermovoltage analysis — the Seebeck coefficient as a fitted quantity.

Ionic thermoelectric cells report a voltage that mixes the thermovoltage with
an electrode-polarisation term. Fitting across several temperature differences
separates the two and attaches an uncertainty to each, which a single reading
cannot do.
"""

from latos.analysis.thermovoltage.slope import ThermovoltageSlopeAnalyzer, fit_seebeck_slope

__all__ = ["ThermovoltageSlopeAnalyzer", "fit_seebeck_slope"]
