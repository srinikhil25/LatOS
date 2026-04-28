"""Concrete parsers, one per technique x file format.

Every parser is a `BaseParser` subclass. They are registered with the
`ParserRegistry` (Stage 1C.5) so the orchestrator can dispatch by
confidence-pick. Modules here should import nothing from each other —
each parser is independent of every other parser.
"""

from __future__ import annotations

from latos.ingestion.parsers.eds_bruker_spx import BrukerSpxParser
from latos.ingestion.parsers.hall_xls import HallXlsParser
from latos.ingestion.parsers.microscopy_tif import MicroscopyTifParser
from latos.ingestion.parsers.thermoelectric_xlsx import ThermoelectricXlsxParser
from latos.ingestion.parsers.uvdrs_xlsx import UvDrsXlsxParser
from latos.ingestion.parsers.xps_casaxps_csv import CasaXpsCsvParser
from latos.ingestion.parsers.xrd_panalytical_xrdml import PanalyticalXrdmlParser
from latos.ingestion.parsers.xrd_rigaku_asc import RigakuXrdAscParser
from latos.ingestion.parsers.xrd_rigaku_txt import RigakuXrdTxtParser

__all__ = [
    "BrukerSpxParser",
    "CasaXpsCsvParser",
    "HallXlsParser",
    "MicroscopyTifParser",
    "PanalyticalXrdmlParser",
    "RigakuXrdAscParser",
    "RigakuXrdTxtParser",
    "ThermoelectricXlsxParser",
    "UvDrsXlsxParser",
]
