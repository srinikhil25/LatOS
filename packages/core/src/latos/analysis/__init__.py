"""Analysis layer: turn parsed Measurements into derived AnalysisResults.

Public API surface:

- `BaseAnalyzer`, `AnalyzerInputs`, `AnalyzerOutput` — the contract every
  analyzer implements.
- `AnalyzerRegistry`, `default_registry` — discovery / dispatch.
- `AnalysisService`, `AnalysisRunOutcome` — orchestration + caching +
  persistence.

The contract types mirror Stage 1's parser framework (`BaseParser` /
`ParsedData`) so anyone who has read the ingestion layer can read this
one without learning a new vocabulary.
"""

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.analysis.registry import AnalyzerRegistry, default_registry
from latos.analysis.service import AnalysisRunOutcome, AnalysisService

__all__ = [
    "AnalysisRunOutcome",
    "AnalysisService",
    "AnalyzerInputs",
    "AnalyzerOutput",
    "AnalyzerRegistry",
    "BaseAnalyzer",
    "default_registry",
]
