"""The generic layer stays generic: by its imports and by its vocabulary."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import latos

PACKAGE = Path(latos.__file__).parent
EXPERIMENTS = PACKAGE / "experiments"

# Every module directly inside `experiments/` is generic. Producers, which know
# a technique, live one level down and are not held to these rules.
GENERIC_MODULES = sorted(EXPERIMENTS.glob("*.py"))

# Words that belong to one technique, instrument or material.
TECHNIQUE_WORDS = (
    "thermal cycle",
    "seebeck",
    "hioki",
    "aliquat",
    "peltier",
    "thermovoltage",
    "ionic liquid",
)


def _imports(path: Path) -> set[str]:
    """Every module name `path` imports, at any nesting depth."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _importing(path: Path, prefix: str) -> list[str]:
    return sorted(n for n in _imports(path) if n == prefix or n.startswith(prefix + "."))


def test_there_are_generic_modules_to_check() -> None:
    assert {p.name for p in GENERIC_MODULES} >= {
        "definition.py",
        "validation.py",
        "values.py",
        "records.py",
        "observations.py",
        "build.py",
        "reduce.py",
        "prepare.py",
        "run.py",
        "identity.py",
    }


@pytest.mark.parametrize("module", GENERIC_MODULES, ids=lambda p: p.name)
@pytest.mark.parametrize(
    "forbidden",
    [
        "latos.ingestion",
        "latos.analysis",
        "latos.server",
        "latos.experiments.producers",
        "latos.campaign_cycle",
        "latos.campaign_trace",
    ],
)
def test_a_generic_module_imports_nothing_technique_specific(module: Path, forbidden: str) -> None:
    assert _importing(module, forbidden) == []


@pytest.mark.parametrize("module", GENERIC_MODULES, ids=lambda p: p.name)
def test_a_generic_module_names_no_technique(module: Path) -> None:
    text = module.read_text(encoding="utf-8").lower()
    found = [word for word in TECHNIQUE_WORDS if re.search(re.escape(word), text)]
    assert found == []


@pytest.mark.parametrize("package", ["optimization", "ingestion", "analysis"])
def test_the_layers_below_never_import_the_experiment_layer(package: str) -> None:
    offenders = {
        str(path.relative_to(PACKAGE)): _importing(path, "latos.experiments")
        for path in sorted((PACKAGE / package).rglob("*.py"))
        if _importing(path, "latos.experiments")
    }
    assert offenders == {}
