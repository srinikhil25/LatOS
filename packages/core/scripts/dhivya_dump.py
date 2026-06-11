"""One-shot script to dump Dhivya Stage 1/Stage 2 numbers for docs.

Not a permanent script. Lives here so the next time we want to refresh
benchmark numbers we can run a single command instead of recreating
the harness ad-hoc.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from latos.ingestion.labeling.pipeline import cluster_project
from latos.ingestion.orchestrator import Orchestrator
from latos.ingestion.registry import default_registry


def main() -> None:
    src = Path("D:/Materials-Informatics/data_raw/dhivya_data")
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td) / "Dhivya"
        shutil.copytree(
            src,
            proj,
            ignore=shutil.ignore_patterns(".latos", "__pycache__", ".DS_Store"),
        )

        orch = Orchestrator(registry=default_registry())
        t0 = time.perf_counter()
        result = orch.ingest(proj, project_name="Dhivya")
        ingest_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        clusters = cluster_project(result.project)
        cluster_time = time.perf_counter() - t0

        n_samples = len(result.project.samples)
        n_clusters = len(clusters)
        n_files = sum(len(c.file_paths) for c in clusters)
        print(f"Stage 1 samples: {n_samples}")
        print(f"Stage 2 clusters: {n_clusters}")
        print(f"Files in clusters: {n_files}")
        print(f"Reduction: {n_samples - n_clusters} merges")
        print(f"Ingest: {ingest_time:.2f}s")
        print(f"Cluster: {cluster_time * 1000:.1f}ms")
        print()
        print("Stage 1 samples (folder-name heuristic):")
        for s in sorted(result.project.samples, key=lambda s: s.canonical_name):
            file_count = sum(len(m.files) for m in s.measurements)
            print(f"  - {s.canonical_name!r} ({file_count} files)")
        print()
        print("Stage 2 clusters (post-pipeline):")
        for c in clusters:
            aliases = list(c.aliases)
            print(f"  - {c.canonical!r}: aliases={aliases}, files={len(c.file_paths)}")


if __name__ == "__main__":
    main()
