"""Smart sample labeling — multi-source hint extraction + fuzzy clustering.

Stage 1's `_infer_sample_name` heuristic looked only at the file's parent
folder name (walking past generic technique labels). It's stable enough
to ship, but it falls over the moment a researcher names the same sample
two different ways across folders — the textbook Dhivya case where
`CS Pure` (XRD folder) and `CS (Pure)` (XPS folder) become two distinct
samples in the database when they should be one.

Stage 2 replaces that single-source heuristic with a pipeline:

1. **Hints** (`hints.py`)        — for each file, gather every plausible
   sample-name candidate from the path, filename, file metadata, and
   parsed content, with a per-source confidence.
2. **Normalize** (`normalize.py`, Stage 2B) — collapse trivial spelling
   variants (case, separators, Unicode) so identical-modulo-formatting
   names don't survive as different strings.
3. **Cluster** (`cluster.py`, Stage 2C) — build a similarity graph over
   normalized candidates with `rapidfuzz`, take connected components
   via `networkx`, and pick a canonical name per cluster.
4. **Confirm** (`ui/components/confirm_samples.py`, Stage 2D) — let the
   user override any decision via drag-merge / right-click split, with
   the result persisted to `samples.aliases` in the DB.

Nothing in this package depends on Qt. The cluster output is consumed by
the orchestrator (replacing `_infer_sample_name`) and by the UI's
confirmation page.
"""
