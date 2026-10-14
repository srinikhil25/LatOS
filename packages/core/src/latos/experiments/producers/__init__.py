"""Record producers: the technique-specific step in front of the generic layer.

A producer turns one pipeline's raw data into a `RecordTable`, one row per
specimen, under a named contract. Everything a technique knows lives here:
which file to read, what a repeated measurement is and how it is reduced, what
a sign means.

The modules one level up know none of that and never import from this package;
`tests/unit/experiments/test_import_direction.py` holds them to it.
"""
