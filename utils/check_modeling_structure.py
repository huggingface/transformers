#!/usr/bin/env python
"""Thin local entrypoint for the external mlinter package."""

import mlinter


CHECKER_CONFIG = {
    "name": "modeling_structure",
    "label": "Modeling file structure",
    "file_globs": [
        "src/transformers/models/**/modeling_*.py",
        "src/transformers/models/**/modular_*.py",
        "src/transformers/models/**/configuration_*.py",
    ],
    "check_args": [],
    "fix_args": None,
}

if __name__ == "__main__":
    raise SystemExit(mlinter.main())
