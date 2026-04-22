#!/usr/bin/env python
"""Thin local entrypoint for the external mlinter package."""

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


def _require_mlinter():
    try:
        import mlinter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "This script requires the standalone `transformers-mlinter` package. "
            'Install the repo quality dependencies with `pip install -e ".[quality]"` and retry.'
        ) from error

    return mlinter


if __name__ == "__main__":
    try:
        raise SystemExit(_require_mlinter().main())
    except ModuleNotFoundError as error:
        raise SystemExit(str(error)) from error
