#!/usr/bin/env python
"""Thin local entrypoint for the external mlinter package."""

import sys
from pathlib import Path


CHECKER_CONFIG = {
    "name": "modeling_structure",
    "label": "Modeling file structure",
    "file_globs": [
        "src/transformers/models/**/modeling_*.py",
        "src/transformers/models/**/modular_*.py",
        "src/transformers/models/**/configuration_*.py",
    ],
    "check_args": ["--rules-toml", "utils/rules.toml"],
    "fix_args": None,
}

RULES_TOML_PATH = Path(__file__).resolve().with_name("rules.toml")


def _require_mlinter():
    try:
        import mlinter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "This script requires the standalone `transformers-mlinter` package. "
            'Install the repo quality dependencies with `pip install -e ".[quality]"` and retry.'
        ) from error

    return mlinter


def _add_default_rules_toml(argv: list[str]) -> list[str]:
    if any(arg == "--rules-toml" or arg.startswith("--rules-toml=") for arg in argv[1:]):
        return argv

    return [argv[0], "--rules-toml", str(RULES_TOML_PATH), *argv[1:]]


if __name__ == "__main__":
    try:
        sys.argv = _add_default_rules_toml(sys.argv)
        raise SystemExit(_require_mlinter().main())
    except ModuleNotFoundError as error:
        raise SystemExit(str(error)) from error
