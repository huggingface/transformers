#!/usr/bin/env python
"""Shim: delegates to utils.mlinter.mlinter for backward compatibility."""

import sys
from pathlib import Path


# Ensure the repo root is on sys.path so `utils.mlinter` is importable as a package.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Re-export subprocess so that `@patch("check_modeling_structure.subprocess.run")` still works in tests.
import subprocess  # noqa: E402, F401

# Re-export everything the test suite uses via `import check_modeling_structure as cms`.
from utils.mlinter._helpers import (  # noqa: E402, F401
    MODELS_ROOT,
    Violation,
    _collect_class_bases,
    _has_rule_suppression,
    _inherits_pretrained_model,
    _model_dir_name,
    full_name,
    is_self_method_call,
    is_super_method_call,
)
from utils.mlinter.mlinter import (  # noqa: E402, F401
    DEFAULT_ENABLED_TRF_RULES,
    TRF_MODEL_DIR_ALLOWLISTS,
    TRF_RULE_CHECKS,
    TRF_RULE_SPECS,
    TRF_RULES,
    _is_rule_allowlisted_for_file,
    analyze_file,
    colored_error_message,
    emit_violation,
    format_rule_details,
    format_rule_summary,
    format_violation,
    get_changed_modeling_files,
    iter_modeling_files,
    main,  # noqa: E402
    maybe_handle_rule_docs_cli,
    parse_args,
    resolve_enabled_rules,
    should_show_progress,
)


CHECKER_CONFIG = {
    "name": "modeling_structure",
    "label": "Modeling file structure",
    # mlinter scans modeling_*.py, modular_*.py, and configuration_*.py via MODELING_PATTERNS.
    "file_globs": [
        "src/transformers/models/**/modeling_*.py",
        "src/transformers/models/**/modular_*.py",
        "src/transformers/models/**/configuration_*.py",
    ],
    "check_args": [],
    "fix_args": None,
}


# Expose rule-id string constants (e.g. cms.TRF001 == "TRF001") for test compatibility.
for _rule_id in TRF_RULE_CHECKS:
    globals()[_rule_id] = _rule_id

if __name__ == "__main__":
    raise SystemExit(main())
