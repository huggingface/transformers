# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from rich import print
from rich.console import Console

from ._helpers import MODELS_ROOT, Violation, _model_dir_name


try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 fallback


MODELING_PATTERNS = ("modeling_*.py", "modular_*.py", "configuration_*.py")
RULE_SPECS_PATH = Path(__file__).with_name("rules.toml")


def _load_rule_specs() -> dict[str, dict]:
    data = tomllib.loads(RULE_SPECS_PATH.read_text(encoding="utf-8"))
    rules = data.get("rules")
    if not isinstance(rules, dict):
        raise ValueError(f"Invalid rule spec file: missing [rules] table in {RULE_SPECS_PATH}")

    required_explanation_keys = {"what_it_does", "why_bad", "diff"}
    specs: dict[str, dict] = {}
    for rule_id, spec in rules.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid rule spec for {rule_id}: expected table")
        description = spec.get("description")
        default_enabled = spec.get("default_enabled")
        explanation = spec.get("explanation")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Invalid rule spec for {rule_id}: missing non-empty description")
        if not isinstance(default_enabled, bool):
            raise ValueError(f"Invalid rule spec for {rule_id}: default_enabled must be bool")
        if not isinstance(explanation, dict) or not required_explanation_keys.issubset(explanation):
            raise ValueError(f"Invalid rule spec for {rule_id}: incomplete explanation block")
        if any(not isinstance(explanation[key], str) for key in required_explanation_keys):
            raise ValueError(f"Invalid rule spec for {rule_id}: explanation values must be strings")

        allowlist_models = spec.get("allowlist_models", [])
        if not isinstance(allowlist_models, list) or any(not isinstance(item, str) for item in allowlist_models):
            raise ValueError(f"Invalid rule spec for {rule_id}: allowlist_models must be list[str]")

        specs[rule_id] = {
            "description": description,
            "default_enabled": default_enabled,
            "explanation": explanation,
            "allowlist_models": set(allowlist_models),
        }

    return specs


TRF_RULE_SPECS = _load_rule_specs()
TRF_RULES = {rule_id: spec["description"] for rule_id, spec in TRF_RULE_SPECS.items()}
DEFAULT_ENABLED_TRF_RULES = {rule_id for rule_id, spec in TRF_RULE_SPECS.items() if spec["default_enabled"]}
TRF_MODEL_DIR_ALLOWLISTS = {
    rule_id: spec["allowlist_models"] for rule_id, spec in TRF_RULE_SPECS.items() if spec["allowlist_models"]
}
CONSOLE = Console(stderr=True)
CACHE_PATH = Path(__file__).with_name(".mlinter_cache.json")


def _is_rule_allowlisted_for_file(rule_id: str, file_path: Path) -> bool:
    model_name = _model_dir_name(file_path)
    if model_name is None:
        return False
    return model_name in TRF_MODEL_DIR_ALLOWLISTS.get(rule_id, set())


def _content_hash(text: str, enabled_rules: set[str]) -> str:
    h = hashlib.sha256(text.encode("utf-8"))
    h.update(",".join(sorted(enabled_rules)).encode("utf-8"))
    return h.hexdigest()


def _load_cache() -> dict[str, str]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _validate_rule_ids(rule_ids: set[str]) -> set[str]:
    unknown = sorted(rule_id for rule_id in rule_ids if rule_id not in TRF_RULES)
    if unknown:
        raise ValueError(f"Unknown rule id(s): {', '.join(unknown)}. Valid rules: {', '.join(sorted(TRF_RULES))}")
    return rule_ids


def _rule_id_from_module_name(name: str) -> str | None:
    if len(name) != 6 or not name.startswith("trf") or not name[3:].isdigit():
        return None
    return name.upper()


def iter_modeling_files(paths: set[Path] | None = None):
    if paths is None:
        for pattern in MODELING_PATTERNS:
            yield from MODELS_ROOT.rglob(pattern)
        return

    for path in sorted(paths):
        if path.exists():
            yield path


def colored_error_message(file_path: str, line_number: int, message: str) -> str:
    return f"[bold red]{file_path}[/bold red]:[bold yellow]L{line_number}[/bold yellow]: {message}"


def _is_modeling_candidate(path: Path) -> bool:
    return (
        path.suffix == ".py"
        and path.name.startswith(("modeling_", "modular_", "configuration_"))
        and MODELS_ROOT in path.parents
    )


def _git_name_only(command: list[str]) -> list[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _git_diff(base_ref: str, triple_dot: bool) -> list[str]:
    diff_operator = "..." if triple_dot else ".."
    range_ref = f"{base_ref}{diff_operator}HEAD"
    return _git_name_only(["git", "diff", "--name-only", "--diff-filter=ACMR", range_ref])


def _git_worktree_changes() -> set[Path]:
    changed_paths = set(_git_name_only(["git", "diff", "--name-only", "--diff-filter=ACMR"]))
    changed_paths.update(_git_name_only(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]))
    changed_paths.update(_git_name_only(["git", "ls-files", "--others", "--exclude-standard"]))
    return {Path(path_str) for path_str in changed_paths}


def get_changed_modeling_files(base_ref: str) -> set[Path]:
    changed_paths = _git_diff(base_ref, triple_dot=True)
    if not changed_paths:
        changed_paths = _git_diff(base_ref, triple_dot=False)

    filtered_paths: set[Path] = set()
    for path in {Path(path_str) for path_str in changed_paths}.union(_git_worktree_changes()):
        if _is_modeling_candidate(path):
            filtered_paths.add(path)
    return filtered_paths


CheckFn = Callable[[ast.Module, Path, list[str]], list[Violation]]


def _build_rule_checks() -> dict[str, CheckFn]:
    """Auto-discover check() functions from trf*.py modules in this package."""
    checks: dict[str, CheckFn] = {}
    package_dir = Path(__file__).parent
    for module_path in sorted(package_dir.glob("trf*.py")):
        module_name = module_path.stem
        rule_id = _rule_id_from_module_name(module_name)
        if rule_id is None:
            continue
        if rule_id not in TRF_RULE_SPECS:
            raise ValueError(f"Missing rule spec for discovered module {module_name} ({rule_id}).")
        mod = importlib.import_module(f".{module_name}", package=__package__)
        check_fn = getattr(mod, "check", None)
        if not callable(check_fn):
            raise ValueError(f"Module {module_name} must define a check() function.")
        mod.RULE_ID = rule_id
        checks[rule_id] = check_fn

    missing_checks = sorted(set(TRF_RULE_SPECS) - set(checks))
    if missing_checks:
        raise ValueError(f"Missing check module(s) for rule id(s): {', '.join(missing_checks)}")
    return dict(sorted(checks.items()))


TRF_RULE_CHECKS = _build_rule_checks()

# Expose rule-id string constants (e.g. TRF001 == "TRF001") for test compatibility.
for _rule_id in TRF_RULE_CHECKS:
    globals()[_rule_id] = _rule_id


def analyze_file(file_path: Path, text: str, enabled_rules: set[str] | None = None) -> list[Violation]:
    if enabled_rules is None:
        enabled_rules = DEFAULT_ENABLED_TRF_RULES

    violations: list[Violation] = []
    source_lines = text.splitlines()
    tree = ast.parse(text, filename=str(file_path))

    for rule_id, check_fn in TRF_RULE_CHECKS.items():
        if rule_id in enabled_rules:
            for v in check_fn(tree, file_path, source_lines):
                violations.append(
                    Violation(
                        file_path=v.file_path,
                        line_number=v.line_number,
                        rule_id=rule_id,
                        message=v.message,
                    )
                )

    return [
        violation
        for violation in violations
        if not (
            violation.rule_id is not None and _is_rule_allowlisted_for_file(violation.rule_id, violation.file_path)
        )
    ]


def format_violation(violation: Violation) -> str:
    return colored_error_message(str(violation.file_path), violation.line_number, violation.message)


def emit_violation(violation: Violation, github_annotations: bool):
    if github_annotations:
        print(
            f"::error file={violation.file_path},line={violation.line_number}::{violation.message}",
            file=sys.stderr,
        )
        return

    print(format_violation(violation), file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only check changed modeling/modular files compared to --base-ref, plus local worktree changes.",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base git ref used with --changed-only (default: origin/main).",
    )
    parser.add_argument(
        "--github-annotations",
        action="store_true",
        help="Emit GitHub Actions annotation format output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable interactive progress animation.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the lint cache and re-check every file.",
    )
    parser.add_argument(
        "--enable-all-trf-rules",
        action="store_true",
        help="Enable all TRF rules (defaults already enable most).",
    )
    parser.add_argument(
        "--enable-rules",
        default="",
        help="Comma-separated TRF rule ids to enable in addition to defaults (e.g. TRF001,TRF002).",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List available TRF rules and exit.",
    )
    parser.add_argument(
        "--rule",
        default="",
        help="Show detailed docs for one rule id (e.g. TRF001) and exit.",
    )
    return parser.parse_args()


def should_show_progress(args: argparse.Namespace) -> bool:
    return (not args.no_progress) and (not args.github_annotations) and sys.stderr.isatty()


def resolve_enabled_rules(args: argparse.Namespace) -> set[str]:
    if args.enable_all_trf_rules:
        return _validate_rule_ids(set(TRF_RULES))

    enabled_rules = set(DEFAULT_ENABLED_TRF_RULES)
    if args.enable_rules.strip():
        enabled_rules.update(rule_id.strip() for rule_id in args.enable_rules.split(",") if rule_id.strip())
    return _validate_rule_ids(enabled_rules)


def format_rule_summary(rule_id: str) -> str:
    spec = TRF_RULE_SPECS[rule_id]
    default_label = "enabled" if spec["default_enabled"] else "disabled"
    return f"{rule_id}: {spec['description']} (default: {default_label})"


def format_rule_details(rule_id: str) -> str:
    spec = TRF_RULE_SPECS[rule_id]
    explanation = spec["explanation"]
    return "\n".join(
        [
            f"### {rule_id}",
            "",
            f"{explanation['what_it_does']} {explanation['why_bad']}",
            "",
            "```diff",
            explanation["diff"].strip(),
            "```",
        ]
    )


def maybe_handle_rule_docs_cli(args: argparse.Namespace) -> bool:
    if args.list_rules:
        for rule_id in sorted(TRF_RULE_SPECS):
            print(format_rule_summary(rule_id))
        return True

    if args.rule:
        rule_id = args.rule.strip().upper()
        _validate_rule_ids({rule_id})
        print(format_rule_details(rule_id))
        return True

    return False


def main() -> int:
    args = parse_args()
    if maybe_handle_rule_docs_cli(args):
        return 0

    violations: list[Violation] = []
    enabled_rules = resolve_enabled_rules(args)
    selected_paths = get_changed_modeling_files(args.base_ref) if args.changed_only else None

    modeling_files = list(iter_modeling_files(selected_paths))

    show_progress = should_show_progress(args)
    status_ctx = (
        CONSOLE.status(f"[bold blue]Checking modeling structure ({len(modeling_files)} files)...[/bold blue]")
        if show_progress
        else nullcontext()
    )

    use_cache = not args.no_cache
    cache = _load_cache() if use_cache else {}
    new_cache: dict[str, str] = {}
    skipped = 0

    with status_ctx:
        for file_path in modeling_files:
            try:
                text = file_path.read_text(encoding="utf-8")
                file_key = str(file_path)
                digest = _content_hash(text, enabled_rules)

                if use_cache and cache.get(file_key) == digest:
                    new_cache[file_key] = digest
                    skipped += 1
                    continue

                file_violations = analyze_file(file_path, text, enabled_rules=enabled_rules)
                violations.extend(file_violations)

                if not file_violations:
                    new_cache[file_key] = digest
            except Exception as exc:
                violations.append(Violation(file_path=file_path, line_number=1, message=f"failed to parse ({exc})."))

    if use_cache:
        _save_cache(new_cache)

    if len(violations) > 0:
        violations = sorted(violations, key=lambda v: (str(v.file_path), v.line_number, v.message))
        for violation in violations:
            emit_violation(violation, github_annotations=args.github_annotations)
        print(f"Found {len(violations)} modeling structure violation(s).", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
