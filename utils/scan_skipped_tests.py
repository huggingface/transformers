# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
import json
import re
from pathlib import Path


# Assumes the script is run from the root of the transformers repository.
REPO_ROOT = Path().cwd()

COMMON_TESTS_FILE = REPO_ROOT / "tests/test_modeling_common.py"
MODELS_DIR = REPO_ROOT / "tests/models"


def get_common_tests(file_path: Path) -> list[str]:
    """Extract all common test function names (e.g., 'test_forward')."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Common tests file not found at {file_path}")
    content = file_path.read_text(encoding="utf-8")
    # find all function definitions starting with 'test_'
    return sorted(
        set(re.findall(r"^\s*def\s+(test_[a-zA-Z0-9_]+)", content, re.MULTILINE))
    )


def get_models_and_test_files(models_dir: Path) -> tuple[list[str], list[Path]]:
    """Discovers all models and their corresponding test files."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found at {models_dir}")

    test_files = sorted(models_dir.rglob("test_modeling_*.py"))
    model_names = sorted({f.parent.name for f in test_files})
    return model_names, test_files


def _extract_reason_from_decorators(decorators: str) -> str:
    """Extracts the reason string from a decorator block, if any."""
    reason_match = re.search(r'reason\s*=\s*["\'](.*?)["\']', decorators)
    if reason_match:
        return reason_match.group(1)

    reason_match = re.search(r'\((?:.*?,\s*)?["\'](.*?)["\']\)', decorators)
    if reason_match:
        return reason_match.group(1)

    return decorators.strip().split("\n")[-1].strip()


def extract_test_info(file_content: str) -> dict[str, tuple[str, str]]:
    """
    Parse a test file once and return a mapping of test functions to their
    status and skip reason, e.g. {'test_forward': ('SKIPPED', 'too slow')}.
    """
    result: dict[str, tuple[str, str]] = {}
    pattern = re.compile(r"((?:^\s*@.*?\n)*?)^\s*def\s+(test_[a-zA-Z0-9_]+)\b", re.MULTILINE)

    for decorators, test_fn in pattern.findall(file_content):
        status, reason = "RAN", ""
        if "skip" in decorators:
            status = "SKIPPED"
            reason = _extract_reason_from_decorators(decorators)
        result[test_fn] = (status, reason)
    return result


def build_overrides(model_files: list[Path]) -> dict[str, dict[str, tuple[str, str]]]:
    """Cache overrides per model (merged across all its files)."""
    overrides: dict[str, dict[str, tuple[str, str]]] = {}
    for file_path in model_files:
        model_name = file_path.parent.name
        content = file_path.read_text(encoding="utf-8")
        overrides.setdefault(model_name, {}).update(extract_test_info(content))
    return overrides


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def summarize_single_test(
    test_name: str,
    all_models: list[str],
    model_overrides: dict[str, dict[str, tuple[str, str]]],
) -> dict[str, object]:
    """Aggregate results for a single test and print a concise terminal summary."""
    models_ran, models_skipped, reasons_for_skipping = [], [], []

    for model in all_models:
        status, reason = model_overrides.get(model, {}).get(test_name, ("RAN", ""))
        if status == "SKIPPED":
            models_skipped.append(model)
            reasons_for_skipping.append(f"{model}: {reason}")
        else:
            models_ran.append(model)

    total_models = len(all_models)
    skipped_proportion = len(models_skipped) / total_models if total_models else 0.0

    print(f"\n== {test_name} ==")
    print(f"Ran on    : {len(models_ran)}/{total_models} models")
    print(f"Skipped on: {len(models_skipped)}/{total_models} models "
          f"({skipped_proportion:.1%})")
    if models_skipped:
        for reason in reasons_for_skipping[:10]:
            print(f" - {reason}")
        if len(reasons_for_skipping) > 10:
            print(" - ...")

    return {
        "models_ran": sorted(models_ran),
        "models_skipped": sorted(models_skipped),
        "skipped_proportion": round(skipped_proportion, 4),
        "reasons_skipped": sorted(reasons_for_skipping),
    }


def summarize_all_tests(
    common_tests: list[str],
    all_models: list[str],
    model_overrides: dict[str, dict[str, tuple[str, str]]],
) -> dict[str, dict[str, object]]:
    """Aggregate results for all common tests (original behaviour)."""
    results: dict[str, dict[str, object]] = {}
    total_models = len(all_models)

    print(f"📝 Aggregating results for {len(common_tests)} common tests...")
    for i, test_fn in enumerate(common_tests, start=1):
        print(f"  ({i}/{len(common_tests)}) {test_fn}", end="\r")

        models_ran, models_skipped, reasons_for_skipping = [], [], []

        for model in all_models:
            status, reason = model_overrides.get(model, {}).get(test_fn, ("RAN", ""))
            if status == "SKIPPED":
                models_skipped.append(model)
                reasons_for_skipping.append(f"{model}: {reason}")
            else:
                models_ran.append(model)

        skipped_proportion = len(models_skipped) / total_models if total_models else 0.0

        results[test_fn] = {
            "models_ran": sorted(models_ran),
            "models_skipped": sorted(models_skipped),
            "skipped_proportion": round(skipped_proportion, 4),
            "reasons_skipped": sorted(reasons_for_skipping),
        }

    print("\n✅ Scan complete.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan model tests for overridden or skipped methods.")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory in which JSON results will be saved (default: %(default)s).",
    )
    parser.add_argument(
        "--test_method_name",
        help="Specific test method to scan. "
             "If provided, only that method is analysed and reported.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    test_method_name = args.test_method_name

    try:
        common_tests = (
            [test_method_name]
            if test_method_name
            else get_common_tests(COMMON_TESTS_FILE)
        )
        all_models, model_files = get_models_and_test_files(MODELS_DIR)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    print(f"🔬 Parsing {len(model_files)} model test files once each...")
    model_overrides = build_overrides(model_files)

    if test_method_name:
        # single-test mode
        result = summarize_single_test(test_method_name, all_models, model_overrides)
        json_path = output_dir / f"scan_{test_method_name}.json"
        save_json(result, json_path)
        print(f"\n📄 JSON saved to {json_path.resolve()}")
    else:
        # full scan mode
        results = summarize_all_tests(common_tests, all_models, model_overrides)
        json_path = output_dir / "all_tests_scan_result.json"
        save_json(results, json_path)
        print(f"\n📄 JSON saved to {json_path.resolve()}")


if __name__ == "__main__":
    main()
