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

import json
import re
from pathlib import Path


# Assumes the script is run from the root of the transformers repository.
REPO_ROOT = Path().cwd()

COMMON_TESTS_FILE = REPO_ROOT / "tests/test_modeling_common.py"
MODELS_DIR = REPO_ROOT / "tests/models"
OUTPUT_FILE = REPO_ROOT / "all_tests_scan_result.json"


def get_common_tests(file_path: Path) -> list[str]:
    """Extracts all test function names (e.g., 'test_forward') from the common test file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Common tests file not found at {file_path}")
    content = file_path.read_text(encoding="utf-8")
    # find all function definitions starting with 'test_'
    test_names = re.findall(r"^\s*def\s+(test_[a-zA-Z0-9_]+)", content, re.MULTILINE)
    return sorted(set(test_names))


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
    # fallback for formats like @skip("reason") or @skipIf(..., "reason")
    reason_match = re.search(r'\((?:.*?,\s*)?["\'](.*?)["\']\)', decorators)
    if reason_match:
        return reason_match.group(1)
    # if nothing matched, take the last decorator line
    return decorators.strip().split("\n")[-1].strip()


def extract_test_info(file_content: str) -> dict[str, tuple[str, str]]:
    """
    Parses a test file once and returns a mapping of test functions to their
    status and skip reason, e.g. {'test_forward': ('SKIPPED', 'too slow')}.
    """
    result: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r"((?:^\s*@.*?\n)*?)^\s*def\s+(test_[a-zA-Z0-9_]+)\b",
        re.MULTILINE,
    )

    for decorators, test_fn in pattern.findall(file_content):
        status = "RAN"
        reason = ""
        if "skip" in decorators:
            status = "SKIPPED"
            reason = _extract_reason_from_decorators(decorators)
        result[test_fn] = (status, reason)
    return result


def main():
    """Scans tests, analyzes overrides in a single pass per file, and generates a JSON report."""
    try:
        common_tests = get_common_tests(COMMON_TESTS_FILE)
        all_models, model_files = get_models_and_test_files(MODELS_DIR)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # cache overrides per model (merged across all its files)
    model_overrides: dict[str, dict[str, tuple[str, str]]] = {m: {} for m in all_models}

    print(f"🔬 Parsing {len(model_files)} model test files once each...")
    for file_path in model_files:
        model_name = file_path.parent.name
        content = file_path.read_text(encoding="utf-8")
        overrides = extract_test_info(content)
        # merge with existing overrides for the model
        model_overrides[model_name].update(overrides)

    final_results: dict[str, dict[str, object]] = {}
    total_models = len(all_models)

    print(f"📝 Aggregating results for {len(common_tests)} common tests...")
    for i, test_fn in enumerate(common_tests, start=1):
        print(f"  ({i}/{len(common_tests)}) {test_fn}", end="\r")

        models_ran: list[str] = []
        models_skipped: list[str] = []
        reasons_for_skipping: list[str] = []

        for model_name in all_models:
            overrides = model_overrides.get(model_name, {})
            if test_fn in overrides:
                status, reason_text = overrides[test_fn]
                if status == "SKIPPED":
                    models_skipped.append(model_name)
                    reasons_for_skipping.append(f"{model_name}: {reason_text}")
                else:  # explicitly overridden and executed
                    models_ran.append(model_name)
            else:
                # not overridden => inherited from common tests and executed
                models_ran.append(model_name)

        skipped_proportion = len(models_skipped) / total_models if total_models else 0.0

        final_results[test_fn] = {
            "models_ran": sorted(models_ran),
            "models_skipped": sorted(models_skipped),
            "skipped_proportion": round(skipped_proportion, 4),
            "reasons_skipped": sorted(reasons_for_skipping),
        }

    print("\n✅ Scan complete.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"📄 Report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
