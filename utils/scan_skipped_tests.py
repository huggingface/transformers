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


def analyze_test_override(test_fn: str, file_content: str) -> tuple[str, str] | None:
    """
    Analyzes if a test is overridden and if it's skipped by checking for decorators.

    Returns a tuple of (status, reason), e.g., ("SKIPPED", "Test is too slow"),
    or None if the test is not explicitly defined in the file content.
    """
    # Pattern to find a function definition and capture any preceding decorator lines
    pattern = re.compile(
        r"((?:^\s*@.*?\n)*?)^\s*def\s+" + re.escape(test_fn) + r"\b",
        re.MULTILINE,
    )
    match = pattern.search(file_content)

    if not match:
        return None  # test is not mentioned, so it's inherited (implicitly run).

    decorators = match.group(1) or ""
    if "skip" in decorators:
        reason_match = re.search(r'reason\s*=\s*["\'](.*?)["\']', decorators)
        if reason_match:
            reason = reason_match.group(1)
        else:
            # fallback for formats like @skip("reason") or @skipIf(..., "reason")
            reason_match = re.search(r'\((?:.*?,\s*)?["\'](.*?)["\']\)', decorators)
            reason = reason_match.group(1) if reason_match else decorators.strip().split("\n")[-1]

        return "SKIPPED", reason

    return "RAN", ""


def main():
    """Scans tests, analyzes overrides, and generates a JSON report."""
    try:
        common_tests = get_common_tests(COMMON_TESTS_FILE)
        all_models, model_files = get_models_and_test_files(MODELS_DIR)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    final_results = {}
    print(f"🔬 Scanning {len(common_tests)} common tests across {len(all_models)} models...")

    for i, test_fn in enumerate(common_tests):
        print(f"  ({i + 1}/{len(common_tests)}) Processing: {test_fn}", end="\r")

        models_that_ran = all_models.copy()
        models_that_skipped = []
        reasons_for_skipping = []

        for file_path in model_files:
            model_name = file_path.parent.name
            content = file_path.read_text(encoding="utf-8")
            result = analyze_test_override(test_fn, content)

            if result and result[0] == "SKIPPED":
                status, reason_text = result
                if model_name in models_that_ran:
                    models_that_ran.remove(model_name)
                models_that_skipped.append(model_name)
                reasons_for_skipping.append(f"{model_name}: {reason_text}")

        # compute statistics for the current test
        total_models = len(all_models)
        skipped_proportion = len(models_that_skipped) / total_models if total_models > 0 else 0.0

        final_results[test_fn] = {
            "models_ran": sorted(models_that_ran),
            "models_skipped": sorted(models_that_skipped),
            "skipped_proportion": round(skipped_proportion, 4),
            "reasons_skipped": sorted(reasons_for_skipping),
        }

    print("\n✅ Scan complete.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"📄 Report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
