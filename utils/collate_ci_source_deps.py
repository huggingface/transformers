"""Collate per-model deps.json artifacts into a single fetcher_source_deps.json.

Each model CI job writes deps.json (list of src/transformers/ files imported during
the test session) into its report directory. This script merges them into a single
mapping: src_file -> [test_files], suitable for consumption by tests_fetcher.py.

Usage:
    python utils/collate_ci_source_deps.py --artifacts-dir <dir> --output <path>

Arguments:
    --artifacts-dir  Directory containing downloaded artifact subdirectories, each
                     named like `aws-g5-4xlarge_run_models_gpu_models_clip_test_reports/`
                     and containing a deps.json file.
    --output         Path to write the merged JSON (e.g. utils/fetcher_source_deps.json).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Artifact dir names look like: aws-g5-4xlarge_run_models_gpu_models_clip_test_reports
# We extract the model folder part after `_run_models_gpu_` (e.g. `models_clip`).
_REPORT_DIR_RE = re.compile(r"_run_models_gpu_(.+)_test_reports$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    src_to_tests: dict[str, set[str]] = defaultdict(set)

    for report_dir in sorted(args.artifacts_dir.iterdir()):
        if not report_dir.is_dir():
            continue
        m = _REPORT_DIR_RE.search(report_dir.name)
        if not m:
            continue
        deps_file = report_dir / "deps.json"
        if not deps_file.is_file():
            continue

        # e.g. "models_clip" -> "tests/models/clip"
        model_folder = m.group(1).replace("_", "/", 1)  # models_clip -> models/clip
        test_dir = repo_root / "tests" / model_folder
        test_files = (
            sorted(f"tests/{model_folder}/{p.name}" for p in test_dir.glob("*.py") if p.name != "__init__.py")
            if test_dir.is_dir()
            else []
        )

        src_files: list[str] = json.loads(deps_file.read_text(encoding="utf-8"))
        for src_file in src_files:
            src_to_tests[src_file].update(test_files)

    result = {k: sorted(v) for k, v in sorted(src_to_tests.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Written {len(result)} src->test edges to {args.output}")


if __name__ == "__main__":
    main()
