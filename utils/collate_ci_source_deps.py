"""Collate per-model deps_by_file.json artifacts into a single fetcher_source_deps.json.

Each model CI job writes deps_by_file.json into its report directory.  Format::

    {
        "tests/models/clip/test_processing_clip.py": {
            "collection": ["src/transformers/...", ...],
            "execution":  ["src/transformers/...", ...]
        },
        ...
    }

This script unions collection + execution phases per test file, merges across all
artifact directories, and writes a single inverted mapping::

    { "src/transformers/models/clip/image_processing_clip.py":
        ["tests/models/clip/test_processing_clip.py", ...], ... }

suitable for consumption by tests_fetcher.py.

Usage:
    python utils/collate_ci_source_deps.py --artifacts-dir <dir> --output <path>

Arguments:
    --artifacts-dir  Directory containing downloaded artifact subdirectories, each
                     named like `aws-g5-4xlarge_run_models_gpu_models_clip_test_reports/`
                     and containing a deps_by_file.json file.
    --output         Path to write the merged JSON (e.g. utils/fetcher_source_deps.json).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    test_to_srcs: dict[str, set[str]] = defaultdict(set)

    for report_dir in sorted(args.artifacts_dir.iterdir()):
        if not report_dir.is_dir():
            continue
        deps_file = report_dir / "deps_by_file.json"
        if not deps_file.is_file():
            continue

        # deps_by_file.json: {test_file: {"collection": [...], "execution": [...]}}
        by_file: dict[str, dict[str, list[str]]] = json.loads(deps_file.read_text(encoding="utf-8"))
        for test_file, phases in by_file.items():
            for src_files in phases.values():
                test_to_srcs[test_file].update(src_files)

    result = {k: sorted(v) for k, v in sorted(test_to_srcs.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Written {len(result)} test->src edges to {args.output}")


if __name__ == "__main__":
    main()
