"""Collate per-model deps_by_file.json artifacts into a single fetcher_source_deps.json.

Each model CI job writes deps_by_file.json into its report directory.  Format::

    {
        "tests/models/clip/test_processing_clip.py": {
            "collection":       ["src/transformers/...", ...],
            "execution":        ["src/transformers/...", ...],
            "graph_transitive": ["src/transformers/...", ...]
        },
        ...
    }

This script merges across all artifact directories, preserving the per-phase split
(collection / execution / graph_transitive), and writes::

    {
        "tests/models/clip/test_processing_clip.py": {
            "collection":       ["src/transformers/...", ...],
            "execution":        ["src/transformers/...", ...],
            "graph_transitive": ["src/transformers/...", ...]
        },
        ...
    }

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

    # Accumulate per-file, per-phase sets across all artifact dirs.
    test_to_phases: dict[str, dict[str, set[str]]] = defaultdict(dict)

    for report_dir in sorted(args.artifacts_dir.iterdir()):
        if not report_dir.is_dir():
            continue
        deps_file = report_dir / "deps_by_file.json"
        if not deps_file.is_file():
            continue

        # deps_by_file.json: {test_file: {"collection": [...], "execution": [...], "graph_transitive": [...]}}
        by_file: dict[str, dict[str, list[str]]] = json.loads(deps_file.read_text(encoding="utf-8"))
        for test_file, phases in by_file.items():
            for phase, src_files in phases.items():
                test_to_phases[test_file].setdefault(phase, set()).update(src_files)

    result = {
        test_file: {phase: sorted(srcs) for phase, srcs in sorted(phases.items())}
        for test_file, phases in sorted(test_to_phases.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Written {len(result)} test->src edges to {args.output}")


if __name__ == "__main__":
    main()
