"""Collate per-shard deps_by_file.json artifacts into one JSON per CI job.

Each CI shard writes a ``deps_by_file.json`` into its report directory.  Format::

    {
        "tests/models/clip/test_processing_clip.py": {
            "collection":       ["src/transformers/...", ...],
            "execution":        ["src/transformers/...", ...],
            "graph_transitive": ["src/transformers/...", ...]
        },
        ...
    }

This script discovers all ``deps_by_file.json`` files under ``--artifacts-dir``,
groups them by CI job (the top-level subdirectory), merges shards within each job,
and writes one output JSON per job.

Expected artifact layout (job / shard / deps_by_file.json)::

    artifacts/
      tests_torch/
        shard0/deps_by_file.json
        shard1/deps_by_file.json
      tests_generation/
        shard0/deps_by_file.json
      pipelines_torch/
        shard0/deps_by_file.json

Output::

    <output-dir>/tests_torch_runtime_deps.json
    <output-dir>/tests_generation_runtime_deps.json
    <output-dir>/pipelines_torch_runtime_deps.json

Usage:
    python utils/collate_ci_source_deps.py --artifacts-dir <dir> --output-dir <dir>

Arguments:
    --artifacts-dir  Root directory of downloaded PR CI artifacts.
    --output-dir     Directory where per-job JSON files are written.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge_deps_files(deps_files: list[Path]) -> dict[str, dict[str, list[str]]]:
    """Merge a list of deps_by_file.json files into one dict."""
    test_to_phases: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for deps_file in deps_files:
        by_file: dict[str, dict[str, list[str]]] = json.loads(deps_file.read_text(encoding="utf-8"))
        for test_file, phases in by_file.items():
            for phase, src_files in phases.items():
                test_to_phases[test_file].setdefault(phase, set()).update(src_files)
    return {
        test_file: {phase: sorted(srcs) for phase, srcs in sorted(phases.items())}
        for test_file, phases in sorted(test_to_phases.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    # Group by top-level subdirectory (job name).
    # Layout: <artifacts-dir>/<job>/<shard>/deps_by_file.json
    jobs: dict[str, list[Path]] = defaultdict(list)
    for deps_file in sorted(args.artifacts_dir.rglob("deps_by_file.json")):
        job_name = deps_file.relative_to(args.artifacts_dir).parts[0]
        jobs[job_name].append(deps_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for job_name, deps_files in sorted(jobs.items()):
        result = merge_deps_files(deps_files)
        out_path = args.output_dir / f"{job_name}_runtime_deps.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{job_name}: {len(deps_files)} shard(s), {len(result)} test files -> {out_path}")


if __name__ == "__main__":
    main()
