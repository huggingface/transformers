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
groups them by CI job (the directory one level above the shard directory), merges
shards within each job, and writes one output JSON per job::

    <output-dir>/tests_torch.json
    <output-dir>/tests_generation.json
    <output-dir>/pipelines_torch.json
    ...

Expected artifact layout (two-level: job / shard)::

    artifacts/
      tests_torch/
        shard0/deps_by_file.json
        shard1/deps_by_file.json
      tests_generation/
        shard0/deps_by_file.json
      pipelines_torch/
        shard0/deps_by_file.json

For a flat layout (single job, shards directly under artifacts-dir)::

    artifacts/
      shard0/deps_by_file.json
      shard1/deps_by_file.json

all shards are merged into a single ``<output-dir>/all.json``.

Usage:
    python utils/collate_ci_source_deps.py --artifacts-dir <dir> --output-dir <dir>

Arguments:
    --artifacts-dir  Root directory of downloaded CI artifacts.
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

    # Group deps_by_file.json files by job name.
    # For a two-level layout (job/shard/deps_by_file.json) the job name is the
    # grandparent directory relative to artifacts-dir.
    # For a flat layout (shard/deps_by_file.json) there is no grandparent, so
    # all shards are grouped under the key "all".
    jobs: dict[str, list[Path]] = defaultdict(list)
    for deps_file in sorted(args.artifacts_dir.rglob("deps_by_file.json")):
        rel = deps_file.relative_to(args.artifacts_dir)
        # rel.parts: ("job", "shard", "deps_by_file.json") -> job name = parts[0]
        #            ("shard", "deps_by_file.json")         -> flat, use "all"
        job_name = rel.parts[0] if len(rel.parts) >= 3 else "all"
        jobs[job_name].append(deps_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for job_name, deps_files in sorted(jobs.items()):
        result = merge_deps_files(deps_files)
        out_path = args.output_dir / f"{job_name}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{job_name}: {len(deps_files)} shard(s), {len(result)} test files -> {out_path}")


if __name__ == "__main__":
    main()
