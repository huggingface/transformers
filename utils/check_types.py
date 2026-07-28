"""Run ty type checking on specified directories.

Usage:
    python utils/check_types.py src/transformers/utils src/transformers/generation
"""

import subprocess
import sys


CHECKER_CONFIG = {
    "name": "types",
    "label": "Type annotations",
    # For contributors:
    # - `check_args` below are the exact roots passed to `ty check`.
    # - `cache_globs` here are only used by `utils/checkers.py` to decide when a
    #   previously clean `types` run can be reused from cache.
    # ty follows imports *beyond* the checked roots, so the cache key must cover every source file
    # that could change a result -- not just the explicitly-checked paths. We hash the whole package
    # (plus the standalone .circleci target) so any source edit busts the cache and forces a
    # re-check. Otherwise a cached pass could silently hide a newly-introduced error in a
    # transitively-imported module that ty pulls in but that isn't one of the checked roots.
    "cache_globs": [
        "src/transformers/**/*.py",
        ".circleci/create_circleci_config.py",
    ],
    "check_args": [
        "src/transformers/_typing.py",
        "src/transformers/cli",
        "src/transformers/modeling_utils.py",
        "src/transformers/utils",
        "src/transformers/generation",
        "src/transformers/pipelines/__init__.py",
        "src/transformers/pipelines/feature_extraction.py",
        "src/transformers/pipelines/image_feature_extraction.py",
        "src/transformers/pipelines/video_classification.py",
        "src/transformers/quantizers",
        ".circleci/create_circleci_config.py",
        "src/transformers/dependency_versions_table.py",
        "src/transformers/dependency_versions_check.py",
        "src/transformers/conversion_mapping.py",
        "src/transformers/time_series_utils.py",
        "src/transformers/debug_utils.py",
        "src/transformers/hyperparameter_search.py",
        "src/transformers/pytorch_utils.py",
        "src/transformers/file_utils.py",
        "src/transformers/trainer_jit_checkpoint.py",
        "src/transformers/trainer_optimizer.py",
    ],
    "fix_args": None,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [<directory> ...]")
        sys.exit(1)

    directories = sys.argv[1:]
    print(f"Running ty check on: {', '.join(directories)}")
    # `--error-on-warning` makes ty exit non-zero on warning-level diagnostics (e.g.
    # possibly-missing-attribute), not just errors. Without it, warnings print but ty exits 0, so
    # `make typing` and CI both pass and the issue is never caught before commit.
    result = subprocess.run(
        ["ty", "check", "--respect-ignore-files", "--error-on-warning", "--exclude", "**/*_pb*", *directories],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
