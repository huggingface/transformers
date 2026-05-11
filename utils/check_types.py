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
    # Approximate: ty follows imports beyond the listed directories; these globs cover
    # the explicitly targeted paths but not transitive dependencies.
    "cache_globs": [
        "src/transformers/_typing.py",
        "src/transformers/cli/**/*.py",
        "src/transformers/modeling_utils.py",
        "src/transformers/utils/**/*.py",
        "src/transformers/generation/**/*.py",
        "src/transformers/pipelines/__init__.py",
        "src/transformers/pipelines/feature_extraction.py",
        "src/transformers/pipelines/image_feature_extraction.py",
        "src/transformers/pipelines/video_classification.py",
        "src/transformers/quantizers/**/*.py",
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
    result = subprocess.run(
        ["ty", "check", "--respect-ignore-files", "--exclude", "**/*_pb*", *directories],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
