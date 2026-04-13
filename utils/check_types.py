"""Run ty type checking on specified directories.

Usage:
    python utils/check_types.py src/transformers/utils src/transformers/generation
"""

import glob
import subprocess
import sys


CHECKER_CONFIG = {
    "name": "types",
    "label": "Type annotations",
    # Approximate: ty follows imports beyond the listed directories; these globs cover
    # the explicitly targeted paths but not transitive dependencies.
    "file_globs": [
        "src/transformers/*.py",
        "src/transformers/utils/**/*.py",
        "src/transformers/generation/**/*.py",
        "src/transformers/quantizers/**/*.py",
        ".circleci/create_circleci_config.py",
    ],
    "check_args": [
        "src/transformers/*.py",
        "src/transformers/utils",
        "src/transformers/generation",
        "src/transformers/quantizers",
        ".circleci/create_circleci_config.py",
    ],
    "fix_args": None,
}


def _expand_args(args):
    """Expand glob patterns in *args* so that ty receives concrete paths."""
    expanded = []
    for arg in args:
        if "*" in arg:
            expanded.extend(sorted(glob.glob(arg)))
        else:
            expanded.append(arg)
    return expanded


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [<directory> ...]")
        sys.exit(1)

    directories = _expand_args(sys.argv[1:])
    print(f"Running ty check on: {', '.join(directories)}")
    result = subprocess.run(
        ["ty", "check", "--respect-ignore-files", "--exclude", "**/*_pb*", *directories],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
