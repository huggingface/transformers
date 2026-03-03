"""Run ty type checking on specified directories.

Usage:
    python utils/check_types.py src/transformers/utils src/transformers/generation
"""

import fnmatch
import subprocess
import sys
from pathlib import Path


def collect_py_files(directories: list[str]) -> list[Path]:
    """Recursively collect .py files from directories, excluding *_pb*.py."""
    files = []
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            print(f"Warning: {directory} is not a directory, skipping")
            continue
        for path in sorted(root.rglob("*.py")):
            if fnmatch.fnmatch(path.name, "*_pb*.py"):
                continue
            files.append(path)
    return files


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [<directory> ...]")
        sys.exit(1)

    directories = sys.argv[1:]
    files = collect_py_files(directories)

    if not files:
        print("No Python files found to check.")
        sys.exit(0)

    print(f"Running ty check on {len(files)} files from: {', '.join(directories)}")
    result = subprocess.run(
        ["ty", "check", "--force-exclude", *[str(f) for f in files]],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
