"""Run pinned MIC fixtures against this checkout; see `make run-mic` in the testing guide."""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).rstrip("\n")


def changed_files(root, base):
    """Include branch commits, staged/unstaged changes, deletions and untracked files."""
    ancestor = git(root, "merge-base", base, "HEAD")
    tracked = git(root, "diff", "--name-only", "--no-renames", "-z", ancestor, "--")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return sorted(set(filter(None, tracked.split("\0") + untracked.split("\0"))))


def main():
    model = os.environ.get("MODEL", "")
    all_fixtures = os.environ.get("ALL", "")
    base = os.environ.get("BASE") or "origin/main"
    if all_fixtures not in ("", "0", "1") or (model and all_fixtures == "1"):
        print("Use MODEL=<model_directory> or ALL=1, not both.", file=sys.stderr)
        return 2
    if model:
        if Path(model).name != model or not (ROOT / "src/transformers/models" / model).is_dir():
            print(f"Unknown model directory: {model!r}. Use a name from src/transformers/models/.", file=sys.stderr)
            return 2
        # Reuse MIC's fixture-derived dependency mapping, including composite models.
        paths = [f"src/transformers/models/{model}/__init__.py"]
    elif all_fixtures == "1":
        paths = []
    else:
        try:
            paths = changed_files(ROOT, base)
        except subprocess.CalledProcessError:
            print(f"Cannot compare against {base!r}. Set BASE=<git-ref>, MODEL=<name>, or ALL=1.", file=sys.stderr)
            return 2
        if not paths:
            print(f"No local changes relative to {base}. Use MODEL=<name> or ALL=1 to run fixtures explicitly.")
            return 0
    if importlib.util.find_spec("transformers_mic") is None:
        print(
            "Install MIC in your active Python environment:\n"
            '  python -m pip install "transformers-mic[runtime]" '
            '"hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git"',
            file=sys.stderr,
        )
        return 2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "transformers_mic.cli", "check", "--fleet", "control", "--fixtures-only"]
    print(f"Running MIC fixtures against {ROOT / 'src'} (revision from tests/contracts.lock).", flush=True)
    with tempfile.TemporaryDirectory(prefix="transformers-mic-") as directory:
        if all_fixtures == "1":
            command.append("--run-all")
        else:
            changed = Path(directory) / "changed-files.txt"
            changed.write_text("\n".join(paths) + "\n")
            command.extend(["--changed-files", str(changed)])
            print("Selecting contracts affected by:\n  " + "\n  ".join(paths), flush=True)
        return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
