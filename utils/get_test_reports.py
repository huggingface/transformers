import os
import argparse
import subprocess
from pathlib import Path

# Mapping from suite name to test directory under `tests/`
SUITE_TO_PATH = {
    "run_models_gpu": "models",
    "run_pipelines_torch_gpu": "pipelines",
    "run_examples_gpu": "examples",
    "run_torch_cuda_extensions_gpu": "utils/torch_cuda_extensions",
}

def is_valid_test_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith("__") and not path.name.startswith(".")

def run_pytest(suite: str, subdir: Path, root_test_dir: Path, machine_type: str, dry_run: bool):
    relative_path = subdir.relative_to(root_test_dir)
    report_name = f"{machine_type}_{suite}_{relative_path}_test_reports"

    cmd = [
        "python3", "-m", "pytest",
        "-rsfE", "-v",
        f"--make-reports={report_name}",
        str(subdir),
        "-m", "not not_device_test"
    ]

    print(f"Suite: {suite} | Running on: {relative_path}")
    print("Command:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=False)

def handle_suite(suite: str, test_root: Path, machine_type: str, dry_run: bool):
    if suite not in SUITE_TO_PATH:
        print(f"Unknown suite: {suite}")
        return

    subpath = SUITE_TO_PATH[suite]
    full_path = test_root / subpath

    if not full_path.exists():
        print(f"Test folder does not exist: {full_path}")
        return

    if full_path.is_file():
        # Top-level test file
        run_pytest(suite, full_path, test_root, machine_type, dry_run)
    else:
        # Recurse into each valid subdirectory
        for subdir in sorted(full_path.iterdir()):
            if is_valid_test_dir(subdir):
                run_pytest(suite, subdir, test_root, machine_type, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Run selected test suites recursively.")
    parser.add_argument("folder", help="Path to test root folder (e.g., ./tests)")
    parser.add_argument("--suites", nargs="+", required=True, help="List of test suite names to run")
    parser.add_argument("--machine-type", default="single-gpu", help="Machine type for report names")
    parser.add_argument("--enable-slow", action="store_true", help="Run slow tests instead of skipping them")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without running them")
    args = parser.parse_args()

    if args.enable_slow:
        os.environ["RUN_SLOW"] = "yes"

    test_root = Path(args.folder).resolve()
    if not test_root.exists():
        print(f"Root test folder not found: {test_root}")
        return

    for suite in args.suites:
        handle_suite(suite, test_root, args.machine_type, args.dry_run)

if __name__ == "__main__":
    main()

# python3 utils/get_test_reports.py ./tests   --suites run_models_gpu run_pipelines_torch_gpu  --machine-type multi-gpu