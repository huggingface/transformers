"""Run pinned Model Integration Contract (MIC) fixtures against this checkout.

Usage (see "Run Model Integration Contracts locally" in the testing guide):

    make run-mic                # contracts affected by local changes vs BASE (default origin/main)
    make run-mic MODEL=bert     # only the contracts whose fixtures are a models/bert model
    make run-mic ALL=1          # every contract in the fleet
    make run-mic FLEET=control  # the control fleet instead of the default prototypes fleet
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CHECKER_CONFIG = {
    "name": "mic",
    "label": "Model integration contracts",
    # Never cached: the result depends on the network (pinned Hub fixtures), git state and the
    # installed runtime, none of which a file glob can capture. Not part of any CI checker list;
    # run it explicitly with `make run-mic` or `python utils/checkers.py mic`.
    "cache_globs": None,
    "check_args": [],
    "fix_args": None,
}

ROOT = Path(__file__).resolve().parents[1]
FLEETS = ("prototypes", "control")
INSTALL_HINT = (
    "Install MIC in your active Python environment:\n"
    '  python -m pip install "transformers-mic[runtime]" '
    '"hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git"'
)
STATUS = {
    "pass": "✓",
    "fail": "✗",
    "error": "✗",
    "xfail": "~",
    "unsupported": "-",
    "recorded": "+",
}
# Raw runner lines kept for context when a run fails without explaining itself.
TAIL_LINES = 30


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).rstrip("\n")


def changed_files(root, base):
    """Include branch commits, staged/unstaged changes, deletions and untracked files."""
    ancestor = git(root, "merge-base", base, "HEAD")
    tracked = git(root, "diff", "--name-only", "--no-renames", "-z", ancestor, "--")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return sorted(set(filter(None, tracked.split("\0") + untracked.split("\0"))))


def model_contracts(root, model, fleet_name="prototypes"):
    """Fleet contracts with a fixture whose top-level `model_type` lives in `models/<model>`."""
    sys.path.insert(0, str(root / "src"))
    from huggingface_hub import hf_hub_download
    from transformers_mic import fleet
    from transformers_mic.hub import canonical_repo

    from transformers.models.auto.configuration_auto import model_type_to_module_name

    lock, _ = fleet.download(fleet_name, fleet.read_pin(root / "tests/contracts.lock"))
    selected = []
    for name, contract in json.loads(Path(lock).read_text(encoding="utf-8"))["contracts"].items():
        for repo, pin in contract.get("fixtures", {}).items():
            config = Path(hf_hub_download(canonical_repo(repo), "config.json", revision=pin["revision"]))
            if model_type_to_module_name(json.loads(config.read_text(encoding="utf-8"))["model_type"]) == model:
                selected.append(name)
                break
    return selected


def format_record(record):
    """One line per target, plus indented details for anything that is not a plain pass."""
    status = record["status"]
    name = record.get("contract_name", record["contract"])
    variant = "" if record.get("variant", "default") == "default" else f" [{record['variant']}]"
    seconds = f"  {record['seconds']:.1f}s" if "seconds" in record else ""
    lines = [f"{STATUS.get(status, '?')} {name}{variant}  {record['target']}  {status}{seconds}"]
    if status == "xfail":
        lines.append(f"    known failure: {record.get('known_failure', '?')}")
    lines += [f"    {failure}" for failure in record.get("failures", [])]
    if status in ("fail", "error") and record.get("environment_differs"):
        differs = ", ".join(f"{k} {v}" for k, v in record["environment_differs"].items())
        lines.append(f"    environment differs from the baseline: {differs}")
    return lines


def format_selection(selection):
    if not selection:
        return ["No contract matched the changed files; nothing to run."]
    return [f"Selected {name}: {reason}" for name, reason in sorted(selection.items())]


def format_summary(counts, elapsed):
    words = {
        "pass": "passed",
        "fail": "failed",
        "error": "errored",
        "xfail": "known failures",
        "unsupported": "unsupported",
        "recorded": "recorded",
    }
    parts = [f"{counts[k]} {word}" for k, word in words.items() if counts.get(k)]
    return f"MIC: {', '.join(parts) or 'nothing ran'} in {elapsed:.1f}s"


def run(command, env, fleet_name="prototypes"):
    """Run the MIC runner and print a readable digest of its JSON stream. Returns its exit code."""
    counts = {}
    tail = []
    started = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        try:
            message = json.loads(line)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            if line.strip():
                tail = [*tail, line.rstrip()][-TAIL_LINES:]
            continue
        if "fleet_revision" in message:
            print(f"Fleet {fleet_name} ({message['fleet_repository']}@{message['fleet_revision'][:10]})", flush=True)
        elif "selection" in message:
            print("\n".join(format_selection(message["selection"])), flush=True)
        elif "status" in message and "target" in message:
            counts[message["status"]] = counts.get(message["status"], 0) + 1
            print("\n".join(format_record(message)), flush=True)
    returncode = process.wait()
    if returncode != 0 and not (counts.get("fail") or counts.get("error")) and tail:
        # The runner failed before (or without) reporting a failing target: show what it said.
        print("Runner output:\n" + "\n".join(f"  {line}" for line in tail), flush=True)
    print(format_summary(counts, time.perf_counter() - started), flush=True)
    return returncode


def main():
    model = os.environ.get("MODEL", "")
    all_fixtures = os.environ.get("ALL", "")
    base = os.environ.get("BASE") or "origin/main"
    fleet_name = os.environ.get("FLEET") or FLEETS[0]
    if fleet_name not in FLEETS:
        print(f"Unknown fleet: {fleet_name!r}. Use FLEET={' or FLEET='.join(FLEETS)}.", file=sys.stderr)
        return 2
    if all_fixtures not in ("", "0", "1") or (model and all_fixtures == "1"):
        print("Use MODEL=<model_directory> or ALL=1, not both.", file=sys.stderr)
        return 2
    if model:
        if Path(model).name != model or not (ROOT / "src/transformers/models" / model).is_dir():
            print(f"Unknown model directory: {model!r}. Use a name from src/transformers/models/.", file=sys.stderr)
            return 2
    elif all_fixtures != "1":
        try:
            paths = changed_files(ROOT, base)
        except subprocess.CalledProcessError:
            print(f"Cannot compare against {base!r}. Set BASE=<git-ref>, MODEL=<name>, or ALL=1.", file=sys.stderr)
            return 2
        if not paths:
            print(f"No local changes relative to {base}. Use MODEL=<name> or ALL=1 to run fixtures explicitly.")
            return 0
    if importlib.util.find_spec("transformers_mic") is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    command = [sys.executable, "-m", "transformers_mic.cli", "check", "--fleet", fleet_name, "--fixtures-only"]
    with tempfile.TemporaryDirectory(prefix="transformers-mic-") as directory:
        if model:
            contracts = model_contracts(ROOT, model, fleet_name)
            if not contracts:
                print(
                    f"No MIC contract has a models/{model} fixture in the pinned {fleet_name} fleet; nothing to run."
                )
                return 0
            print(f"Selected for models/{model}: {', '.join(contracts)}", flush=True)
            for name in contracts:
                command.extend(["--contract", name])
        elif all_fixtures == "1":
            print("Selected: every contract (ALL=1)", flush=True)
            command.append("--run-all")
        else:
            changed = Path(directory) / "changed-files.txt"
            changed.write_text("\n".join(paths) + "\n", encoding="utf-8")
            command.extend(["--changed-files", str(changed)])
            print(f"Changed files vs {base}: {len(paths)}", flush=True)
        return run(command, env, fleet_name)


if __name__ == "__main__":
    sys.exit(main())
