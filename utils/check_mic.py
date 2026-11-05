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
import time
from concurrent.futures import ThreadPoolExecutor
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


# The fixture files MIC's selector reads (`select_contracts.code_dirs`); never the weights.
FIXTURE_CONFIGS = ["config.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"]


def use_local_source(root):
    """Resolve `transformers` (and the classes MIC's selector looks up) from this checkout."""
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))


def fleet_contracts(root, fleet_name):
    """The contracts of the fleet pinned in `tests/contracts.lock`."""
    from transformers_mic import fleet

    lock, _ = fleet.download(fleet_name, fleet.read_pin(root / "tests/contracts.lock"))
    return json.loads(Path(lock).read_text(encoding="utf-8"))["contracts"]


def fixture_configs(contracts):
    """{contract: [folders with only its fixtures' config files]}, fetched in parallel and cached by the Hub."""

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    from transformers_mic.hub import canonical_repo

    disable_progress_bars()
    jobs = [
        (name, repo, pin["revision"])
        for name, contract in contracts.items()
        for repo, pin in contract.get("fixtures", {}).items()
    ]

    def fetch(job):
        name, repo, revision = job
        return name, Path(snapshot_download(canonical_repo(repo), revision=revision, allow_patterns=FIXTURE_CONFIGS))

    folders = {name: [] for name in contracts}
    started = time.perf_counter()
    step = max(1, len(jobs) // 10)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for done, (name, folder) in enumerate(pool.map(fetch, jobs), 1):
            folders[name].append(folder)
            # Progress only when it is slow (first run); cached configs read in well under a second.
            if (done % step == 0 or done == len(jobs)) and time.perf_counter() - started > 1:
                print(f"Reading fixture configs: {done}/{len(jobs)}", flush=True)
    print(f"Read fixture configs of {len(contracts)} contracts in {time.perf_counter() - started:.1f}s", flush=True)
    return folders


def human_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def download_fixtures(contracts, names):
    """Fetch the selected contracts' Hub repos before the runner does, showing what is fetched.

    The runner downloads them silently; doing it here (into the same Hub cache) shows what is
    fetched and how big it is, and the runner then finds everything cached. Checking what is
    already cached costs one Hub call per repo, so those run in parallel.
    """
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    from huggingface_hub.utils import disable_progress_bars
    from transformers_mic.hub import canonical_repo

    disable_progress_bars()
    repos = {}
    for name in names:
        contract = contracts[name]
        pins = [contract["integration"], *({"repo": r, **pin} for r, pin in contract.get("fixtures", {}).items())]
        for pin in pins:
            repos.setdefault((canonical_repo(pin["repo"]), pin["revision"]), name)
    api = HfApi()

    def uncached(job):
        repo, revision = job
        files = api.model_info(repo, revision=revision, files_metadata=True).siblings or []
        return [f for f in files if not isinstance(try_to_load_from_cache(repo, f.rfilename, revision=revision), str)]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        missing = dict(zip(repos, pool.map(uncached, repos)))
    to_fetch = [job for job in repos if missing[job]]
    print(
        f"Fixtures: {len(repos) - len(to_fetch)}/{len(repos)} repos cached "
        f"(checked in {time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    for index, (repo, revision) in enumerate(to_fetch, 1):
        files = missing[repo, revision]
        # One line per repo, printed before the fetch: the latest line is the one in progress.
        print(
            f"[{index}/{len(to_fetch)}] {repos[repo, revision]}: {repo} downloading {len(files)} files, "
            f"{human_size(sum(f.size or 0 for f in files))}",
            flush=True,
        )
        snapshot_download(repo, revision=revision)


def model_contracts(root, model, fleet_name="prototypes"):
    """Fleet contracts with a fixture whose top-level `model_type` lives in `models/<model>`."""
    use_local_source(root)
    from transformers.models.auto.configuration_auto import model_type_to_module_name

    selected = []
    for name, folders in fixture_configs(fleet_contracts(root, fleet_name)).items():
        model_types = [json.loads((f / "config.json").read_text(encoding="utf-8"))["model_type"] for f in folders]
        if any(model_type_to_module_name(model_type) == model for model_type in model_types):
            selected.append(name)
    return selected


def changed_contracts(root, paths, fleet_name="prototypes"):
    """MIC's own selection for `paths`: ({contract: reason}, fleet-wide reasons, fleet contracts).

    Same rules as `tmic check --changed-files`, but only fetches the fixtures' config files, and
    not even those when a change (setup.py, the fleet pin, shared code) selects every contract:
    the fleet-wide reasons then name every such path and the selection is empty.
    """
    from transformers_mic import resources

    use_local_source(root)
    sys.path.insert(0, str(resources.backend()))
    import select_contracts

    from transformers.utils import logging

    # The selector resolves class names, some deprecated (`...Fast`); their warnings are noise here.
    logging.set_verbosity_error()
    contracts = fleet_contracts(root, fleet_name)
    nothing = {name: [] for name in contracts}
    fleet_wide = []
    for path in paths:
        selected = select_contracts.select([path], nothing)
        if contracts and set(selected) == set(contracts):
            fleet_wide.append(next(iter(selected.values())))
    if fleet_wide:
        return {}, fleet_wide, contracts
    dirs = {
        name: sorted({d for folder in folders for d in select_contracts.code_dirs(folder)})
        for name, folders in fixture_configs(contracts).items()
    }
    return select_contracts.select(paths, dirs), [], contracts


# A fixture whose baseline has no expectations for this platform yet (e.g. cpu-arm64 locally).
NO_BASELINE = "no reviewed baseline at "
SUMMARY_WORDS = {
    "pass": "passed",
    "unchecked": "ran without a baseline",
    "fail": "failed",
    "error": "errored",
    "xfail": "known failures",
    "unsupported": "unsupported",
    "recorded": "recorded",
}


def supports_allow_missing_baseline(env):
    """`tmic check --allow-missing-baseline` exists (transformers-mic > 0.1.1)."""
    command = [sys.executable, "-m", "transformers_mic.cli", "check", "--help"]
    help_text = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True).stdout
    return "--allow-missing-baseline" in help_text


def missing_baseline(record):
    return record["status"] == "fail" and any(f.startswith(NO_BASELINE) for f in record.get("failures", []))


def format_record(record, note=""):
    """One line per target, plus indented details for anything that is not a plain pass."""
    status = record["status"]
    name = record.get("contract_name", record["contract"])
    variant = "" if record.get("variant", "default") == "default" else f" [{record['variant']}]"
    seconds = f"  {record['seconds']:.1f}s" if "seconds" in record else ""
    label = f"{status} ({note})" if note else status
    lines = [f"{STATUS.get(status, '?')} {name}{variant}  {record['target']}  {label}{seconds}"]
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
    parts = [f"{counts[k]} {word}" for k, word in SUMMARY_WORDS.items() if counts.get(k)]
    return f"MIC: {', '.join(parts) or 'nothing ran'} in {elapsed:.1f}s"


def stream(command, env, fleet_name, on_record, announce=True, total=None):
    """Run the MIC runner, print its fleet/selection messages and hand each result record to `on_record`.

    Returns the runner's exit code and the last non-JSON lines it printed (tracebacks, warnings).
    """
    tail = []
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
        elif "fleet_revision" in message:
            if announce:
                print(
                    f"Fleet {fleet_name} ({message['fleet_repository']}@{message['fleet_revision'][:10]})", flush=True
                )
                # The runner reports a contract only once it has finished; say so instead of going quiet.
                what = f"{total} contract{'s' if total != 1 else ''}" if total else "the selected contracts"
                print(f"Running {what}, results print as each one finishes...", flush=True)
        elif "selection" in message:
            print("\n".join(format_selection(message["selection"])), flush=True)
        elif "status" in message and "target" in message:
            on_record(message)
    return process.wait(), tail


def without_selection(command):
    """`command` minus its contract selection (--contract/--changed-files/--run-all)."""
    kept, skip = [], False
    for arg in command:
        if skip:
            skip = False
        elif arg in ("--contract", "--changed-files"):
            skip = True
        elif arg != "--run-all":
            kept.append(arg)
    return kept


def run(command, env, fleet_name="prototypes", total=None):
    """Run the selected fixtures and print a readable digest. Returns the exit code for the checker.

    Fixtures without a reviewed baseline for this platform are run without comparison (in one pass
    with `--allow-missing-baseline` when MIC supports it, else re-run with `--no-baseline`): they
    are reported as "ran without a baseline" (the usage path executes, outputs are not compared)
    instead of failing the check. Any other failure fails it.
    """
    counts = {}
    missing = []
    started = time.perf_counter()

    def count(record, key=None):
        key = key or record["status"]
        counts[key] = counts.get(key, 0) + 1

    def first_pass(record):
        if record.get("baseline_missing"):
            # `--allow-missing-baseline`: outputs validated in this run, not compared.
            count(record, "unchecked")
            print("\n".join(format_record(record, f"no {record.get('platform', 'platform')} baseline")), flush=True)
        elif missing_baseline(record):
            missing.append(record)
        else:
            count(record)
            print("\n".join(format_record(record)), flush=True)

    returncode, tail = stream(command, env, fleet_name, first_pass, total=total)
    if missing:
        names = sorted({r.get("contract_name", r["contract"]) for r in missing})
        platform = missing[0].get("platform", "this platform")
        print(f"No reviewed {platform} baseline for {', '.join(names)}; running without comparison...", flush=True)
        note = f"no {platform} baseline"

        def second_pass(record):
            count(record, "unchecked" if record["status"] == "pass" else None)
            print("\n".join(format_record(record, note)), flush=True)

        retry = without_selection(command) + [arg for name in names for arg in ("--contract", name)]
        retry_code, retry_tail = stream([*retry, "--no-baseline"], env, fleet_name, second_pass, announce=False)
        # The first run's non-zero code came from the missing baselines, now accounted for.
        returncode, tail = retry_code, retry_tail
    failed = counts.get("fail", 0) + counts.get("error", 0)
    if returncode != 0 and not failed and tail:
        # The runner failed before (or without) reporting a failing target: show what it said.
        print("Runner output:\n" + "\n".join(f"  {line}" for line in tail), flush=True)
    print(format_summary(counts, time.perf_counter() - started), flush=True)
    return 1 if failed else returncode


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
    if supports_allow_missing_baseline(env):
        command.append("--allow-missing-baseline")
    if model:
        names = model_contracts(ROOT, model, fleet_name)
        if not names:
            print(f"No MIC contract has a models/{model} fixture in the pinned {fleet_name} fleet; nothing to run.")
            return 0
        print(f"Selected for models/{model}: {', '.join(names)}", flush=True)
        contracts = fleet_contracts(ROOT, fleet_name)
    elif all_fixtures == "1":
        contracts = fleet_contracts(ROOT, fleet_name)
        names = list(contracts)
        print(f"Selected: every contract in the {fleet_name} fleet ({len(names)}, ALL=1)", flush=True)
    else:
        print(f"Changed files vs {base}: {len(paths)}", flush=True)
        selection, fleet_wide, contracts = changed_contracts(ROOT, paths, fleet_name)
        if fleet_wide:
            print(
                f"These changes select every contract in the {fleet_name} fleet ({len(contracts)}):\n"
                + "\n".join(f"  {reason}" for reason in fleet_wide)
                + "\nThat is a long local run; nothing was run. Use MODEL=<name> for one model, "
                "or ALL=1 to run the whole fleet.",
                flush=True,
            )
            return 0
        print("\n".join(format_selection(selection)), flush=True)
        if not selection:
            return 0
        names = sorted(selection)
    download_fixtures(contracts, names)
    if all_fixtures == "1":
        command.append("--run-all")
    else:
        for name in names:
            command.extend(["--contract", name])
    return run(command, env, fleet_name, total=len(names))


if __name__ == "__main__":
    sys.exit(main())
