"""Run a framework's MICs from its fleet lock and compare with its own baselines.

    python example/run.py --framework transformers                  # every contract, target, default variant
    python example/run.py --framework transformers --contract T5 --variant eager
    python example/run.py --framework transformers --target proto-t5-small-slim
    python example/run.py --framework transformers --fixtures-only  # what a framework PR runs
    python example/run.py --framework transformers --isolate        # one process per target
    python example/run.py --framework transformers.js
    python example/run.py ... --record                              # candidate baselines for review
    python example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu
    python example/run.py ... --output results.jsonl                # one result record per line
    git diff --name-only A B | python example/run.py --framework transformers --fixtures-only --changed-files -
    python utils/contracts/run.py --framework transformers --lock tests/contracts/fleet-lock.json  # lock elsewhere

For each contract in <framework>/tests/contracts/fleet.lock (or --lock, with
expectations/ next to it) it reads the
manifest from the pinned integration repo, expands the targets (each slim
fixture once, then each checkpoint), runs the contract, and compares the result
with <framework>/tests/contracts/expectations/<contract>/<platform>/<target>[.<variant>].json.
The manifest, fleet lock, fixture builds, baselines, tolerances, and result
records are validated against the JSON schemas in schemas/.

Runnable-markdown contracts are collected with doc-builder's own block
collector (doc-builder from main), so the runner executes exactly what the docs
build tests and renders. They run in this process, one after the other, so
imports are paid once for the whole run; --isolate runs each target in its own
process instead. Script contracts (Transformers.js) always run in their own process.

--changed-files runs only the contracts the listed Transformers files can
affect (select_contracts.py): a model directory selects the contracts whose
fixtures use it, other library code selects all of them, and files outside the
library select none. With --run-all every contract runs anyway and each record
carries selected_by (null when not selected), to measure what selection misses.

Contracts are downloaded at the Hub revision the fleet lock pins and checked
against its files_sha256; --local-contracts runs the folders in example/hub
instead (while editing a contract; results say "local" when they differ from
the pin). Script contracts (Transformers.js) run from the local folder, where
their packages are installed, which must match the pin.

Fixtures are downloaded at the Hub revision the fleet lock pins (or taken
from their local build folder when unpublished, or with --local-fixtures), and
checked against the file hashes in their build.json, which the lock also pins; a fixture that is missing or differs is an error, not a skip
(build it with build_fixtures.py). Comparison tolerances come from the
manifest, overridden per platform by expectations/<contract>/<platform>/tolerances.json,
since GPU and lower-precision runs drift more than CPU float32.

Every manifest output is required (training outputs only for a framework that
declares training), in the result and in the baseline: a missing output fails
rather than being skipped. Numeric outputs are compared by shape as well as
value. all_true outputs (the contract's named gradient checks: present, nonzero,
finite) must be a non-empty dict whose names match the baseline's.
Missing outputs, non-finite numeric outputs, and failed assertions (all_true,
true) fail on their own, and are never recorded;
--record also reruns each target in a fresh process and refuses outputs that
differ. A
framework can list a target under known_failures in its fleet lock, for a bug
in the framework under test. A plain reason string covers the whole target: a
failure is reported as xfail, and a pass as a failure, so the entry is removed
once the bug is fixed. An object {issue, match, skip_outputs} narrows it:
with match, the target is xfail only if every failure line contains that text
(any other failure still fails, and a pass asks to remove the entry); with
skip_outputs, those outputs are left out of the comparison and recorded as
skipped_outputs, and the others are checked as usual. A contract that raises
stops there, so a matched error still hides the checks after it. local_patches in the fleet lock name
workarounds from patches.py, applied before every contract (in this process and
in isolated ones) and recorded in each result's environment.
--record never touches an approved baseline; it writes <name>.candidate.json
next to it, and a reviewer renames it. --golden <platform> checks a run against
another platform's reviewed baseline: discrete outputs must match exactly;
logits are reported as max delta, not compared.
The platform is the device the contract reports, with the CPU architecture
for CPUs (cpu-arm64, cpu-x86_64): numeric outputs differ across CPU
architectures by more than 1e-4 for some fixtures (Whisper), so each has its
own baselines. bfloat16 targets on x86 CPUs with native bfloat16 (avx512_bf16,
amx_bf16) use cpu-x86_64-bf16: their logits differ from emulated bfloat16 by
up to 0.25 (SmolLM-135M), and their tokens can differ. --golden cpu means this
machine's CPU platform.

An unsupported target records why it was skipped (skip): declared in the
manifest, missing hardware on this host, or an undefined variant. A run in
which no target ran (all unsupported, or nothing matched) fails; with
--require-all, any skip the manifest does not declare fails too.
"""

import argparse
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
SCHEMAS = HERE / "schemas"
_DECORATOR = re.compile(r"^\s*#\s*pytest-decorator:.*$", re.MULTILINE)
WEIGHT_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*", "*.tflite", "flax_model*", "tf_model*", "rust_model*"]


@cache
def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


def validate(value, name, where):
    import jsonschema

    try:
        jsonschema.validate(value, schema(name))
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise SystemExit(f"{where}: invalid {name} at {path}: {exc.message}") from None
    return value


def load(path, name):
    return validate(json.loads(Path(path).read_text()), name, path)


def short(repo):
    return repo.split("/")[-1]


def digest(files):
    """One hash for a fixture's {file: sha256} map, as pinned in the fleet lock."""
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def file_hashes(folder):
    skip = ("README.md", "build.json", ".gitattributes")
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(folder.iterdir())
        if f.is_file() and f.name not in skip
    }


def fixture_problem(folder, pin):
    """Why this fixture folder cannot be used for the pinned build, or None."""
    build_file = folder / "build.json"
    hint = f"build it with: python example/build_fixtures.py --fixture {short(str(folder))}"
    if not build_file.exists():
        return f"fixture not built ({hint})"
    build = load(build_file, "fixture-build")
    if pin.get("files_sha256") and digest(build["sha256"]) != pin["files_sha256"]:
        return f"build.json does not match the fleet lock pin ({hint})"
    actual = file_hashes(folder)
    if actual != build["sha256"]:
        changed = sorted(set(actual) ^ set(build["sha256"]) | {k for k in actual if actual[k] != build["sha256"].get(k)})
        return f"fixture files differ from build.json: {changed} ({hint})"
    return None


def contract_files(folder):
    """{relative path: sha256} of a contract's files, as the fleet lock pins them (README and installed packages excluded)."""
    return {
        f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(folder.rglob("*"))
        if f.is_file() and f.name not in ("README.md", ".gitattributes")
        and not {"node_modules", ".cache", ".git"} & set(f.relative_to(folder).parts)
    }


def contract_folder(lock_dir, pin, local):
    """(folder, revision label) of a contract: the pinned Hub revision, or the local folder if unpublished or asked for.

    A local folder is labeled with the pinned revision when its files match the pin, else "local"."""
    if local or pin["revision"] == "unpublished":
        path = source_folder(lock_dir, pin["repo"], pin)
        same = pin.get("files_sha256") and digest(contract_files(path)) == pin["files_sha256"]
        return path, pin["revision"] if same or pin["revision"] == "unpublished" else "local"
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(pin["repo"], revision=pin["revision"]))
    if digest(contract_files(folder)) != pin.get("files_sha256"):
        raise SystemExit(f"{pin['repo']}@{pin['revision'][:10]}: files differ from the fleet lock's files_sha256")
    return folder, pin["revision"]


def checkpoint_path(repo, revision):
    """Local snapshot of a pinned checkpoint with its weights (downloaded if needed)."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cached = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if cached.is_dir() and any(f.suffix in (".safetensors", ".bin") for f in cached.iterdir()):
        return str(cached)
    files = HfApi().list_repo_files(repo, revision=revision)
    ignore = WEIGHT_PATTERNS + (["*.bin", "*.pt"] if any(f.endswith(".safetensors") for f in files) else [])
    return snapshot_download(repo, revision=revision, ignore_patterns=ignore)


def source_folder(lock_dir, repo, pin):
    """The local folder a lock entry names (path), needed only for unpublished or local runs."""
    if "path" not in pin:
        raise SystemExit(f"{repo}: the fleet lock gives no path, so it can only run from its pinned Hub revision")
    return (lock_dir / pin["path"]).resolve()


def fixture_folder(lock_dir, repo, pin, local):
    """The pinned fixture: its Hub revision, or the local build folder if unpublished or asked for."""
    if local or pin["revision"] == "unpublished":
        return source_folder(lock_dir, repo, pin)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=pin["revision"]))


def targets(lock_dir, contract, manifest, framework, fixtures_only=False, local_fixtures=False):
    """Yield (target, kind, model, revision, problem) for every target of a contract."""
    integration = manifest["integrations"][framework]
    unsupported = integration.get("unsupported", {})
    if "fixtures" in integration["targets"]:
        for repo, pin in contract.get("fixtures", {}).items():
            folder = fixture_folder(lock_dir, repo, pin, local_fixtures)
            yield short(repo), "fixture", str(folder), pin["revision"], fixture_problem(folder, pin)
    elif "fixtures" in unsupported:
        for repo in manifest["fixtures"]:
            yield short(repo), "fixture", None, None, unsupported["fixtures"]
    if fixtures_only:
        return
    for repo in manifest["checkpoints"]:
        if repo in unsupported:
            yield short(repo), "checkpoint", None, None, unsupported[repo]
        elif repo not in contract["checkpoints"]:
            continue  # not in this framework's lock
        elif framework == "transformers.js":
            artifact = integration["artifacts"][repo]["repo"]
            yield short(repo), "checkpoint", artifact, contract["artifacts"][artifact], None
        else:
            # Pin by path: the runner resolves the locked revision, the contract
            # just loads what it is given.
            revision = contract["checkpoints"][repo]
            yield short(repo), "checkpoint", lambda r=repo, v=revision: checkpoint_path(r, v), revision, None


def contract_code(entry):
    from doc_builder.testing import DocIntegrationTest

    blocks = DocIntegrationTest._collect_runnable_blocks_from_text(entry.read_text())
    return "\n\n".join(_DECORATOR.sub("", block.code) for block in blocks)


@contextlib.contextmanager
def environ(values):
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def chdir(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def execute_inline(code, filename, env):
    """Run contract code in this process; return (ok, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    namespace = {"__name__": "__mic_contract__", "__file__": filename}
    ok = True
    with environ(env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, filename, "exec"), namespace)
        except BaseException:  # noqa: BLE001 - a contract failure is a result, not a runner crash
            traceback.print_exc(file=err)
            ok = False
    namespace.clear()
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ok, out.getvalue(), err.getvalue()


def patch_prelude(names):
    """Code that applies the fleet lock's local patches in a contract's own process."""
    if not names:
        return ""
    return f"import sys  # mic: local patches\nsys.path.insert(0, {str(HERE)!r})\nimport patches\npatches.apply({sorted(names)!r})\n\n"


class Executor:
    """Runs one contract's code for many targets: in this process, or one process each."""

    def __init__(self, entry, fmt, isolate, workdir, patch_names=(), pinned=None):
        self.entry, self.fmt, self.pinned = entry, fmt, pinned or {}
        self.inline = fmt == "runnable-markdown" and not isolate
        if fmt == "script":
            self.command, self.cwd = ["node", entry.name], entry.parent
        else:
            self.code = contract_code(entry)
            script = workdir / f"{entry.parent.parent.parent.name}-{entry.stem}.py"
            script.write_text(patch_prelude(patch_names) + self.code)
            self.command, self.cwd = [sys.executable, str(script)], entry.parent

    def run(self, env, fresh_process=False):
        if self.inline and not fresh_process:
            with chdir(self.cwd):
                return execute_inline(self.code, str(self.entry), env)
        run = subprocess.run(self.command, cwd=self.cwd, capture_output=True, text=True, env={**os.environ, **env})
        return run.returncode == 0, run.stdout, run.stderr

    def environment(self):
        if self.fmt == "script":
            node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
            package = self.cwd / "node_modules/@huggingface/transformers/package.json"
            return {"node": node, "@huggingface/transformers": json.loads(package.read_text())["version"]}
        names = ["transformers", *(n for n in self.pinned if n != "python")]
        probe = (
            "import importlib.metadata as md, json, platform\n"
            "versions = {'python': platform.python_version()}\n"
            f"for name in {names!r}:\n"
            "    try:\n"
            "        versions[name] = md.version(name)\n"
            "    except md.PackageNotFoundError:\n"
            "        versions[name] = None\n"
            "print(json.dumps(versions))\n"
        )
        if self.inline:
            ok, out, err = execute_inline(probe, "<environment>", {})
            return json.loads(out)
        return json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout)


def environment_differs(pinned, actual):
    """{package: "installed (pinned)"} for every package not at the fleet lock's version."""
    def same(want, have):
        # "2.11.0" pins 2.11.0+cpu too; "3.10" pins any 3.10.x.
        return have is not None and (have == want or have.split("+")[0] == want or have.startswith(want + "."))
    return {name: f"{actual.get(name)} (pinned {want})" for name, want in pinned.items() if not same(want, actual.get(name))}


def lookup(outputs, dotted):
    value = outputs
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def flatten(value):
    if isinstance(value, list):
        return [x for item in value for x in flatten(item)]
    return [value]


def structure(value):
    """The nesting of a list output with its leaves blanked: equal for equal shapes, ragged ones included."""
    return [structure(x) for x in value] if isinstance(value, list) else None


def dims(value):
    """A readable shape, (2, 3) or (2, ragged)."""
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        if len({json.dumps(structure(x)) for x in value}) > 1:
            shape.append("ragged")
            break
        value = value[0] if value else None
    return f"({', '.join(map(str, shape))})"


def known_entry(known, contract, target):
    """The fleet lock's known_failures entry for a target, as a dict (a plain string covers the whole target)."""
    entry = known.get(f"{contract.split('/')[-1].removesuffix('-integration')}/{target}")
    return {"issue": entry} if isinstance(entry, str) else entry


def shown(path):
    """A path for result records: relative to the runner, else to the working directory."""
    for base in (HERE, Path.cwd()):
        try:
            return str(Path(path).resolve().relative_to(base.resolve()))
        except ValueError:
            pass
    return str(path)


def required_outputs(rules, integration):
    """The outputs every result and baseline of this framework must hold, whatever a candidate emits:
    all of the manifest's, minus training outputs for a framework that declares no training."""
    return {name for name in rules if integration["training"] or name.split(".")[0] != "training"}


def missing(required, actual):
    return [f"{name}: missing" for name in sorted(required) if lookup(actual, name) is None]


def check_names(name, got, want=None):
    """Problems with an all_true output: a non-empty {check: bool}, with the baseline's check names when given."""
    if not isinstance(got, dict) or not got:
        return [f"{name}: no named checks ({got!r})"]
    problems = []
    if want is not None:
        if sorted(got) != sorted(want):
            lost, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
            problems.append(f"{name}: checks differ from the baseline (missing {lost}, unexpected {extra})")
    failed = [k for k, v in got.items() if v is not True]
    if failed:
        problems.append(f"{name}: {failed} false")
    return problems


def compare(rules, actual, expected, required=()):
    """Failures of actual against expected. A required output is checked even when one side lacks it;
    an optional one only when either side has it."""
    failures = []
    for name, rule in rules.items():
        got, want = lookup(actual, name), lookup(expected, name)
        if want is None and got is None and name not in required:
            continue  # optional and not produced
        if got is None:
            failures.append(f"{name}: missing")
            continue
        if want is None:
            failures.append(f"{name}: not in the baseline")
            continue
        kind = rule["compare"]
        if kind == "exact" and got != want:
            failures.append(f"{name}: {got!r} != {want!r}")
        elif kind == "allclose":
            a, b = flatten(got), flatten(want)
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in a):
                failures.append(f"{name}: non-finite values")
            elif structure(got) != structure(want):
                failures.append(f"{name}: shape {dims(got)}, baseline has {dims(want)}")
            elif any(abs(x - y) > rule["atol"] + rule["rtol"] * abs(y) for x, y in zip(a, b)):
                failures.append(f"{name}: max delta {max(abs(x - y) for x, y in zip(a, b)):.3g} over tolerance")
        elif kind == "all_true":
            failures += check_names(name, got, want if isinstance(want, dict) else None)
        elif kind == "true" and got is not True:
            failures.append(f"{name}: {got!r}")
    return failures


def violated(rules, actual):
    """The contract's own assertions (all_true, true) that fail, with or without a baseline."""
    failures = []
    for name, rule in rules.items():
        got = lookup(actual, name)
        if got is None:
            continue  # reported by missing()
        if rule["compare"] == "all_true":
            failures += check_names(name, got)
        elif rule["compare"] == "true" and got is not True:
            failures.append(f"{name}: {got!r}")
    return failures


def non_finite(rules, actual):
    """allclose outputs holding NaN or inf: a failure with or without a baseline."""
    return [
        f"{name}: non-finite values"
        for name, rule in rules.items()
        if rule["compare"] == "allclose"
        and lookup(actual, name) is not None
        and not all(isinstance(x, (int, float)) and math.isfinite(x) for x in flatten(lookup(actual, name)))
    ]


@cache
def host():
    """This machine's RAM, free disk, and GPUs, for checking a target's minimum hardware."""
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            gpus = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return {
        "memory_gb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9,
        "disk_gb": shutil.disk_usage(os.environ.get("HF_HOME", Path.home())).free / 1e9,
        "gpus_gb": gpus,
    }


def resource_problem(spec, variant):
    """Why this host cannot run a target with these minimum specs, or None."""
    if not spec:
        return None
    spec = {**spec, **spec.get("variants", {}).get(variant, {})}
    h = host()
    device = "cuda" if h["gpus_gb"] else "cpu"  # what the contracts pick
    devices = spec.get("devices", ["cpu", "cuda"])
    if spec.get("gpus", 0) and "cpu" in devices:
        devices = [d for d in devices if d != "cpu"]
    if device not in devices:
        return f"needs {' or '.join(devices)}; this host has {'no GPU' if device == 'cpu' else device}"
    problems = []
    if device == "cuda":
        if len(h["gpus_gb"]) < max(spec.get("gpus", 1), 1):
            problems.append(f"{spec.get('gpus', 1)} GPUs (host has {len(h['gpus_gb'])})")
        if spec.get("gpu_memory_gb", 0) > min(h["gpus_gb"]):
            problems.append(f"{spec['gpu_memory_gb']:g} GB per GPU (host has {min(h['gpus_gb']):.0f})")
    if spec.get("memory_gb", 0) > h["memory_gb"]:
        problems.append(f"{spec['memory_gb']:g} GB RAM (host has {h['memory_gb']:.0f})")
    if spec.get("disk_gb", 0) > h["disk_gb"]:
        problems.append(f"{spec['disk_gb']:g} GB free disk (host has {h['disk_gb']:.0f})")
    return f"needs {', '.join(problems)}" if problems else None


def native_bfloat16():
    """Whether this x86 CPU computes bfloat16 natively (Linux only)."""
    try:
        flags = set(Path("/proc/cpuinfo").read_text().split())
    except OSError:
        return False
    return bool(flags & {"avx512_bf16", "amx_bf16"})


def platform_class(device, dtype=None):
    """The baseline platform: the device, with the architecture for CPUs, and
    native bfloat16 support for bfloat16 runs on x86."""
    if device != "cpu":
        return device
    machine = os.uname().machine.lower()
    arch = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
    if arch == "x86_64" and dtype == "bfloat16" and native_bfloat16():
        return "cpu-x86_64-bf16"
    return f"cpu-{arch}"


def rules_for(manifest, platform_dir):
    """The manifest's comparison rules with this platform's tolerance overrides."""
    rules = {name: dict(rule) for name, rule in manifest["outputs"].items()}
    overrides = platform_dir / "tolerances.json"
    if overrides.exists():
        for name, tolerance in load(overrides, "tolerances")["outputs"].items():
            if rules.get(name, {}).get("compare") != "allclose":
                raise SystemExit(f"{overrides}: {name} is not an allclose output of this contract")
            rules[name].update(tolerance)
    return rules


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", choices=("transformers", "transformers.js"), required=True)
    parser.add_argument("--contract", action="append", help="run only this contract (repeatable)")
    parser.add_argument("--target", action="append", help="run only this target, fixture or checkpoint name (repeatable)")
    parser.add_argument("--fixtures-only", action="store_true", help="skip real checkpoints")
    parser.add_argument("--local-fixtures", action="store_true", help="use the local build folders instead of the pinned Hub revisions")
    parser.add_argument("--local-contracts", action="store_true",
                        help="run the contracts in example/hub instead of their pinned Hub revisions (records contract_revision 'local' if they differ)")
    parser.add_argument("--lock", type=Path, help="fleet lock to run (default <framework>/tests/contracts/fleet.lock next to this script); expectations/ sits next to it")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--isolate", action="store_true", help="run each target in its own process")
    parser.add_argument("--record", action="store_true", help="write candidate baselines for review")
    parser.add_argument("--golden", metavar="PLATFORM", help="check discrete outputs against this platform's baseline")
    parser.add_argument("--output", type=Path, help="append result records to this JSON lines file")
    parser.add_argument("--changed-files", type=Path, metavar="FILE",
                        help="run only the contracts these changed Transformers files can affect (one path per line; - for stdin)")
    parser.add_argument("--run-all", action="store_true",
                        help="with --changed-files: run every contract anyway, recording selected_by (null if not selected)")
    parser.add_argument("--require-all", action="store_true",
                        help="fail on targets skipped for this host's hardware or an undefined variant; only skips the manifest declares are allowed")
    args = parser.parse_args()
    if args.golden:
        args.golden = platform_class(args.golden)

    lock_path = args.lock or HERE / args.framework / "tests/contracts/fleet.lock"
    lock_dir = lock_path.resolve().parent
    lock = load(lock_path, "fleet-lock")
    unknown = sorted(set(args.contract or ()) - set(lock["contracts"]))
    if unknown:
        raise SystemExit(f"no contract named {unknown} in {args.framework}'s fleet lock ({sorted(lock['contracts'])})")
    started = time.perf_counter()
    counts = {}
    status = 0

    selected_by = None
    if args.changed_files:
        import select_contracts

        lines = sys.stdin.read() if str(args.changed_files) == "-" else args.changed_files.read_text()
        dirs = {
            name: sorted({d for repo, pin in contract.get("fixtures", {}).items()
                          for d in select_contracts.code_dirs(fixture_folder(lock_dir, repo, pin, args.local_fixtures))})
            for name, contract in lock["contracts"].items()
        }
        selected_by = select_contracts.select([line.strip() for line in lines.splitlines() if line.strip()], dirs)
        print(json.dumps({"selection": selected_by, "code_dirs": dirs}), file=sys.stderr, flush=True)
    known = lock.get("known_failures", {})
    local_patches = sorted(lock.get("local_patches", {}))
    if local_patches and args.framework == "transformers":
        import patches

        patches.apply(local_patches)

    def emit(record):
        nonlocal status
        entry = known_entry(known, record["contract"], record["target"])
        if entry and not args.record and not args.golden and ("match" in entry or "skip_outputs" not in entry):
            issue = entry["issue"]
            if record["status"] in ("fail", "error"):
                if "match" not in entry or all(entry["match"] in f for f in record["failures"]):
                    record = {**record, "status": "xfail", "known_failure": issue}
            elif record["status"] == "pass":
                record = {**record, "status": "fail", "failures": [f"listed in known_failures but passes; remove it: {issue}"]}
        validate(record, "result", f"result for {record['contract']}/{record['target']}")
        print(json.dumps(record), flush=True)
        if args.output:
            with args.output.open("a") as f:
                f.write(json.dumps(record) + "\n")
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        status |= record["status"] in ("fail", "error")
        status |= args.require_all and record["status"] == "unsupported" and record["skip"] != "declared"

    with tempfile.TemporaryDirectory() as tmp:
        for name, contract in lock["contracts"].items():
            if args.contract and name not in args.contract:
                continue
            if selected_by is not None and name not in selected_by and not args.run_all:
                continue
            root, contract_revision = contract_folder(lock_dir, contract["integration"], args.local_contracts)
            manifest = load(root / "integration.json", "integration")
            integration = manifest["integrations"][args.framework]
            if integration["format"] == "script" and contract_revision not in ("local", "unpublished"):
                # Script contracts run where their packages are installed: the local folder, which must match the pin.
                root, contract_revision = contract_folder(lock_dir, contract["integration"], True)
                if contract_revision == "local":
                    raise SystemExit(f"{root}: differs from the pinned {contract['integration']['repo']}; publish it or pass --local-contracts")
            base = {
                "contract": contract["integration"]["repo"],
                "contract_revision": contract_revision,
                "framework": args.framework,
                "variant": args.variant,
                **({"selected_by": selected_by.get(name)} if selected_by is not None else {}),
            }
            variant = integration["variants"].get(args.variant)
            required = required_outputs(manifest["outputs"], integration)
            executor = Executor(root / integration["entry_point"], integration["format"], args.isolate, Path(tmp), local_patches, lock.get("environment"))
            env_info = None
            for target, kind, model, revision, problem in targets(lock_dir, contract, manifest, args.framework, args.fixtures_only, args.local_fixtures):
                if args.target and target not in args.target:
                    continue
                record = {**base, "target": target, "kind": kind, "revision": revision}
                if variant is None:
                    emit({**record, "status": "unsupported", "skip": "variant", "failures": [f"variant {args.variant!r} not defined"]})
                    continue
                if problem:
                    if model is None:  # the manifest's unsupported entry
                        emit({**record, "status": "unsupported", "skip": "declared", "failures": [problem]})
                    else:
                        emit({**record, "status": "error", "failures": [problem]})
                    continue
                resources = integration.get("resources", {})
                short_names = {short(r): r for r in manifest["checkpoints"]}
                spec = resources.get("fixture") if kind == "fixture" else resources.get(short_names.get(target))
                too_small = resource_problem(spec, args.variant)
                if too_small:
                    emit({**record, "status": "unsupported", "skip": "hardware", "failures": [too_small]})
                    continue
                if callable(model):
                    model = model()
                env_info = env_info or executor.environment()
                start = time.perf_counter()
                env = {
                    "MIC_RUNNER": "1",
                    "MIC_MODEL": model,
                    "MIC_MODEL_REVISION": "" if os.path.isdir(model) else revision,
                    "MIC_MODEL_KWARGS": json.dumps(variant),
                }
                ok, stdout, stderr = executor.run(env)
                environment = {**env_info, "local_patches": local_patches} if local_patches else env_info
                record.update(environment=environment, seconds=round(time.perf_counter() - start, 2))
                differs = environment_differs(lock.get("environment", {}), env_info) if integration["format"] != "script" else {}
                if differs:
                    record["environment_differs"] = differs
                lines = [line for line in stdout.splitlines() if line.startswith("{")]
                if not ok or not lines:
                    emit({**record, "status": "error", "failures": [stderr.strip()[-1500:] or "no result record"]})
                    continue
                actual = json.loads(lines[-1])
                platform = platform_class(actual["runtime"]["device"], actual["runtime"].get("dtype"))
                record["platform"] = platform
                entry = known_entry(known, record["contract"], target)
                skipped = sorted(entry.get("skip_outputs", [])) if entry and not args.record else []
                outputs = {k: r for k, r in manifest["outputs"].items() if k not in skipped}
                checked = required - set(skipped)
                if skipped:
                    record["skipped_outputs"] = skipped
                suffix = "" if args.variant == "default" or args.golden else f".{args.variant}"
                platform_dir = lock_dir / "expectations" / name / (args.golden or platform)
                baseline = platform_dir / f"{target}{suffix}.json"
                record["baseline"] = shown(baseline)
                if args.golden:
                    if not baseline.exists():
                        emit({**record, "status": "error", "failures": [f"no golden baseline at {record['baseline']}"]})
                        continue
                    expected = load(baseline, "baseline")["outputs"]
                    exact = {k: r for k, r in outputs.items() if r["compare"] == "exact" and k != "runtime"}
                    failures = compare(exact, actual, expected, checked)
                    deltas = {
                        k: max(abs(x - y) for x, y in zip(flatten(lookup(actual, k)), flatten(lookup(expected, k))))
                        for k, r in outputs.items()
                        if r["compare"] == "allclose" and lookup(actual, k) is not None
                    }
                    emit({**record, "golden": args.golden, "status": "fail" if failures else "pass", "failures": failures, "max_delta": deltas})
                    continue
                broken = missing(checked, actual) + non_finite(outputs, actual) + violated(outputs, actual)
                if broken:
                    # Never record incomplete or non-finite outputs or failed assertions: a baseline would approve the bug.
                    emit({**record, "status": "fail", "failures": broken})
                    continue
                if args.record:
                    ok2, stdout2, _ = executor.run(env, fresh_process=True)  # must reproduce outside this process
                    again = [line for line in stdout2.splitlines() if line.startswith("{")]
                    # Same rules as a check: exact outputs equal, numeric ones within this platform's tolerance.
                    unstable = compare(rules_for(manifest, platform_dir), json.loads(again[-1]), actual, required) if again else ["no result record"]
                    if not ok2 or unstable:
                        emit({**record, "status": "fail", "failures": [f"not recorded: a fresh-process rerun differs: {unstable}"]})
                        continue
                    candidate = baseline.with_name(f"{target}{suffix}.candidate.json")
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    recorded = {k: v for k, v in record.items() if k != "baseline"}
                    candidate.write_text(json.dumps(validate({"recorded_with": recorded, "outputs": actual}, "baseline", candidate), indent=1) + "\n")
                    emit({**record, "status": "recorded", "baseline": shown(candidate)})
                    continue
                if not baseline.exists():
                    failures = [f"no reviewed baseline at {record['baseline']}"]
                else:
                    reviewed = load(baseline, "baseline")
                    identity = reviewed["recorded_with"]
                    mismatch = [k for k in ("target", "variant", "framework") if identity.get(k) != record[k]]
                    if identity.get("platform", platform) != platform:
                        mismatch.append("platform")
                    failures = [f"baseline identity differs: {mismatch}"] if mismatch else []
                    rules = {k: r for k, r in rules_for(manifest, platform_dir).items() if k not in skipped}
                    failures += compare(rules, actual, reviewed["outputs"], checked)
                emit({**record, "status": "fail" if failures else "pass", "failures": failures})
    if not set(counts) - {"unsupported"} and (counts or args.contract or args.target):
        # Every requested target was skipped, or none matched: nothing was checked, which is not a pass.
        print(f"no target ran: {counts['unsupported']} unsupported" if counts else "no target matched", file=sys.stderr)
        status = 1
    summary = {"summary": counts, "seconds": round(time.perf_counter() - started, 1), "isolate": args.isolate}
    print(json.dumps(summary), file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
