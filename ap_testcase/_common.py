"""Shared bootstrap and reporting helpers for the Apertus integration scripts."""

import os
from dataclasses import dataclass
from pathlib import Path


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
MAX_MESSAGE_LENGTH = 120
DEFAULT_CHECKPOINT = "swiss-ai/Apertus-v1.5-8B"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRANSFORMERS_DIR = (REPO_ROOT / "src" / "transformers").resolve()


@dataclass
class CaseResult:
    name: str
    status: str
    message: str


class CaseSkipped(Exception):
    """Signal that a case cannot run in the current environment."""


def one_line(message):
    message = " ".join(str(message).split()).replace("|", "¦")
    if len(message) > MAX_MESSAGE_LENGTH:
        return f"{message[: MAX_MESSAGE_LENGTH - 3]}..."
    return message


def case_name(case):
    return case.__doc__.splitlines()[0] if case.__doc__ else case.__name__


def require_local_transformers(required_symbols=()):
    import transformers

    imported_from = Path(transformers.__file__).resolve().parent
    if imported_from != EXPECTED_TRANSFORMERS_DIR:
        raise RuntimeError(
            f"expected Transformers from {EXPECTED_TRANSFORMERS_DIR}, but imported it from {imported_from}; "
            'install this checkout with `uv pip install -e ".[testing,vision,audio]"`'
        )

    missing = [symbol for symbol in required_symbols if not hasattr(transformers, symbol)]
    if missing:
        raise RuntimeError(f"this Transformers checkout is missing required symbols: {', '.join(missing)}")
    return transformers


def resolve_checkpoint(processor_only=False):
    checkpoint = os.environ.get("APERTUS1P5_CHECKPOINT", DEFAULT_CHECKPOINT)
    expanded = Path(os.path.expandvars(os.path.expanduser(checkpoint)))

    if expanded.is_dir():
        resolved = expanded.resolve()
    elif expanded.is_absolute() or checkpoint.startswith((".", "~")):
        raise FileNotFoundError(f"local checkpoint directory does not exist: {expanded}")
    else:
        from huggingface_hub import snapshot_download

        repo_id, _, revision = checkpoint.partition("@")
        ignore_patterns = ["*.safetensors*"] if processor_only else None
        resolved = Path(
            snapshot_download(repo_id, revision=revision or None, ignore_patterns=ignore_patterns)
        ).resolve()

    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint has no config.json: {resolved}")
    return str(resolved)


def bootstrap(required_symbols=(), processor_only=False):
    transformers = require_local_transformers(required_symbols)
    checkpoint = resolve_checkpoint(processor_only=processor_only)
    return transformers, checkpoint


def run_case(case, *args):
    name = case_name(case)
    try:
        message = case(*args)
    except CaseSkipped as error:
        return CaseResult(name, SKIP, one_line(error))
    except AssertionError as error:
        details = str(error) or "no details"
        return CaseResult(name, FAIL, one_line(f"AssertionError: {details}"))
    except Exception as error:
        return CaseResult(name, FAIL, one_line(f"{type(error).__name__}: {error}"))
    return CaseResult(name, PASS, one_line(message))


def skipped_result(case, message):
    return CaseResult(case_name(case), SKIP, one_line(message))


def setup_failure(error):
    return CaseResult("SETUP", FAIL, one_line(f"{type(error).__name__}: {error}"))


def print_results(results):
    rows = [("CASE", "STATUS", "MESSAGE")]
    rows.extend((result.name, result.status, result.message) for result in results)
    widths = [max(len(row[index]) for row in rows) for index in range(3)]
    separator = f"+-{'-+-'.join('-' * width for width in widths)}-+"

    print("\nRESULTS")
    print(separator)
    for index, row in enumerate(rows):
        print(f"| {' | '.join(value.ljust(width) for value, width in zip(row, widths))} |")
        if index == 0:
            print(separator)
    print(separator)

    counts = {status: sum(result.status == status for result in results) for status in (PASS, FAIL, SKIP)}
    print(f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped")


def finish(results):
    print_results(results)
    return int(any(result.status == FAIL for result in results))
