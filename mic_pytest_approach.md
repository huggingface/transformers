# MIC → pytest approach discussion

## Context

The `fast-integration / run` CI job (e.g. run 41873731105, job 126738176377) runs
**Model Integration Contracts (MIC)**, NOT the `FastIntegrationTestMixin` tests from
`fast_integration_tests.md`. These are two separate systems.

Workflow file: `huggingface/transformers-ci/.github/workflows/pr-ci_contracts.yml@pr_ci_torch_215`

## What the job currently does

Three sequential shell steps calling `run-model-contracts` (a CLI wrapping `run.main()`):

```
fleet=control    + variant=default   →   8 results
fleet=control    + variant=bfloat16  →   8 results
fleet=prototypes + variant=bfloat16  → 190 results (182 pass, 3 fail, 1 error, 4 unsupported)
```

Results written to `/tmp/results.jsonl` (one JSONL line per `(contract_name, target, variant)`).
`report-model-contracts` renders a Markdown table into the GitHub Step Summary.

Total: **206 "test items"** already in the data, just not expressed as pytest.

## The proposed approach (Option B)

**Goal**: make each contract check a real pytest test item.

**Flow**:
```
1. Generate test_fast_integration.py   — 206 parametrized entries baked in as a static list
2. pytest test_fast_integration.py     — each test executes its own contract
```

No separate `run-model-contracts` pre-run step. Each test body calls `run.main()` directly.

### Key facts

- `run.main(argv)` is a **regular Python function**, not a subprocess. It runs in the same
  pytest process. Already used this way in `transformers-ci/tests/test_contracts.py`:
  ```python
  code = run.main(["--lock", str(fleet[0]), "--fixtures-only", *args])
  ```
- Same process → `HF_HOME` disk cache shared across all 206 calls. First test for each
  model downloads its fixture; subsequent pytest runs find it cached. No re-download issue.
- `--contract SmolLM --target mic-SmolLM-slim` flags allow single-item targeting.
- `_DECORATOR = re.compile(r"^\s*#\s*pytest-decorator:.*$", ...)` in run.py suggests
  pytest integration was always a design intent.

### Generated test file shape

```python
# test_fast_integration.py  (generated before pytest runs)

import pytest
from transformersci.contracts import run as run_api

PARAMS = [
    # (contract_name, target, variant)  — 206 entries baked in from fleet manifest
    ("SmolLM", "mic-SmolLM-slim", "default"),
    ("SmolLM", "mic-SmolLM-slim", "bfloat16"),
    ("BERT",   "mic-bert-base-uncased-slim", "default"),
    # ... 203 more
]

@pytest.mark.parametrize("contract,target,variant", PARAMS)
def test_contract(contract, target, variant):
    code = run_api.main([
        "--fixtures-only", "--run-all",
        "--contract", contract,
        "--target", target,
        "--variant", variant,
    ])
    assert code == 0
```

### Collection time: where do PARAMS come from?

The parametrize list must be known **at collection time** (before any test runs).
The "generate upfront" step solves this: before calling `pytest`, a script reads the
fleet manifest from `huggingface-mic/transformers-fleet` (or the fleet lock file already
downloaded during install) and writes the static list into `test_fast_integration.py`.
Collection just reads the file — no Hub call needed at collection time.

### Runtime

Same total time as current (~320s) since the same 206 contracts execute.
With `pytest-xdist`, tests can be parallelized across workers (each worker gets its own
process, `HF_HOME` cache shared via disk).

## Fleet / contract source

- Fleet definition: `huggingface-mic/transformers-fleet` Hub dataset
- Python API: `from transformersci.contracts.fleet import lock_path, resolve_revision`
- Lock file locations: `fleet-lock.json` (control) and `prototype-fleet.lock` (prototypes)
- Fleet revision pinned in `tests/contracts.lock` (40-char SHA)

## Status

Discussion only — not implemented yet. Next step would be writing the generation script
and the test file template, then wiring into the `pr-ci_contracts.yml` workflow.
