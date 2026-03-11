---
name: circleci-diagnose-failures
description: Diagnose and fix failing CircleCI checks on GitHub pull requests. Use when a user shares a GitHub PR URL with failing CI checks, asks about CircleCI failures, wants to understand why CI is red, or asks to fix failing checks. Works with public open-source repos without authentication.
---

# CircleCI Diagnose Failures

Retrieve and diagnose failing CircleCI check details for a GitHub PR using the GitHub CLI and CircleCI REST APIs. No CircleCI CLI installation or API token is needed for public/open-source repositories.

## Workflow

### Step 1: List all checks on the PR

Use the GitHub CLI to get a summary of all checks, their status, and their CircleCI URLs:

```bash
gh pr checks <PR_NUMBER> --repo <owner>/<repo>
```

Output columns: `name`, `status (pass/fail)`, `duration`, `URL`, `description`.

Extract the CircleCI job numbers from the URLs. CircleCI job URLs look like:
`https://circleci.com/gh/<org>/<repo>/<JOB_NUMBER>`

### Step 2: Get job metadata

For each failing job, fetch metadata via the CircleCI v2 API:

```bash
curl -s "https://circleci.com/api/v2/project/gh/<org>/<repo>/job/<JOB_NUMBER>" | python3 -m json.tool
```

Key fields: `status`, `name`, `duration`, `parallel_runs` (shows which parallel index failed), `latest_workflow.name`.

### Step 3: Get step-level details with output URLs

Use the CircleCI v1.1 API for full step details including output URLs:

```bash
curl -s "https://circleci.com/api/v1.1/project/github/<org>/<repo>/<JOB_NUMBER>"
```

Parse the response to find failed steps:

```python
import json, sys
data = json.load(sys.stdin)
for step in data.get('steps', []):
    name = step.get('name', 'unnamed')
    for action in step.get('actions', []):
        if action.get('status') == 'failed':
            print(f"FAILED: {name}")
            print(f"  output_url: {action.get('output_url')}")
            print(f"  parallel_index: {action.get('index')}")
```

### Step 4: Fetch the actual error output

Each failed action has an `output_url` containing a presigned URL. Fetch it to get the error message:

```bash
curl -s "<output_url>"
```

The response is JSON: an array of objects with `message` (the log text), `type` ("out" or "error"), and `time` fields. The `message` field contains ANSI escape codes; read through them for the actual error.

### Step 5: Diagnose and fix

Common failure patterns in `huggingface/transformers`:

| Check name | Common cause | Fix |
|---|---|---|
| `check_repository_consistency` → `check_modular_conversion.py` | Generated file was manually edited and diverged from `modular_*.py` | Run `make fix-repo` to regenerate |
| `check_repository_consistency` → `check_copies.py` | A `# Copied from` block was edited directly | Edit the source, then `make fix-repo` |
| `check_code_quality` → `ruff check` | Linting/formatting errors | Run `make style` |
| `tests_torch` → "worker crashed" | OOM or flaky CI worker crash, not a code bug | Re-run the workflow; typically transient |
| `check_config_attributes.py` | Config class attributes don't match docstring | Update config docstring or defaults |

After identifying the root cause, apply the fix locally and push.

## Parallel jobs

For jobs with `parallelism > 1`, the v2 API `parallel_runs` array shows which index failed. Use the v1.1 API to filter actions by `action['index'] == <failed_index>` to find the specific failing step and its output.

## Notes

- **No authentication required** for public repos. The CircleCI v1.1 and v2 APIs serve public project data without a token.
- **Output URLs are presigned** with short-lived JWTs. Fetch them promptly; they expire.
- **Workflow-level checks** (`run_tests`, `setup_and_quality`) aggregate multiple jobs. Drill into the individual `ci/circleci:*` jobs for actionable details.
- For the `huggingface/transformers` repo, use `gh/huggingface/transformers` (v2) or `github/huggingface/transformers` (v1.1) as the project slug.
