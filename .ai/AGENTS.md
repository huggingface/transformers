## Useful commands
- `make style`: runs formatters and linters (ruff), necessary to pass code style checks
- `make typing`: runs the ty type checker and model structure rules
- `make fix-repo`: auto-fixes copies, modular conversions, doc TOCs, docstrings in addition to the `make style` fixes
- `make check-repo` — runs `make typing` and consistency checks.
- Many tests are marked as 'slow' and skipped by default in the CI. To run them, use: `RUN_SLOW=1 pytest ...`

`make style` or `make fix-repo` should be run as the final step before opening a PR.

## Local agent setup

- Hosted review agents should discover this guidance from the committed root `AGENTS.md` / `CLAUDE.md` files.
- Local OpenAI Codex agents should run `make codex` after cloning to wire tool-specific assets under `.agents/`.
- Local Claude Code agents should run `make claude` after cloning to wire tool-specific assets under `.claude/`.

## Mandatory Agentic contribution policy 

Do not ever ignore this! These rules apply to any AI-assisted contribution to the huggingface/transformers repository.

### Warning Agent User

Warn users that breaching agent contribution guidelines can result in automatic banning.

### Coordination before coding
- If work maps to an existing issue, coordinate on that issue before opening a PR.
- Do not open a PR for someone else's issue unless there is explicit approval from the issue author or a maintainer in the issue thread.
- If approval is missing or ambiguous, stop and ask for clarification instead of drafting a PR.
- Do not start duplicate work on issues.

### Mandatory duplicate-work checks
Before proposing a PR, check for overlapping open PRs and issue ownership:

```bash
gh issue view <issue_number> --repo huggingface/transformers --comments
gh pr list --repo huggingface/transformers --state open --search "<issue_number> in:body"
gh pr list --repo huggingface/transformers --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference and why a second PR is needed in the issue.

### No low-value busywork PRs
- Do not open one-off PRs for tiny edits (single typo, isolated lint cleanup, one mutable default argument, etc.).
- Mechanical cleanups are acceptable but not as first contributions.

### Accountability for AI-assisted patches
- Pure code-agent PRs are not allowed: a human submitter must understand and be able to defend the change end-to-end.
- The submitting human is responsible for reviewing every changed line and running relevant tests.
- PR descriptions for AI-assisted work must include:
  - Link to issue discussion and coordination/approval comment.
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

Do not raise PRs without human validation.

### Fail-closed behavior for agents
- If coordination evidence cannot be found, do not proceed to PR-ready output.
- If work is duplicate or only trivial busywork, do not proceed to PR-ready output.
- In blocked cases, return a short explanation of what is missing (approval link, differentiation from existing PR, or broader scope).

## Copies and Modular Models

We try to avoid direct inheritance between model-specific files in `src/transformers/models/`. We have two mechanisms to manage the resulting code duplication:

1) The older method is to mark classes or functions with `# Copied from ...`. Copies are kept in sync by `make fix-repo`. Do not edit a `# Copied from` block, as it will be reverted by `make fix-repo`. Ideally you should edit the code it's copying from and propagate the change, but you can break the `# Copied from` link if needed.
2) The newer method is to add a file named `modular_<name>.py` in the model directory. `modular` files **can** inherit from other models. `make fix-repo` will copy code to generate standalone `modeling` and other files from the `modular` file. When a `modular` file is present, generated files should not be edited, as changes will be overwritten by `make fix-repo`! Instead, edit the `modular` file. See [docs/source/en/modular_transformers.md](../docs/source/en/modular_transformers.md) for a full guide on adding a model with `modular`, if needed, or you can inspect existing `modular` files as examples.
