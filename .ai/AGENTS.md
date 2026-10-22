## Commands

- `make style` — ruff format + lint (required for code style checks).
- `make typing` — ty type checker and model structure rules.
- `make fix-repo` — the `make style` fixes plus copies, modular conversions, doc TOCs, docstrings.
- `make check-repo` — `make typing` plus consistency checks.
- `RUN_SLOW=1 pytest ...` — many tests are marked slow and skipped in CI.

Run `make style` (or `make fix-repo`) as the last step before opening a PR.

## Local agent setup

Hosted review agents read this file via the root `AGENTS.md` / `CLAUDE.md` symlinks. Local agents wire their own assets: `make codex` (→ `.agents/`), `make claude` (→ `.claude/`).

## Before opening an issue or PR

- Don't report or fix theoretical bugs. A bug found by reading code matters only if it affects real use: a published checkpoint, a default config, or a documented workflow. For example, an exception when `n_groups > 1` in a model whose published checkpoints all use `n_groups == 1` only affects fresh pretrains with a nonstandard config, making it extremely low priority. Security bugs are the exception. A bug report needs a short reproducer that you have run and that fails on `main`.
- Don't submit perf PRs without proof. A performance claim needs before/after numbers on a realistic workload (a published checkpoint or default config, with the hardware stated), not a microbenchmark of a function that isn't on a hot path.
- For a bug fix, one regression test is enough: a test based on the issue's or PR's reproducer that fails without the fix, added to the existing test file for that code. Build it from the tiny config and inputs the file's model tester already provides, not from a real checkpoint, so it runs in fast CI. Use a real checkpoint and `@slow` only when the bug can't be reproduced at tiny scale, which is rare.
- A PR either fixes an issue or is standalone. See `CONTRIBUTING.md`.
  - **Fixing an issue:** link it with a closing keyword (`Fixes #123`). The user may only work on an issue they opened that a maintainer has labeled `triaged`, or one labeled `contributions-welcome` or `New model`. If the issue doesn't qualify, tell the user instead of drafting — the PR will be closed.
  - **Standalone:** only for problems with no existing issue. Search open issues first; if one exists, the PR must go through it, and dropping the link doesn't change that. Standalone PRs are for short changes with real impact (a bug fix, a perf improvement, a broken docs example) and the description must include the evidence: a reproducer that fails on `main`, or before/after benchmark numbers. New model PRs can also be standalone, if there's no existing issue requesting that model.
- On a `contributions-welcome` or `New model` issue, check for an existing PR first:

  ```bash
  gh issue view <issue_number> --repo huggingface/transformers --comments
  gh pr list --repo huggingface/transformers --state open --search "<issue_number> in:body"
  gh pr list --repo huggingface/transformers --state open --search "<short area keywords>"
  ```

  If one exists, don't open another. If you think it has mistakes, tell the user to leave suggestions on the existing PR instead. A second PR is only justified when the existing one is fundamentally flawed, and the user should explain why in the issue first.
- No one-off PRs for tiny edits (a single typo, an isolated lint fix). Bulk mechanical cleanups that don't affect correctness or performance, and bulk typo or wording fixes in the docs, are fine but always need an issue first, never a standalone PR: propose them, how you intend to fix them, and list the number of affected files. Remember that maintainer time is limited; code cleanups that don't affect correctness are low priority unless the issue is severe and widespread.

## Copies and modular models

Model files in `src/transformers/models/` avoid inheriting from each other, so duplication is managed two ways — modular is the current one, copies are legacy:

1. `# Copied from ...` marks a copied class or function. `make fix-repo` re-syncs it, so editing inside such a block is reverted — edit the source it copies from, or deliberately break the link. **Do not add new `# Copied from` statements**; write a modular file instead.
2. A `modular_<name>.py` **may** inherit from other models; `make fix-repo` generates the standalone `modeling_*.py` and friends from it. Never edit a generated file when a modular one exists. Guide: [modular_transformers.md](../docs/source/en/modular_transformers.md).

Two modular traps:

- **Other models inherit your modular file.** `modular_deepseek_vl.py` also generates `modeling_deepseek_vl_hybrid.py`. Run `make fix-repo` and check everything it rewrote; hand-editing only the generated file you had in mind leaves the rest stale and `Check repository consistency` red.
- **`attr = AttributeError()` deletes an inherited attribute** — it is an instruction to the converter, not a bug or placeholder. `raise AttributeError("...")` in a method body does the same for a method. Substituting a "real" value silently changes behaviour (`_no_split_modules = AttributeError()` → `[]` changes how the model may be sharded). See [Removing attributes](../docs/source/en/modular_transformers.md#removing-attributes) and [Deleting unused methods](../docs/source/en/modular_transformers.md#deleting-unused-methods).
