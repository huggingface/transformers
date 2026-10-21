<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribute to 🤗 Transformers

## Contribution rules

Every PR must link an open issue that you're allowed to work on. That means either:

- an issue you opened, once a maintainer has labeled it `triaged`
- an issue labeled [`contributions-welcome`](https://github.com/huggingface/transformers/labels/contributions-welcome)

Other people's issues aren't open for PRs unless a maintainer has said so. PRs that don't link a qualifying issue will be closed. Hugging Face staff, maintainers and some long-standing contributors are exempt.

How you write your code is up to you, but you're responsible for everything you submit, and we may block accounts that repeatedly open low-quality issues or PRs. Please also respect our [code of conduct](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

<details>

<summary> What we look for in issues and PRs </summary>

- Clear diagnosis of bugs. Quick workarounds often cause code bloat or incompatibilities with other models. Spending time to track down the exact cause of a problem, and in particular
locating the first commit where it appeared (for example with [git bisect](https://git-scm.com/docs/git-bisect)) is valuable.
- Minimize the diff. Check your PR to eliminate any unnecessary changes. Ensure that you did not commit any testing scripts
or unrelated files. Add comments only if they're really necessary. If your PR can be a 1-line fix,
make it a 1-line fix. This makes the PR much easier to review and improves the chances that it will be accepted.
- Take the time to reproduce the problem. Very often when a user reports an issue, the issue is actually caused by
environment issues on their machine, or they misdiagnose the problem and suggest an invalid solution. Trusting the
report too much leads to bad solutions, sometimes for problems that do not exist! Writing a simple reproducer script and running it to make sure you see the problem is valuable.
- Compare against other models. The Transformers repo is very large, and many models are doing similar things. When
fixing a bug, it's valuable to see if the bug exists in other models. If your PR says
"fixed by using the same approach as (other model)", with a link to the relevant code, that is very helpful for maintainers,
because it tells us that the fix is likely to be correct and compatible with the rest of the codebase.
- Avoid small or "busywork" PRs. We don't have time for small style changes or typo fixes in comments, or for
"theoretical bugs" found by code analysis that rarely if ever cause problems in practice. These are generally not worth
fixing unless they can be exploited by an attacker.
- Verify tests locally and in the CI. Before opening a PR, run `make fix-repo` and use `utils/tests_fetcher.py` to 
see a list of tests that cover the files you have changed in your PR branch. Run those tests locally, and make sure 
they pass before you open a PR. After you open your PR, please verify that the CI is green and fix any issues before 
pinging anyone for review! This reduces notification spam a lot, which keeps maintainers sane.

</details>

## Set up

1. Fork [Transformers](https://github.com/huggingface/transformers) with the **Fork** button on GitHub to get a copy under your account, then clone it locally.

  ```bash
  git clone git@github.com:<your Github handle>/transformers.git
  cd transformers
  ```

2. Add the Transformers repository as a remote named `upstream`. `origin` points at your fork and `upstream` points at the source of truth. That lets you sync your local `main` before pushing to `origin` and opening a PR.

  ```bash
  git remote add upstream https://github.com/huggingface/transformers.git
  git remote -v
  # origin    git@github.com:<your Github handle>/transformers.git (fetch)
  # origin    git@github.com:<your Github handle>/transformers.git (push)
  # upstream  https://github.com/huggingface/transformers.git (fetch)
  # upstream  https://github.com/huggingface/transformers.git (push)
  ```

3. Keep your fork up to date and start a branch to work on. You can also sync your fork from the GitHub UI with the **Sync fork** button.

  ```bash
  git checkout main && git pull upstream main && git push origin main
  git switch -c a-descriptive-name-for-my-branch
  ```

4. Install the library in editable mode. Use `[dev]` for most contributions because it includes the development tools needed for style and quality checks. Use `[torch,testing]` for model contributions, or `[quality]` for docs-only changes and small fixes when the full development dependency set is not needed.

  ```bash
  pip install -e ".[dev]"

  # for model contributions
  pip install -e ".[torch,testing]"

  # for docs-only changes and small fixes
  pip install -e ".[quality]"
  ```

### Windows

On Windows, unless you're using [WSL](https://learn.microsoft.com/en-us/windows/wsl/), configure git to convert `CRLF` line endings to `LF`.

```bash
git config core.autocrlf input
```

One way to run the `make` command on Windows is with MSYS2.

1. [Download MSYS2](https://www.msys2.org/) (we assume it's installed in `C:\msys64`).
2. Open the command line `C:\msys64\msys2.exe` (it should be available from the **Start** menu).
3. Run in the shell: `pacman -Syu` and install `make` with `pacman -S make`.
4. Add `C:\msys64\usr\bin` to your PATH environment variable.

You can now use `make` from any terminal (PowerShell, cmd.exe, etc.).

## Opening issues

Use the issue [templates](https://github.com/huggingface/transformers/tree/main/.github/ISSUE_TEMPLATE), and make sure the issue wasn't already reported (use the search bar on GitHub under Issues). A bug report should be about a bug in Transformers, not in your own code, and it should affect real use and real checkpoints; a "theoretical bug" found by reading the code that requires a very nonstandard config to trigger is very low priority to fix. Include a short, self-contained script that reproduces the bug on the latest release or `main`, using a published checkpoint or default config. Bug reports without a runnable reproducer won't be triaged.

If you're unsure whether the bug is in your code or the library, ask in the [forum](https://discuss.huggingface.co/) or on [Discord](https://discord.com/invite/hugging-face-879548962464493619) first. That helps keep GitHub focused on actionable library issues.

## Adding a new model

Adding a model to Transformers makes it available for anyone to load and fine-tune it or run inference. Before you start the implementation, isolate the differences between your architecture and recent state-of-the-art models that are already in Transformers. Explain those differences in the issue or PR so reviewers can quickly understand what challenges the model brings, what parts can reuse existing patterns, and what should be standardized across models.

Follow the guides below.

1. [Add a model with modular Transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/modular_transformers.md) — implement the model using the modular file and generate standalone modeling files.
2. [Auto-generate docstrings](https://github.com/huggingface/transformers/blob/main/docs/source/en/auto_docstring.md) — use the `@auto_docstring` decorator to generate consistent docstrings without boilerplate.
3. [Testing](https://github.com/huggingface/transformers/blob/main/docs/source/en/testing.md) — write and run model tests to verify correctness and keep the contribution maintainable. For causal language models, inherit from the causal LM tester where possible. Prioritize integration tests because they verify the full processor, tokenizer, and model path against real checkpoints.
4. [Model structure rules](https://github.com/huggingface/transformers/blob/main/docs/source/en/modeling_rules.md) — check your files pass the static model structure rules enforced by `make typing`.

Some model additions need extra integration work.

- [Add vision processing components](https://github.com/huggingface/transformers/blob/main/docs/source/en/add_vision_processing_components.md) if your model requires image or video inputs.
- [Dynamic weight loading](https://github.com/huggingface/transformers/blob/main/docs/source/en/weightconverter.md) if published checkpoint parameter names do not match the Transformers implementation.

### Model addition timeline

There are four timelines for model additions depending on the model contributor and community demand for an architecture.

- Day-0 integration: make the model available in Transformers on release day with a new version release. We provide help to support the most important core features (quantization, FlashAttention, KV-cache, etc.), review early drafts, and can also take care of the model implementation depending on timelines.

  Email transformers@huggingface.co a few weeks ahead, especially for novel architectures. We'll iterate with you on a private fork until your checkpoint and release are ready.

- Same-week integration: high-demand models usually land within a week, even when the author doesn't reach out.

  Open an issue with the [new model template](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&projects=&template=new-model-addition.yml) to request one. Issues with more activity move up the queue faster.

- Post-release integration: models without strong demand, or that we don't have bandwidth to take on, land after the upstream release.

  Open ["New model"](https://github.com/huggingface/transformers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+model%22+label%3Acontributions-welcome) issues labeled `contributions-welcome` are the entry point for outside contributors. We'll review and guide you through it.

- Hub-first release: ship your model directly on the Hub via Transformers' [remote-code](https://github.com/huggingface/transformers/blob/main/docs/source/en/models.md#custom-models) support, with no upstream PR required.

  Popular Hub-first models often get integrated into Transformers later, which unlocks first-class docs, maintenance, and optimization. Hub-first is the lowest-friction way to add a model. However, it can be more fragile if a model requires weight conversion or other backward compatibility issues.

## Docs

For API reference generated from source files, use the [`@auto_docstring`](https://github.com/huggingface/transformers/blob/main/docs/source/en/auto_docstring.md) decorator when it applies. Refer to the docs [README](https://github.com/huggingface/transformers/blob/main/docs/README.md) for more details about how to edit the docs and the syntax we use.
