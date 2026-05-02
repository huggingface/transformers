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

# Contribute to Transformers

>[!WARNING]
>The Transformers repo is currently being overwhelmed by a large number of PRs and issue comments written by
>code agents. We are currently bottlenecked by our ability to review and respond to them. As a result, 
>**we ask that new users do not submit pure code agent PRs** at this time. 
>You may use code agents in drafting or to help you diagnose issues. We'd also ask autonomous agents
>not to open any PRs or issues for the moment.
>
>PRs that appear to be fully agent-written will probably be closed without review, and we may block users who do this
>repeatedly or maliciously.

<details>

<summary> Our code agent philosophy in detail </summary>

We understand that code agents are extremely powerful tools, and many people at Hugging Face use them in their work.
However, it's important to realize that **if you simply run a code agent
and generate a PR to an open-source project, you are merely a middleman between the reviewers and the agent**. 
Although doing this creates something that looks very much like a useful contribution, in reality there was no reason 
for you to be involved; the reviewers could have simply run the code agent themselves.

If you want to contribute usefully to open-source in the agent era, **you need to do things that agents can't do on
their own**. In particular, we've found the following to be very helpful:
- Clear diagnosis of bugs. Code agents like to quickly fix problems with a workaround that often causes code bloat
or incompatibilities with other models. Spending time to track down the exact cause of a problem, and in particular
locating the first commit where it appeared (for example with [git bisect](https://git-scm.com/docs/git-bisect)) is valuable.
- Minimize the diff. Check your PR to eliminate any unnecessary changes. Ensure that you did not commit any testing scripts
or unrelated files. Add comments only if they're really necessary; code agents love adding three new functions and
multi-line comments to draw attention to all the hard work they did. If your PR can be a 1-line fix,
make it a 1-line fix. This makes the PR much easier to review and improves the chances that it will be accepted.
- Take the time to reproduce the problem. Very often when a user reports an issue, the issue is actually caused by
environment issues on their machine, or they misdiagnose the problem and suggest an invalid solution. Many code agents
trust the user comments too much, which results in bad solutions, sometimes for problems that 
do not exist! Writing a simple reproducer script and running it to make sure you see the problem is valuable.
- Compare against other models. The Transformers repo is very large, and many models are doing similar things. When
fixing a bug, it's valuable to see if the bug exists in other models. If your PR says
"fixed by using the same approach as (other model)", with a link to the relevant code, that is very helpful for maintainers,
because it tells us that the fix is likely to be correct and compatible with the rest of the codebase. Code agents often
look at the code "narrowly", and make a fix that causes models to diverge from the rest of the codebase.
- Avoid small or "busywork" PRs. In the past, we used to accept these, but given the current flood, we simply don't
have time for small style changes or typo fixes in comments. You can provide value beyond a code
agent simply by having good taste about what's really important.
- Verify tests locally and in the CI. Before opening a PR, run `make fix-repo` and use `utils/tests_fetcher.py` to 
see a list of tests that cover the files you have changed in your PR branch. Run those tests locally, and make sure 
they pass before you open a PR. After you open your PR, please verify that the CI is green and fix any issues before 
pinging anyone for review! This reduces notification spam a lot, which keeps maintainers sane.

Please bear in mind that this is an exciting, rapidly-changing but challenging era for open-source development, and indeed
for the software industry as a whole. We will likely be rapidly updating these guidelines as we learn more about
dealing effectively with code agents. Have patience with us if reviews are slower than normal, or if some
PRs are closed without review!

</details>

Transformers welcomes all contributions whether it is fixing bugs, submitting feature requests, implementing new models, or improving docs.

If you aren't sure where to start, take a look at the [Good First Issues](https://github.com/huggingface/transformers/contribute) for more beginner-friendly issues. For a more challenging issue, check out the [Good Second Issues](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue).

However you choose to contribute, please be mindful and respect our [code of conduct](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

This guide was heavily inspired by the [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).

## Set up

Don't push branches directly to `huggingface/transformers`. Fork [Transformers](https://github.com/huggingface/transformers) with the **Fork** button on GitHub to get a copy under your account, then clone it locally.

```bash
git clone git@github.com:<your Github handle>/transformers.git
cd transformers
```

Add the Transformers repository as a remote named `upstream`. `origin` points at your fork and `upstream` points at the source of truth. That lets you sync your local `main` before pushing to `origin` and opening a PR.

```bash
git remote add upstream https://github.com/huggingface/transformers.git
```

Keep your fork up to date and start a branch to work on.

```bash
git checkout main && git pull upstream main && git push origin main
git switch -c a-descriptive-name-for-my-branch
```

Install the library in editable mode. Use `[torch,testing]` for model contributions (includes PyTorch, pytest, and style tools) or `[quality]` for docs and small fixes (style tools only).

```bash
pip install -e ".[torch,testing]"

# for docs and small fixes only
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

Use the issue [templates](https://github.com/huggingface/transformers/tree/main/.github/ISSUE_TEMPLATE) to help you get started.

### Bug-related issue

Make sure the bug wasn't already reported before opening an issue (use the search bar on GitHub under Issues). The issue should be a bug in Transformers, not in your own code.

Include the following so we can resolve it quickly.

* Your *OS type and version* and *Python*, and *PyTorch* versions when applicable.
* A short, self-contained, code snippet that allows us to reproduce the bug.
* The *full* traceback if an exception is raised.
* Any other information, like screenshots, that may help.

To get the OS and software versions automatically, run the following command.

```bash
transformers env
```

### Feature request

Open an issue and describe:

* The *motivation* such as a frustration with the library, a project need, or work you've done that could benefit the community.
* The feature in as much detail as possible.
* A *code snippet* demonstrating the feature's usage.
* A link to the relevant paper, if applicable.

## Adding a new model

Adding a model to Transformers makes it available for anyone to load and fine-tune it or run inference. Follow the guides below to get started with a model addition.

1. [Add a model with modular Transformers](docs/source/en/modular_transformers.md) — implement the model using the modular file and generate standalone modeling files.
2. [Add vision processing components](docs/source/en/add_vision_processing_components.md) — add image processors and other vision-specific components if your model requires image inputs.
3. [Auto-generate docstrings](docs/source/en/auto_docstring.md) — use the `@auto_docstring` decorator to generate consistent docstrings without boilerplate.
4. [Convert checkpoints](docs/source/en/modular_transformers.md#checkpoint-conversion) — write a conversion script to translate upstream weights into a Transformers-compatible format and upload to the Hub.
5. [Dynamic weight loading](docs/source/en/weightconverter.md) — add a runtime mapping if the published checkpoint layout doesn't match your module's parameter names.
6. [Testing](docs/source/en/testing.md) — write and run model tests to verify correctness and keep the contribution maintainable.
7. [Model structure rules](docs/source/en/modeling_rules.md) — check your files pass the static model structure rules enforced by `make typing`.
8. [Pull request checks](docs/source/en/pr_checks.md) — understand the Hugging Face CI checks and how to run them locally before opening a PR.

### Model addition timeline

There are four timelines for model additions depending on the model contributor and community demand for an architecture.

- Day-0 integration: ship the model in Transformers on release day. We optimize the architecture (quantization, FlashAttention, KV-cache, etc.), review early drafts, and tighten the docs before launch.

  Email transformers@huggingface.co a few weeks ahead, especially for novel architectures. We'll iterate with you on a private fork until your checkpoint and release are ready.

- Same-week integration: high-demand models usually land within a week, even when the author doesn't reach out.

  Open an issue with the [new model template](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&projects=&template=new-model-addition.yml) to request one. Issues with more activity move up the queue faster.

- Post-release integratio*: models without strong demand, or that we don't have bandwidth to take on, land after the upstream release.

  Open issues tagged ["New model"](https://github.com/huggingface/transformers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+model%22) are the entry point for outside contributors. Start with the most-requested architectures to maximize impact. We'll review and guide you through it.

- Hub-first release: ship your model directly on the Hub via Transformers' [remote-code](./models#custom-models) support, with no upstream PR required.

  Popular Hub-first models often get integrated into Transformers later, which unlocks first-class docs, maintenance, and optimization. Hub-first is the lowest-friction way to add a model.

## Docs

Improvements to the docs, like typos, missing content or unclear explanations, are always welcome. For docstrings, use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Open a pull request directly for any documentation fix.

Refer to the docs [README](./docs/README.md) for more details about how to edit the docs and the syntax we use.

## Agentic contributions

AI-assisted contributions are welcome. They must be coordinated, scoped, and verified to keep review load manageable.

- Do not submit "pure agent" PRs. The human submitter is responsible for reviewing all changed lines, validating behavior end-to-end, and running relevant tests.
- If AI tools were used, disclose this in the PR description and include: coordination link, differentiation from existing PRs (if applicable), and test commands/results.
- Avoid one-off "busywork" PRs (single typo, isolated style cleanup, one mutable default fix, etc.). Bundle mechanical cleanups into a clear, systematic scope.
- Coordinate on issues before opening a PR, review similar PRs, and wait for approval. 

> [!NOTE] 
> These topics are outlined for agents in `AGENTS.md` with instruction for how to autonomously implement them. 
