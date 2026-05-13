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

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Pull request checks

When you open a pull request, the Hugging Face CI runs several checks that must pass before your PR can be merged.

- [Fixing the CI](#fixing-the-ci) lists the commands to run locally to pass the checks.
- The sections after that describe what each check validates, such as [code quality](#code-quality) and [repository consistency](#repository-consistency).
- [Tests](#tests) describe which tests are run, test categories, and slow tests.

## Fixing the CI

In most cases, `make style` is enough to clear the code quality check, which is the most common failure.

```bash
make style
```

For failures in repository consistency, copies, or auto-generated files, run `make fix-repo`. It fixes style, copies, docstrings, and auto-generated files in one pass.

```bash
make fix-repo
```

For a larger change, the three-command sequence below covers every check. It is heavier, but catches everything before you push.

```bash
make fix-repo   # auto-fix everything that can be auto-fixed
make typing     # check types and model structure, fix any errors manually
make check-repo # verify all checks pass, fix anything that remains
```

`make typing` catches type errors and model structure violations that you fix manually. `make check-repo` does a final read-only pass so you can confirm everything is ready.

> [!NOTE]
> The CI is occasionally flaky. If a check fails on something unrelated to your change, ping a maintainer to re-run it.

## Code quality

The code quality check covers formatting, imports, type checking, and model structure rules. It corresponds to `make fix-repo` and `make typing`.

`make style` (included in `make fix-repo`) auto-fixes [Ruff](https://docs.astral.sh/ruff/) linting and formatting, `__init__.py` import sort order, and auto-mapping consistency.

`make typing` performs type checking with [ty](https://docs.astral.sh/ty/) and validates TRansFormers (TRF) rules, which cover config class naming conventions and `forward()` signatures. Type errors and TRF violations report a specific rule number and must be fixed manually. The rules live in the [mlinter](https://github.com/huggingface/transformers-mlinter) repository. Run `python -m utils.mlinter --list-rules` to see every TRF rule, or `python -m utils.mlinter --rule TRFXXX` to view the full documentation for a specific rule.

If a TRF rule needs an exception, choose one of these options (see [Suppressing violations](./modeling_rules#suppressing-violations) for more details).

- Add your model name to the `allowlist_models` list for the relevant rule in `utils/mlinter/rules.toml`. Use this when the whole model file needs an exception.
- Add `# trf-ignore: TRFXXX` on the same line as the flagged construct, or on the line immediately above it. Use this when only one flagged construct needs an exception.

## Repository consistency

The repository consistency check is similar to `make check-repo`, except it stops on the first failure. It keeps the repository internally consistent across the categories below: public objects stay importable, copied code stays in sync with its source, and auto-generated files (dummies, doctests, metadata) reflect the current state of the code. For new models, it also verifies every new model class is registered in the auto-mappings.

| Category | What it validates | Auto-fixed? |
|---|---|---|
| Init files | Every new public object must appear in both `_import_structure` (lazy loading) and the `if TYPE_CHECKING` block (type checker imports) in `__init__.py` | Manual |
| Copies and modular | `# Copied from` blocks match their source and modular-generated files are up to date | `make fix-repo` |
| Docstrings and docs | Argument docstrings match function signatures and documentation table of contents | `make fix-repo` |
| Auto-generated files | Dummies, pipeline typing, doctest list, metadata, dependency table | `make fix-repo` |
| Config validation | Config classes have valid checkpoints in docstrings and config attributes match modeling file | Manual |

## Tests

CI runs a targeted subset of tests based on what your PR changes. The CI runs tests in a slightly randomized order with [pytest-random-order](https://github.com/jbasko/pytest-random-order) to catch coupled tests. The run prints the random seed at the start so you can replay the same order with `--random-order-seed=<seed>`.

If a test passes locally on GPU but fails in CI, set `TRANSFORMERS_TEST_DEVICE="cpu"` to check whether you can reproduce the failure on CPU.

```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/models/my_model/ -v
```

The sections below explain how test selection works, which jobs run, and how to handle slow tests.

### Test selection

CI doesn't run the full test suite on every PR. CI skips tests decorated with `@slow` on every PR, and a maintainer triggers them on GPU once the PR is under review.

`utils/tests_fetcher.py` traces import dependencies from your changed files to identify affected tests, and only runs those. It also catches regressions in other models when you touch shared utilities. The fetcher prints which files changed and which tests are impacted, then writes the list to `tests_torch_test_list.txt`.

Use the fetcher to replicate exactly what CI runs.

```bash
python utils/tests_fetcher.py
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_preparation/tests_torch_test_list.txt)
```

> [!TIP]
> Changing core files like `modeling_utils.py` or `generation/utils.py` triggers all model tests, not just the affected subset.

You can also run every test for your model unconditionally. A direct run is a faster local sanity check, but it won't catch regressions in other models caused by shared code you touched.

```bash
pytest tests/models/my_model/ -v
```

### Test job categories

Tests are split across parallel CI jobs, and each job picks up files by path pattern. The relevant jobs for a model PR are:

- `tests_torch`: modeling tests (`tests/models/*/test_modeling_*.py`)
- `tests_tokenization`: tokenizer tests (`tests/models/*/test_tokenization_*.py`)
- `tests_processors`: processor and feature extractor tests (`tests/models/*/test_(processing|image_processing|feature_extractor)_*.py`)
- `tests_generate`: generation tests
- `pipelines_torch`: pipeline tests
- `tests_training_ci`: training loop tests
- `tests_tensor_parallel_ci`: tensor parallel tests

### Slow tests

Regular CI runs skip tests decorated with `@slow`. They download real checkpoints or need significant compute, so they run on GPU instances, and maintainers trigger them once your PR is under review.

Slow tests run on an NVIDIA A10, and numerical results can vary slightly between the CI hardware and your local machine. Maintainers usually adjusts those values on tests if needed when adding a new model.

Run slow tests locally with the command below.

```bash
RUN_SLOW=1 python -m pytest tests/models/my_model/ -v
```

### Documentation build

A `build_pr_documentation` job builds and generates a preview of the documentation. A bot posts a preview link in your PR, and the check must pass before merging. Most failures are a missing entry in the `toctree`. To build the documentation locally, see the [README.md](https://github.com/huggingface/transformers/tree/main/docs) in the docs folder.

## `# Copied from` syntax

> [!WARNING]
> For new models, always prefer the [modular workflow](modular_transformers) (`modular_*.py`) over `# Copied from`. Avoid `# Copied from` whenever possible.

The `# Copied from` mechanism keeps copied code in sync with its source. When `make fix-repo` runs, it checks every `# Copied from` block and updates it to match the original, so edits inside a `# Copied from` block are overwritten. Edit the source instead and let `make fix-repo` propagate the change.

The basic forms of `# Copied from` include the following.

```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput

# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta

# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

The `with model->newModel` syntax applies string replacements after copying. Separate multiple replacements with commas, which are applied from left to right. The `all-casing` option replaces every casing variant at once (`Bert`, `bert`, `BERT` become `MobileBert`, `mobilebert`, `MOBILEBERT`).
