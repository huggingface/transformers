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

When you open a pull request, the Hugging Face CI runs several checks that must pass before your PR can be merged. This page explains each check and how to pass it locally before pushing.

Before you push, run these three commands in order:

```bash
make fix-repo   # auto-fix everything that can be auto-fixed
make typing     # check types and model structure, fix any errors manually
make check-repo # verify all checks pass, fix anything that remains
```

`make fix-repo` handles most issues automatically (style, copies, docstrings, auto-generated files). `make typing` catches type errors and model structure violations that require manual fixes. `make check-repo` does a final read-only pass over everything so you can confirm nothing is left.

## Code quality

The code quality check covers formatting, imports, type checking, and model structure rules. It corresponds to `make fix-repo` and `make typing`.

`make style` (included in `make fix-repo`) auto-fixes [Ruff](https://docs.astral.sh/ruff/) linting and formatting, `__init__.py` import sort order, and auto-mapping consistency.

`make typing` performs type checking with [ty](https://docs.astral.sh/ty/) and validates model structure rules, which cover config class naming conventions and `forward()` signatures. Type errors and TRF violations are reported with specific rule numbers and must be fixed manually. Use `mlinter --list-rules` to see all available TRF rules and `mlinter --rule TRFXXX` to view the full documentation for a specific rule.

## Repository consistency

The repository consistency check is similar to `make check-repo`, except it stops on the first failure. It ensures the repository stays internally consistent across the categories below. For new models, it also verifies that all new model classes are registered in the auto-mappings.

| Category | What it validates | Auto-fixed? |
|---|---|---|
| Init files | Every new public object must appear in both `_import_structure` (lazy loading) and the `if TYPE_CHECKING` block (type checker imports) in `__init__.py` | Manual |
| Copies and modular | `# Copied from` blocks match their source and modular-generated files are up to date | `make fix-repo` |
| Docstrings and docs | Argument docstrings match function signatures and documentation table of contents | `make fix-repo` |
| Auto-generated files | Dummies, pipeline typing, doctest list, metadata, dependency table | `make fix-repo` |
| Config validation | Config classes have valid checkpoints in docstrings and config attributes match modeling file | Manual |

## Tests

CI runs a targeted subset of tests based on what your PR changes. The sections below explain how test selection works, which jobs run, and how to handle slow tests.

### Test selection

CI doesn't run the full test suite on every PR. `utils/tests_fetcher.py` traces import dependencies from your changed files to identify affected tests and only runs those. It also catches regressions in other models if you touched shared utilities. The fetcher prints which files changed, which tests are impacted, and writes the list to `tests_torch_test_list.txt`.

Use the fetcher to replicate exactly what CI runs.

```bash
python utils/tests_fetcher.py
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_preparation/tests_torch_test_list.txt)
```

> [!TIP]
> Changing core files like `modeling_utils.py` or `generation/utils.py` triggers all model tests, not just the affected subset.

You could also run every test for your model unconditionally. This is a faster local sanity check, but it won't catch regressions in other models caused by shared code you touched.

```bash
pytest tests/models/my_model/ -v
```

### Test job categories

Tests are distributed across parallel CI jobs. Your `test_modeling_my_model.py` is detected by its path pattern. The relevant jobs for a model PR are:

- `tests_torch` — modeling tests (`tests/models/*/test_modeling_*.py`)
- `tests_tokenization` — tokenizer tests (`tests/models/*/test_tokenization_*.py`)
- `tests_processors` — processor and feature extractor tests
- `tests_generate` — generation tests
- `pipelines_torch` — pipeline tests
- `tests_training_ci` — training loop tests
- `tests_tensor_parallel_ci` — tensor parallel tests

### Slow tests

Tests decorated with `@slow` are skipped in regular CI runs. They require a real checkpoint download or significant compute, so they run on GPU instances and are triggered by maintainers once your PR is under review.

Run slow tests locally with the command below.

```bash
RUN_SLOW=1 python -m pytest tests/models/my_model/ -v
```

## Documentation build

A `build_pr_documentation` job builds and generates a preview of the documentation. A bot posts a preview link in your PR, and the check must pass before merging. Most failures are a missing entry in the `toctree`. To build the documentation locally, see the [`README.md`](https://github.com/huggingface/transformers/tree/main/docs) in the docs folder.

## `# Copied from` syntax

> [!TIP]
> For new models, the [modular workflow](modular_transformers) (`modular_*.py`) is preferred over `# Copied from`. Use `# Copied from` when the modular approach doesn't apply.

The `# Copied from` mechanism keeps copied code in sync with its source. When `make fix-repo` runs, it checks every `# Copied from` block and updates it to match the original. This means edits to a `# Copied from` block will 
be overwritten. Edit the source instead and let `make fix-repo` propagate the change.

The basic forms of `# Copied from` include the following.

```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput

# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta

# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

The `with model->newModel` syntax applies string replacements after copying. Multiple replacements are comma-separated and applied left to right. The `all-casing` option replaces all casing variants simultaneously (`Bert`, `bert`, `BERT` → `MobileBert`, `mobilebert`, `MOBILEBERT`).
