# AGENTS.md — Hugging Face Transformers

## Commands
- `make style` — ruff format + repo formatters (run before finishing)
- `make fix-repo` — auto-fixes copies, modular conversions, doc TOCs, docstrings
- `make check-repo` — CI-style consistency checks
- Slow tests: `RUN_SLOW=1 pytest ...`

## Workflow
- After any code change: `make style` -> targeted `pytest tests/models/<name>/...`
- After touching copied code, modular files, docs, or docstrings: `make fix-repo` instead

## Copies and Modular Models

We try to avoid direct inheritance between model-specific files. We have two mechanisms to manage the resulting code duplication:

1) The older method is to mark classes or functions with `# Copied from ...`. Copies are kept in sync by linters and `make fix-repo`. Do not edit a `# Copied from` block, as it will be reverted by `make fix-repo`. Ideally you should edit the code it's copying from and propagate the change, but you can break the `# Copied from` link if needed.
2) The newer method is to add a file named `modular_<name>.py` in the model directory. `modular` files **can** inherit from other models. `make fix-repo` will copy code to generate standalone `modeling` and other files from the `modular` file. When a `modular` file is present, that file should be edited directly. Changes made to the generated files will be overwritten by `make fix-repo`!

## Adding a Model (modular way)
- Create `src/transformers/models/<name>/modular_<name>.py`, inherit from a similar model, define only what changes.
- `super().__init__(...)` unravels the parent body; use `del self.attr` after to drop unwanted attributes.
- Run `python utils/modular_model_converter.py <name>` to generate files, then `make check-repo`.
- Full guide: [docs/source/en/modular_transformers.md](docs/source/en/modular_transformers.md)

## CI Test Selection
- CI runs only tests impacted by the diff.
- Reproduce locally: `python utils/tests_fetcher.py`, then `python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)`

## Boundaries
- Ask before: adding dependencies, modifying CI, changing public APIs, large refactors.
- Docs: adding a page requires updating `docs/source/en/_toctree.yml` (caught by `make check-repo`).
