## Useful commands
- `make style`: runs formatters and linters, necessary to pass code style checks
- `make fix-repo`: auto-fixes copies, modular conversions, doc TOCs, docstrings in addition to the `make style` fixes
- `make check-repo` — CI-style consistency checks
- Many tests are marked as 'slow' and skipped by default in the CI. To run them, use: `RUN_SLOW=1 pytest ...`

`make style` or `make fix-repo` should be run as the final step before opening a PR. The CI will run `make check-repo` and fail if any issues are found.

## Copies and Modular Models

We try to avoid direct inheritance between model-specific files in `src/transformers/models/`. We have two mechanisms to manage the resulting code duplication:

1) The older method is to mark classes or functions with `# Copied from ...`. Copies are kept in sync by `make fix-repo`. Do not edit a `# Copied from` block, as it will be reverted by `make fix-repo`. Ideally you should edit the code it's copying from and propagate the change, but you can break the `# Copied from` link if needed.
2) The newer method is to add a file named `modular_<name>.py` in the model directory. `modular` files **can** inherit from other models. `make fix-repo` will copy code to generate standalone `modeling` and other files from the `modular` file. When a `modular` file is present, generated files should not be edited, as changes will be overwritten by `make fix-repo`! Instead, edit the `modular` file. See [docs/source/en/modular_transformers.md](docs/source/en/modular_transformers.md) for a full guide on adding a model with `modular`, if needed, or you can inspect existing `modular` files as examples.
