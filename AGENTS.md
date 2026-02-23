## Useful commands
- `make style`: runs formatters, linters and type checking, necessary to pass code style checks
- `make fix-repo`: auto-fixes copies, modular conversions, doc TOCs, docstrings in addition to the `make style` fixes
- `make check-repo` — CI-style consistency checks
- Many tests are marked as 'slow' and skipped by default in the CI. To run them, use: `RUN_SLOW=1 pytest ...`

`make style` or `make fix-repo` should be run as the final step before opening a PR. The CI will run `make check-repo` and fail if any issues are found.
