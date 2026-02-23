# AGENTS.md — Hugging Face Transformers

## Commands
- `make style` — ruff format + repo formatters (run before finishing)
- `make fix-repo` — auto-fixes copies, modular conversions, doc TOCs, docstrings
- `make check-repo` — CI-style consistency checks
- Slow tests: `RUN_SLOW=1 pytest ...`

## Workflow
- After any code change: `make style` -> targeted `pytest tests/models/<name>/...`
- After touching copied code, modular files, docs, or docstrings: `make fix-repo` instead

## Copies and Modular Models (critical)
- Files with `# Copied from ...`: edit the source file, then run `make fix-repo` to propagate. Never edit the copy directly.
- Files with `modular_<name>.py`: never edit generated `modeling_*.py` directly. Edit the modular file, then run `make fix-repo` or `python utils/modular_model_converter.py <name>`.
- Generated files are identified by `This file was automatically generated from` on the second line, or by `_pb2.py` suffix.

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
