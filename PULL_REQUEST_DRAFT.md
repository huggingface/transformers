PR Title: utils: add DCO Signed-off-by checker and unit tests

What changed
- Added `src/transformers/utils/dco_check.py` providing `has_dco_signoff(commit_message: str) -> bool`.
- Added unit tests at `tests/utils/test_dco_check.py` (positive / negative / empty cases).

Why it matters
- Small, dependency-free utility to help contributors/maintainers programmatically verify a Signed-off-by trailer in commit messages (useful for projects that require DCO or when maintainers request sign-offs). Minimal, well-tested addition with no breaking changes.

How to test locally
- python -m venv .venv; .\.venv\Scripts\Activate.ps1
- python -m pip install -e ".[dev]"
- python -m pytest -q tests/utils/test_dco_check.py
- make fixup
- make quality

Related links
- CONTRIBUTING.md in repository root.

Backward compatibility
- No public API changes; new utility only. Fully backward-compatible.

Reviewer checklist
- [ ] Code is small and focused.
- [ ] Tests added and pass locally: `pytest tests/utils/test_dco_check.py`.
- [ ] `make fixup` / `make quality` passes.
- [ ] No runtime import side effects.

Notes
- Replace the commit author/committer details when committing if needed.
