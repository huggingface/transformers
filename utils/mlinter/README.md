# mlinter

Lint modeling, modular, and configuration files under `src/transformers/models` for structural conventions.

## How rule registration works

- Rule metadata lives in `utils/mlinter/rules.toml`.
- Executable TRF rules are auto-discovered from `trf*.py` modules in the `utils/mlinter/` package.
- Each module must define a `check(tree, file_path, source_lines) -> list[Violation]` function.
- The module name determines the rule id: `trf003.py` → `TRF003`.
- A `RULE_ID` module-level constant is set automatically by the discovery mechanism.
- Every discovered rule must have a matching entry in the TOML file, and every TOML rule must have a matching module. Import-time validation fails if either side is missing.
- Suppressions use `# trf-ignore: TRFXXX` on the same line or the line immediately above the flagged construct.

## How to add a new TRF rule

1. Add a `[rules.TRFXXX]` entry to `utils/mlinter/rules.toml`.
2. Fill in `description`, `default_enabled`, `explanation.what_it_does`, `explanation.why_bad`, `explanation.bad_example`, and `explanation.good_example`. Optional model-level exceptions go in `allowlist_models`.
3. Create a new module `utils/mlinter/trfXXX.py` with a `check(tree, file_path, source_lines) -> list[Violation]` function.
4. Use the `RULE_ID` module constant instead of hardcoding `"TRFXXX"` inside the check.
5. Add or update focused tests in `tests/repo_utils/test_check_modeling_structure.py`.

## CLI usage

```bash
# Check all modeling, modular, and configuration files
python -m utils.mlinter

# Only check files changed against a git base ref
python -m utils.mlinter --changed-only --base-ref origin/main

# List all available TRF rules and their default state
python -m utils.mlinter --list-rules

# Show detailed documentation for one rule
python -m utils.mlinter --rule TRF001

# Enable additional rules on top of the defaults
python -m utils.mlinter --enable-rules TRF003

# Enable every TRF rule, including ones disabled by default
python -m utils.mlinter --enable-all-trf-rules

# Emit GitHub Actions error annotations
python -m utils.mlinter --github-annotations
```
