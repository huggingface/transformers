---
name: add-mlinter-rule
description: Add a new TRF rule to the mlinter. Checks for duplicates, creates the rule module and TOML entry, runs against all models, and handles violations (fix or allowlist).
---

# Add Mlinter Rule

## Input

- `<description>`: Natural-language description of what the rule should detect.
- Optional: specific AST pattern or code example showing the bad/good pattern.

## Constraints

- Rules MUST use static analysis only (Python `ast` module). NEVER import runtime libraries like `torch`, `tensorflow`, etc.
- Rules MUST follow the `check(tree, file_path, source_lines) -> list[Violation]` interface.
- Use `RULE_ID` module constant (set automatically by discovery) instead of hardcoding the rule ID string.

## Workflow

1. **Check for duplicate rules** in `utils/mlinter/rules.toml`:
   - Read the full TOML file and review all existing rule descriptions and explanations.
   - If an existing rule already covers the same concern (even partially), stop and ask the user whether to proceed, extend the existing rule, or abort.

2. **Determine the next rule number**:
   - List all `utils/mlinter/trf*.py` files and find the highest number.
   - The new rule gets that number + 1, zero-padded to 3 digits (e.g., `TRF014`).

3. **Add the TOML entry** to `utils/mlinter/rules.toml`:
   - Append a new `[rules.TRFXXX]` section at the end of the file with:
     - `description` - one-line summary
     - `default_enabled = true`
     - `allowlist_models = []`
     - `[rules.TRFXXX.explanation]` with `what_it_does`, `why_bad`, `bad_example`, `good_example`
   - Follow the exact formatting style of existing entries.

4. **Create the rule module** at `utils/mlinter/trfXXX.py`:
   - Start with the Apache 2.0 license header (copy from any existing `trf*.py`).
   - Add a module docstring: `"""TRFXXX: <short description>."""`
   - Import `ast`, `Path`, and needed helpers from `._helpers`.
   - Define `RULE_ID = ""  # Set by discovery`.
   - Implement `def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:`.
   - Refer to existing rules in `utils/mlinter/trf*.py` for patterns and helpers.

5. **Run the rule against all models**:
   ```bash
   python -m utils.mlinter --enable-rules TRFXXX
   ```
   - If the run itself errors (import error, crash), fix the rule code and re-run.

6. **Handle violations**:
   - Present the list of violations to the user.
   - Ask: "Should I fix these models, or add them to `allowlist_models` in rules.toml?"
   - If **fix**: apply the fixes to each violating model file, then re-run the rule to confirm zero violations.
   - If **allowlist**: extract the model directory names from the violation file paths and add them to the `allowlist_models` list in the TOML entry.
   - The user may choose a mix (fix some, allowlist others).

7. **Add tests** in `tests/repo_utils/test_mlinter.py`:
   - Add at least one positive test (valid code, no violations) and one negative test (bad code, expected violation).
   - Follow the pattern of existing tests: create source strings, call `mlinter.analyze_file()`, and assert on violations.
   - For cross-file rules (rules that read config or other files from disk), use `tempfile.TemporaryDirectory` to create real file structures. The test file already imports `tempfile`.
   - If the rule maps a modeling class to a specific config class, add a regression where another config class in the same file would otherwise cause a false positive/false negative.
   - Run the tests:
     ```bash
     python -m pytest tests/repo_utils/test_mlinter.py -x -v -k "trfXXX"
     ```

8. **Final validation**:
   ```bash
   make style
   make check-repo
   ```

## Model architecture knowledge

The mlinter processes files **one at a time** via `analyze_file(file_path, text, enabled_rules)`. When a rule needs cross-file information, the rule module must read the other file from disk. Be aware of these patterns:

### Multi-config directories
Some model directories contain multiple configuration files (e.g., `data2vec/` has `configuration_data2vec_audio.py`, `configuration_data2vec_text.py`, `configuration_data2vec_vision.py`). When finding a config file for a modeling file, **match by suffix first**: `modeling_foo_text.py` -> `configuration_foo_text.py`. Only fall back to picking the first config file if there's a single one or no suffix match. See `trf014.py:_find_config_file()` for the pattern.

### Multi-class configuration files
A single `configuration_*.py` file can define multiple config classes (e.g., a main config plus text/vision sub-configs). If the rule is checking a property that should belong to one specific config class, **do not scan the file and accept the first matching class**. First resolve the modeling class's target config class:

- Prefer `config_class` from the model class, following local modeling inheritance if it is declared on a parent `*PreTrainedModel`.
- If there is no explicit `config_class`, infer the best match from class names, typically by longest shared prefix (`FooTextForCausalLM` -> `FooTextConfig`, not `FooConfig`).

Then validate only that config class. This avoids early-return bugs where an unrelated sub-config masks a missing field on the actual target config.

### Inherited configs
Some config classes inherit from another model's config rather than directly from `PreTrainedConfig` (e.g., `VoxtralRealtimeTextConfig(MistralConfig)`). These inherit fields like `tie_word_embeddings` from their parent. When checking for a field in a config class, **if the base class is not `PreTrainedConfig`/`PretrainedConfig` and ends with `Config`**, assume the field may be inherited and skip the violation.

### Composite models (vision-language, audio-video, etc.)
Models like `janus`, `perception_lm`, `pe_audio_video` use composite configs with `sub_configs = {"text_config": AutoConfig, "vision_config": ...}`. Text-related fields (like `tie_word_embeddings`) live in the text sub-config (e.g., LlamaConfig), not in the composite config itself. When checking for a text-related field, **if the config class has a `sub_configs` dict containing `"text_config"`**, the field is delegated to the sub-config and should not be flagged.

### `tie_word_embeddings` is NOT in `PreTrainedConfig`
The base `PreTrainedConfig` in `src/transformers/configuration_utils.py` does **not** define `tie_word_embeddings`. Each model config must define it explicitly (as a class attribute like `tie_word_embeddings: bool = True`, or via `self.tie_word_embeddings = ...` in `__init__`/`__post_init__`).

## Reference

- Rule modules: `utils/mlinter/trf*.py`
- Rule config: `utils/mlinter/rules.toml`
- Helpers: `utils/mlinter/_helpers.py` (Violation, iter_pretrained_classes, _has_rule_suppression, _class_methods, _get_class_assignments, full_name, _simple_name, etc.)
- Tests: `tests/repo_utils/test_mlinter.py`
- README: `utils/mlinter/README.md`
