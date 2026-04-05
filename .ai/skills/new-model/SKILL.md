---
name: new-model
description: Add a new model to huggingface/transformers using the modular approach. Can fetch models from the Hub, run the modular model detector, auto-generate modular files, then scaffold tests and docs following reviewer-enforced standards.
argument-hint: [model_name] [hub_repo_or_modeling_file_or_parent_model] [checkpoint]
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Agent
effort: max
---

# Add a New Model to HuggingFace Transformers

You are adding a new model called `$0`, with reference checkpoint `$2`.

The second argument `$1` can be one of three things:

- A **Hub repo ID** (e.g., `sarvamai/sarvam-105b`) — the full automated pipeline downloads the modeling file from the Hub, runs the detector, generates the modular file via LLM, and converts it. This is the recommended flow for models with `trust_remote_code=True` on the Hub.
- A **path to a local modeling file** (e.g., `my_beit3_modeling.py` or `src/transformers/models/foo/modeling_foo.py`) — the modular model detector analyzes it to find the best parent model automatically.
- A **parent model name** (e.g., `llama`, `clip`) — used directly as the inheritance target, skipping all detection.

**Detection heuristic:**
- Contains `/` but no `.py` → Hub repo ID (e.g., `org/model-name`)
- Contains `.py` extension → local modeling file path
- Otherwise → parent model name

## Human ownership requirement

The human user is responsible for understanding and defending every line of code produced here. Do NOT generate code the user hasn't asked for or can't explain. When facing non-trivial design decisions, stop and ask the user rather than guessing. During PR review, the user — not an agent — should address reviewer comments. The agent's role is to assist with implementation, not to autonomously handle the full contribution lifecycle.

## Phased approach

Work in phases. Do not try to do everything at once. Focus on one checkpoint, one task at a time.

**Phase 0 — Discovery & scaffolding:** Depending on the input, either run the full automated pipeline (Hub repo), run the detector on a local file, or skip straight to coding with a known parent.

**Phase 1 — Review & refine:** Review the auto-generated modular file (or write it from scratch if parent was given directly). Fix issues, ensure correctness.

**Phase 2 — Conversion script:** Write the weight conversion script. Verify one checkpoint converts correctly.

**Phase 3 — Validation:** Run `make style`, `make typing`, `make check-repo`. Fix all errors.

**Phase 4 — Tests + docs:** Write tests and documentation. Run the test suite.

**Phase 5 — Human review:** The user reviews all code, opens the PR, and handles reviewer feedback themselves.

## Phase 0 — Discovery & Scaffolding

### Path A: Hub repo provided (recommended for `trust_remote_code` models)

The `utils/auto_modular_pr.py` script automates the entire discovery-to-modular-file pipeline:

```bash
python utils/auto_modular_pr.py \
    --hub-repo <org/model-name> \
    --modeling-file <filename_in_repo.py> \
    --model-name $0 \
    --hf-model "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --dry-run
```

This runs 4 steps automatically:
1. **Downloads** the modeling file from the Hub repo
2. **Runs the modular model detector** to find the best parent model and generate per-class inheritance instructions
3. **Calls an LLM** (via HF Inference API) to write the modular file based on detector output
4. **Runs the modular converter** to generate standalone `modeling_*.py` and `configuration_*.py`

The `--dry-run` flag skips the git/PR step — we want to review the output first.

**Before running**, you need to determine the modeling filename in the Hub repo. Check it with:
```bash
# List Python files in the repo
python -c "from huggingface_hub import list_repo_files; print([f for f in list_repo_files('$1') if f.endswith('.py')])"
```

**After running**, the generated files will be in `src/transformers/models/$0/`. Present the results to the user:
- Which parent model was detected
- The per-class inheritance mapping
- The generated modular file for review

The user must approve before proceeding to Phase 1 (refinement).

**Requirements:**
- `HF_TOKEN` env var or `huggingface-cli login` for the Inference API
- The detector needs GPU/significant RAM and downloads an embedding index from the Hub on first run
- Warn the user about resource requirements before running

### Path B: Local modeling file provided

Run the detector directly:

```bash
# With prompt generation — produces per-class inheritance recommendations
python utils/modular_model_detector.py --modeling-file <path_to_modeling_file> --generate-prompt

# Exclude specific models from results
python utils/modular_model_detector.py --modeling-file <path_to_modeling_file> --generate-prompt --ignore-models "bert,gpt2"
```

The detector produces:
1. **Per-class similarity results** — top-N most similar classes in the transformers codebase per source class
2. **Model class match summary** — ranks which existing model covers the most classes
3. **Generated prompt** (with `--generate-prompt`) — per-class "inherit from X" or "copy as-is" instructions, saved to `<model>_MODULAR_PROMPT`

**Present the model class match summary to the user.** Show the top 3-5 candidates. Let the user confirm before proceeding. Then either:
- Run the full pipeline with `auto_modular_pr.py --from-dir` if files are ready, or
- Use the generated prompt to guide manual modular file creation in Phase 1

### Path C: Parent model name provided directly

The user already knows which model to inherit from. Skip detection entirely. Proceed to Phase 1.

## Required reading before writing any code

Read these files in the repo — they are the authoritative reference:

1. **Parent model source:** `src/transformers/models/<parent>/` — read the modular file (if it exists) or modeling file, plus config and tests. The parent is either the detector's recommendation (Phase 0) or `$1` if provided directly.
2. **Modular guide:** `docs/source/en/modular_transformers.md` — how to write modular files, inheritance patterns, `super()` semantics, dependency tracing, the converter
3. **Legacy model guide:** `docs/source/en/add_new_model.md` — file structure, auto registration steps, checkpoint conversion, test conventions
4. **Weight conversion:** `docs/source/en/weightconverter.md` — conversion script patterns
5. **Model linter rules:** `utils/mlinter/rules.toml` — the 13 structural rules (TRF001–TRF013) enforced by `make typing`
6. **PR checks:** `docs/source/en/pr_checks.md` — all CI gates (style, typing, repo consistency)
7. **Design philosophy:** `docs/source/en/philosophy.md` — single model/single file policy, composition over abstraction
8. **Auto docstrings:** `docs/source/en/auto_docstring.md` — `@auto_docstring` decorator usage
9. **Reviewer code quality standards:** [review-standards.md](review-standards.md) — soft standards not in the linter but enforced during review

Also look at a recent similar model as a concrete example — find one by browsing `src/transformers/models/` for a model that inherits from the chosen parent or a nearby architecture.

## Critical design decisions (decide BEFORE writing code)

### Standalone config vs. inheriting from parent config

**Prefer standalone config** (`MyConfig(PreTrainedConfig)`) over inheriting (`MyConfig(ParentConfig)`) for composite models where only one sub-component changes. Inheriting from a composite parent config causes the modular converter to:
- Rename ALL sub-configs (e.g., `ParentVisionConfig` → `MyVisionConfig`), each with a new `model_type`
- Require registering every renamed `model_type` in CONFIG_MAPPING
- Break `sub_configs` dict serialization when the text/vision config type changes
- Generate function-level imports that trigger TRF009

A standalone config with explicit `sub_configs` using `AutoConfig` for shared components avoids all of this. Example:

```python
class MyConfig(PreTrainedConfig):
    model_type = "my_model"
    sub_configs = {
        "vision_config": AutoConfig,          # resolves via model_type in JSON
        "text_config": MyNewTextConfig,        # your new config class
        "decoder_config": AutoConfig,          # reuses parent's decoder config
    }
```

### Reuse existing attention/encoder layers

For new sub-encoders (e.g., a replacement text encoder), **inherit from existing transformers building blocks** like `SiglipAttention`, `SiglipEncoderLayer`, `SiglipMLP`, or `CLIPAttention` rather than writing custom attention from scratch. Benefits:
- Free SDPA / FlashAttention / FlexAttention support through existing infrastructure
- No need to skip 25+ parameterized SDPA test variants
- `@capture_outputs` and `_can_record_outputs` work automatically for hidden states and attentions

Only write custom attention if the architecture genuinely cannot be expressed through existing layers (e.g., RepMixer conv-based token mixing).

### Use `@capture_outputs` decorator

The modern pattern for models that produce multiple output types (hidden states, attentions, masks, boxes):

```python
class MyModel(MyPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": MyEncoderLayer,
        "attentions": MyAttention,
    }

    @capture_outputs
    def forward(self, ...):
        ...
```

This eliminates manual hidden state/attention collection and makes `test_training`, `test_hidden_states_output`, and gradient checkpointing tests pass without overrides.

### Conditional layers via config flags

If a component (e.g., RepMixer blocks) is architecturally incompatible with standard attention tests, add a config flag to disable it:

```python
class MyTextConfig(PreTrainedConfig):
    use_repmixer_blocks: bool = True  # set False in tests for SDPA compat
```

### Simplify inference-only optimizations

Do NOT implement reparameterization (`reparameterize()` methods that fuse multi-branch convolutions) unless specifically needed for the checkpoint format. If the checkpoint stores unfused weights, the HF model should match that structure. Reparameterization adds complexity without benefit for standard HF usage.

## Modular converter pitfalls

The modular converter (`utils/modular_model_converter.py`) has limitations. These apply when you DO inherit from a parent model's classes:

### Function-level imports for cross-model class references

The converter only traces class references at **class-level attributes and inheritance**. Classes referenced only inside method bodies (e.g., `ParentVisionModel` used in `__init__`) will NOT be imported in the generated file. Fix: use **function-level imports** inside the method body:

```python
def __init__(self, config):
    from ..parent_model.modeling_parent import ParentVisionModel, ParentEncoder
    self.vision_encoder = ParentVisionModel(config.vision_config)
```

This triggers TRF009 (cross-model imports). Add the model to `utils/mlinter/rules.toml` allowlist for TRF009.

### sub_configs dict and changed sub-config types

If you replace a sub-config type (e.g., `CLIPTextConfig` → `NewTextConfig`), the parent's `sub_configs` dict is copied verbatim with the OLD type. This breaks config save/load roundtrips because deserialization creates the wrong type.

Fixes:
- Override `__post_init__` to check `isinstance` and convert: if the loaded config is NOT your new type, create one from `config.to_dict()`.
- If the sub-config is a new type you defined, register its `model_type` in `CONFIG_MAPPING_NAMES` and `SUBCONFIG_TO_MODEL_TYPE_MAP`.

### _init_weights for nn.Parameter attributes

If your model adds `nn.Parameter` attributes (e.g., `layer_scale`, positional embeddings), you MUST handle them in `_init_weights` of **both**:
1. The sub-model's PreTrainedModel (e.g., the text encoder) — called during its own `post_init()`
2. The parent PreTrainedModel — called during the top-level model's `post_init()`

Missing either causes `test_can_init_all_missing_weights` to fail. Also: do NOT initialize `nn.Parameter` with random values in `__init__` AND again in `_init_weights` — the double initialization consumes random state differently on meta vs CPU, causing the test to fail. Initialize ONLY in `_init_weights`.

## Auto registration checklist

Add entries **alphabetically** in ALL of these locations:

- [ ] `src/transformers/models/__init__.py` — add `from .$0 import *`
- [ ] `src/transformers/models/auto/configuration_auto.py`:
  - `CONFIG_MAPPING_NAMES` — main model_type AND all sub-config model_types
  - `MODEL_NAMES_MAPPING` — human-readable name for each model_type
  - `SUBCONFIG_TO_MODEL_TYPE_MAP` — map sub-config model_types to parent model_type
- [ ] `src/transformers/models/auto/modeling_auto.py`:
  - `MODEL_MAPPING_NAMES` — main model AND any sub-models needed by `AutoModel.from_config()`
- [ ] `src/transformers/models/auto/image_processing_auto.py` — if applicable
- [ ] `src/transformers/models/auto/processing_auto.py` — if applicable
- [ ] `src/transformers/models/auto/tokenization_auto.py` — if applicable
- [ ] `utils/check_repo.py` `IGNORE_NON_TESTED` — for building-block sub-models (e.g., ViTModel)
- [ ] `utils/mlinter/rules.toml` TRF009 allowlist — only if cross-model imports are needed

## Step-by-step process

### 0. Run Phase 0 — Discovery & scaffolding

Follow the appropriate path (A/B/C) from the Phase 0 section above. After this step you should have:
- A confirmed `PARENT_MODEL` (the best parent to inherit from)
- Optionally, an auto-generated modular file in `src/transformers/models/$0/`
- Optionally, a `<model>_MODULAR_PROMPT` file with per-class instructions

### 1. Review and understand the parent model

Read from `src/transformers/models/<PARENT_MODEL>/`:
- `modular_<PARENT_MODEL>.py` (preferred) or `modeling_<PARENT_MODEL>.py`
- `configuration_<PARENT_MODEL>.py`
- `tests/models/<PARENT_MODEL>/test_modeling_<PARENT_MODEL>.py`

If the detector/pipeline was run, use its per-class breakdown to understand exactly which components the new model reuses vs. replaces.

### 2. Create or refine the modular file (SOURCE OF TRUTH)

**If Phase 0 Path A was used (Hub pipeline):** A modular file was auto-generated by the LLM. Read it carefully, compare against the source modeling file, and fix any issues:
- Incorrect inheritance (wrong parent class for a component)
- Missing overrides (differences between new and parent not captured)
- Extra overrides (code identical to parent that should be removed)
- Import errors or missing dependencies
- Style violations (see reviewer standards below)

**If Phase 0 Path B was used (detector only):** Read the generated `<model>_MODULAR_PROMPT` file and use its per-class inheritance instructions as the blueprint to write the modular file:
- Classes marked "inherit from X" → create a subclass that only overrides what differs
- Classes marked "copy as-is" → reproduce them from the source file without inheriting
- If classes match different parent models, the modular file can import from multiple parents

**If Phase 0 Path C was used (parent given directly):** Write the modular file from scratch following the patterns in `docs/source/en/modular_transformers.md` and using a sibling model's modular file as a template.

The modular file at `src/transformers/models/$0/modular_$0.py` is the ONLY modeling file you write. The converter generates `modeling_$0.py` and `configuration_$0.py`.

The modular file must pass the reviewer standards in [review-standards.md](review-standards.md). Key rules:
- `nn.ModuleList` not `nn.Sequential` for layer lists
- `nn.Linear` for projections, not `nn.Parameter(torch.empty(...))`
- Inherit from existing components when possible (`SiglipAttention`, `SiglipEncoderLayer`, `CLIPMLP`, etc.)
- Make all magic numbers into config attributes
- Only override PreTrainedModel attributes that actually differ from defaults
- Data transforms (permute, reshape) go inside layer forward methods, not parent loops
- `nn.Identity` ternaries for conditional layers
- Descriptive names, not opaque abbreviations from original codebases
- Use `@capture_outputs` and `_can_record_outputs` for output collection

### 3. Create the __init__.py

Create `src/transformers/models/$0/__init__.py` — use any sibling model's `__init__.py` as a template, it's boilerplate with lazy loading.

### 4. Register in auto mappings

Follow the auto registration checklist above. This is error-prone — missing a single registration causes cryptic runtime failures. Do all registrations before running the converter.

### 5. Write conversion script and verify one checkpoint

Create `src/transformers/models/$0/convert_$0_to_hf.py`. See `docs/source/en/weightconverter.md` for patterns and the parent model's conversion script for a concrete example.

Verify by comparing forward pass outputs between original and HF implementation on the same dummy inputs. Focus on converting one checkpoint successfully before attempting others.

### 6. Generate standalone files, smoke test, and lint

```bash
python utils/modular_model_converter.py $0
```

If it fails, fix the modular file. Never hand-edit generated files.

**Smoke test immediately** — catch registration and import issues before proceeding:

```python
from transformers import $0Config, $0Model
config = $0Config()
model = $0Model(config)
print(f"OK: {sum(p.numel() for p in model.parameters()):,} params")
```

If this crashes, fix before continuing. Common failures:
- `KeyError` in CONFIG_MAPPING → missing sub-config registration
- `ImportError` → converter didn't include a class; use function-level imports
- `ValueError: Unrecognized configuration class` → sub-config model_type not in MODEL_MAPPING

Then run the model structure linter:
```bash
make typing
```

This runs `utils/check_modeling_structure.py` → `utils/mlinter/mlinter.py` which enforces the 13 rules in `utils/mlinter/rules.toml` (TRF001–TRF013). Fix all errors before proceeding — this is a CI gate.

### 7. Write tests

Create `tests/models/$0/__init__.py` (empty) and `tests/models/$0/test_modeling_$0.py`. Use the parent's test file as a template.

Test rules:
- `@require_torch` on test classes
- Exact expected values in integration tests (no TODOs)
- Set `_supports_flash_attn = False` in model class instead of skipping attention tests
- `gc.collect()` + `backend_empty_cache` in integration test tearDown
- Small configs in unit tests (hidden_size=32, num_layers=1-2)
- For composite models with conditional layers (e.g., RepMixer), consider disabling them in tests via a config flag for SDPA/attention compatibility

Test overrides commonly needed for composite models:
- `test_hidden_states_output` — if outputs use component-specific fields (e.g., `vision_hidden_states`) instead of generic `hidden_states`. Using `@capture_outputs` avoids this.
- `test_training` / `test_training_gradient_checkpointing` — may fail if model uses component-specific output structure. Using `@capture_outputs` avoids this.
- `test_eager_matches_sdpa_inference` — parameterized test generating 25+ variants; if the new sub-model doesn't support SDPA, override with `self.skipTest(...)` using `*args, **kwargs` signature, PLUS explicit skips for each numbered variant. Reusing existing attention layers (e.g., `SiglipAttention`) avoids this entirely.
- `test_config` — if `sub_configs` has wrong types, skip `create_and_test_config_from_and_save_pretrained_composite` and run individual config tests instead. Standalone config avoids this.

### 8. Write documentation

Create `docs/source/en/model_doc/$0.md` and add entry to `docs/source/en/_toctree.yml`.

Doc rules:
- Normal Python scripts, not notebook style
- `device_map="auto"`, not manual CUDA checks
- Auto classes in examples (`AutoModel`, `AutoProcessor`)

After creating the doc file, run:
```bash
python utils/check_doc_toc.py --fix_and_overwrite   # fix toctree ordering
python utils/add_dates.py --models $0               # add model dates
```

### 9. Final validation

Run in order — each is a CI gate:

```bash
make style                                          # ruff formatting
make fix-repo                                       # regenerate from modular + fix copies/docstrings/TOCs
make typing                                         # model structure linter (mlinter TRF001-TRF013) + type checker
make check-repo                                     # auto mapping consistency + repo-wide checks
pytest tests/models/$0/ -v                          # unit tests
RUN_SLOW=1 pytest tests/models/$0/ -v -k "Integration"  # integration tests (if weights available)
```

### 10. Pre-PR checklist (present to user before they open the PR)

- [ ] `modular_$0.py` is the only hand-written modeling file
- [ ] Generated files are up to date (re-run converter if unsure)
- [ ] All auto registrations are alphabetically placed (see checklist above)
- [ ] Tests pass with small configs
- [ ] Passes [review-standards.md](review-standards.md) (no nn.Sequential, no nn.Parameter projections, no hardcoded constants)
- [ ] Documentation uses Auto classes and `device_map="auto"`
- [ ] `make style` passes clean
- [ ] `make typing` passes clean
- [ ] `make check-repo` passes clean
- [ ] User has reviewed every changed line and can defend the design decisions
- [ ] PR description includes: issue link, duplicate check, test results, AI disclosure
