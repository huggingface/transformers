# Transformers `src/transformers/` module tree (curated) — **v4.57.6**

> **Purpose**: Fast repo navigation for Transformers API without guessing.
> **Pinned revision (current)**: `transformers==4.57.6` (PyPI release: **2026-01-16**).
> **Design goal**: 
> - Prefer **patterns + canonical entry points + grep keywords** over enumerating every file.
> - Treat this as **generated**: pin a Transformers revision (tag/commit or exact PyPI version) and regenerate on upgrades.
> **Not exhaustive**: For model-specific code, use the `models/<model_name>/` patterns and grep tips.

---

## How to use this file

1. Pick the **surface area** below (Loading, Preprocessing, Generation, Pipelines, Training, Integrations/Quantization, Export/ONNX, CLI).
2. Jump to the **canonical entry point(s)** and search there.
3. If you need the exact implementation:
   - `git grep -n "<keyword>" src/transformers` (keywords provided per area)
   - follow imports into submodules

---

## Core package entry points

```

src/transformers/
**init**.py
dependency_versions_check.py
dependency_versions_table.py

```

- `__init__.py` is the public import surface (re-exports / lazy-import wiring for `from transformers import X`).
- `dependency_versions_check.py` is where import-time version guards often trigger.

Grep keywords:
- `_LazyModule`
- `dependency_versions_check`
- `require_version`

---

## Configuration, modeling, and loading (PyTorch)

Canonical entry points:

```

src/transformers/
configuration_utils.py
modeling_utils.py
pytorch_utils.py
modeling_outputs.py
modeling_layers.py

```

Primary responsibilities:
- `PreTrainedConfig` (config serialization, `from_pretrained` for configs, validation helpers)
- `PreTrainedModel` (weight loading/saving, `from_pretrained` for models, sharding, tying weights)
- torch helpers + shared model output dataclasses/layers

Grep keywords:
- `from_pretrained(`
- `save_pretrained(`
- `get_checkpoint_shard_files`
- `tie_weights`
- `state_dict`

Related (often on stack traces):

```

src/transformers/
dynamic_module_utils.py

src/transformers/utils/
hub.py
import_utils.py

```

- `dynamic_module_utils.py` is where `trust_remote_code` plumbing typically lands.
- `utils/hub.py` is where Hub/caching helpers like `cached_file` and shard resolution live.
- `utils/import_utils.py` is lazy-import + optional dependency gating.

---

## Tokenization and preprocessing (text / vision / audio / video / multimodal)

### Tokenization (slow + fast)

Canonical entry points:

```

src/transformers/
tokenization_utils_base.py
tokenization_utils.py
tokenization_utils_fast.py
tokenization_mistral_common.py

```

Notes:
- Slow (Python) tokenizers: `tokenization_utils.py`
- Fast tokenizers (Rust `tokenizers` wrappers): `tokenization_utils_fast.py`
- Shared bases: `tokenization_utils_base.py`
- Newer/common helpers for Mistral ecosystem: `tokenization_mistral_common.py`

Grep keywords:
- `PreTrainedTokenizerBase`
- `BatchEncoding`
- `AutoTokenizer`
- `TokenizerFast`
- `convert_tokens_to_ids`

Related conversion helpers:

```

src/transformers/
convert_slow_tokenizer.py
convert_slow_tokenizers_checkpoints_to_fast.py
convert_slow_tokenizers_checkpoints_to_fast.py

```

Grep keywords:
- `convert_slow_tokenizer`
- `SpmConverter`
- `sentencepiece`

### Processors / image / feature extraction / audio / video

Canonical entry points:

```

src/transformers/
processing_utils.py
feature_extraction_utils.py
feature_extraction_sequence_utils.py
image_processing_base.py
image_processing_utils.py
image_processing_utils_fast.py
image_transforms.py
image_utils.py
audio_utils.py
video_processing_utils.py
video_utils.py

```

Primary responsibilities:
- Processor composition (combining tokenizer + modality preprocessors)
- Feature extractors and base contracts
- Image processing base classes + shared image transforms/utils
- Audio/video helpers used by processors and pipelines

Grep keywords:
- `AutoProcessor`
- `ProcessorMixin`
- `FeatureExtractionMixin`
- `ImageProcessingMixin`
- `VideoProcessingMixin`

---

## Generation (text generation / decoding / streaming)

Canonical entry points:

```
src/transformers/generation/
configuration_utils.py
utils.py
logits_process.py
stopping_criteria.py
streamers.py
beam_search.py
beam_constraints.py
candidate_generator.py
watermarking.py

# Cache utilities used by generation (and models)

src/transformers/
cache_utils.py
```

Primary responsibilities:
- `GenerationConfig` (defaults + `generation_config.json` serialization)
- `GenerationMixin.generate()` (PyTorch generation loop)
- Logits processors/warpers, stopping criteria, streamers
- Beam search + constraints, candidate generation helpers, watermarking
- KV cache helpers (`cache_utils.py`)

Grep keywords:
- `class GenerationMixin`
- `def generate(`
- `LogitsProcessor`
- `StoppingCriteria`
- `TextStreamer`
- `DynamicCache` / `StaticCache`

---

## Pipelines (high-level inference)

Canonical entry points:

```
src/transformers/pipelines/
**init**.py
base.py
```

Notes:
- `pipelines/__init__.py` defines the task registry and the `pipeline()` entry point.
- `pipelines/base.py` contains the core `Pipeline` base class and shared inference glue.
- Task-specific pipelines typically follow `pipelines/<task>.py`.

Grep keywords:
- `class Pipeline`
- `pipeline(`
- `SUPPORTED_TASKS`

---

## Training / evaluation (Trainer)

Canonical entry points:

```
src/transformers/
trainer.py
trainer_seq2seq.py
trainer_callback.py
trainer_utils.py
trainer_pt_utils.py
training_args.py
training_args_seq2seq.py
optimization.py

src/transformers/data/
**init**.py
data_collator.py
```

Primary responsibilities:
- `Trainer` training/eval loops, logging, checkpointing
- callback system
- `TrainingArguments` and helper utilities
- optimizer/scheduler helpers (`optimization.py`)
- data collators

Grep keywords:
- `class Trainer`
- `TrainingArguments`
- `def training_step(`
- `CallbackHandler`
- `get_scheduler`
- `DataCollator`

---

## Auto classes (model/config/tokenizer/processor dispatch)

Canonical entry points:

```
src/transformers/models/auto/
configuration_auto.py
modeling_auto.py
modeling_tf_auto.py
modeling_flax_auto.py
tokenization_auto.py
processing_auto.py
feature_extraction_auto.py
image_processing_auto.py
video_processing_auto.py
auto_factory.py
```

Primary responsibilities:
- mapping tables from `model_type` / config class → model/tokenizer/processor classes
- common auto-loading errors are raised from Auto* dispatch stack (often `configuration_auto.py` / `auto_factory.py`)

Grep keywords:
- `MODEL_MAPPING`
- `CONFIG_MAPPING`
- `TOKENIZER_MAPPING`
- `PROCESSOR_MAPPING`
- `model_type`

---

## Models (per-architecture packages)

**Pattern (model implementations):**

```
src/transformers/models/<model_name>/
configuration_<model_name>.py
modeling_<model_name>.py
modeling_tf_<model_name>.py          # optional
modeling_flax_<model_name>.py        # optional
tokenization_<model_name>.py         # optional
tokenization_<model_name>*fast.py    # optional
processing*<model_name>.py           # optional
image_processing_<model_name>.py     # optional
feature_extraction_<model_name>.py   # optional
generation_<model_name>.py           # optional (model-specific generation helpers)

# sometimes: video_processing_<model_name>.py, etc.
```

Handy anchors (examples you’ll often see):

```
src/transformers/models/bert/modeling_bert.py
src/transformers/models/t5/modeling_t5.py
src/transformers/models/llama/modeling_llama.py
src/transformers/models/qwen2/modeling_qwen2.py
src/transformers/models/clip/modeling_clip.py
```

Grep keywords:
- `class .*Model`
- `class .*PreTrainedModel`
- `CONFIG_CLASS`

---

## Performance / kernels / attention backends (common “why is this slow / different?”)

Canonical entry points:

```
src/transformers/
modeling_attn_mask_utils.py
modeling_flash_attention_utils.py
modeling_rope_utils.py
modeling_gguf_pytorch_utils.py
```

Related integration shims (backend-specific routing often lives here):

```
src/transformers/integrations/
flash_attention.py
flex_attention.py
sdpa_attention.py
tensor_parallel.py
```

Grep keywords:
- `flash_attention`
- `scaled_dot_product_attention`
- `sdpa`
- `use_flash_attention`
- `gguf`

---

## Utilities and internals

Canonical entry points (frequently involved in stack traces):

```
src/transformers/utils/
import_utils.py
hub.py
logging.py
versions.py
generic.py
doc.py
chat_template_utils.py
peft_utils.py
quantization_config.py

src/transformers/
file_utils.py
debug_utils.py
testing_utils.py
```

Primary responsibilities:
- Lazy import mechanics and optional dependency gating
- Hub caching/download helpers used by `from_pretrained`
- logging + version utilities
- docstring tooling and generic helpers
- chat template parsing/formatting helpers
- PEFT helper glue
- quantization config objects
- legacy helpers (`file_utils.py`) + debugging/testing utilities

Grep keywords:
- `_LazyModule`
- `requires_backends`
- `is_torch_available`
- `cached_file`
- `apply_chat_template`
- `BitsAndBytesConfig`

---

## Integrations and quantization
### Integrations (external libs + runtimes)

Canonical entry points:

```
src/transformers/integrations/
integration_utils.py
accelerate.py
deepspeed.py
fsdp.py
peft.py
bitsandbytes.py
tiktoken.py
awq.py
quanto.py
```

What lives here:
- external library shims (Accelerate/DeepSpeed/FSDP/PEFT)
- tokenizer backends (e.g., tiktoken) and quant backends (AWQ/Quanto/etc.)
- backend-specific feature routing + capability checks

Grep keywords:
- `requires_backends`
- `is_accelerate_available`
- `is_deepspeed_available`
- `is_bitsandbytes_available`
- `device_map`

### Quantizers (unified quantization abstraction)

Canonical entry points:

```
src/transformers/quantizers/
auto.py
base.py
quantizers_utils.py
quantizer_bnb_4bit.py
quantizer_bnb_8bit.py
quantizer_awq.py
quantizer_gptq.py
quantizer_quanto.py


src/transformers/utils/
quantization_config.py
```

Grep keywords:
- `HfQuantizer`
- `quant_method`
- `BitsAndBytesConfig`
- `load_in_4bit` / `load_in_8bit`
- `AutoHfQuantizer`

---

## Export / ONNX

Canonical entry points:

```
src/transformers/
convert_graph_to_onnx.py
src/transformers/onnx/
**main**.py
config.py
convert.py
features.py
utils.py
```

Grep keywords:
- `OnnxConfig`
- `export`
- `opset`
- `transformers.onnx`

---

## CLI / repo tooling (developer workflows)

Canonical entry points:

```
src/transformers/commands/
transformers_cli.py
chat.py
serving.py
add_new_model_like.py
add_fast_image_processor.py
convert.py
download.py
env.py
run.py
train.py
```

Notes:
- `transformers_cli.py` is the CLI dispatcher.
- `chat.py` implements `transformers chat ...`
- `serving.py` implements `transformers serve ...`

Grep keywords:
- `main(`
- `argparse`
- `transformers chat`
- `transformers serve`
- `add_new_model_like`

---

## Production notes (for Skills maintainers)

1. **Pin Transformers**: tie generated references to a specific tag/commit or exact PyPI version.
2. **Regenerate on upgrade**: when bumping Transformers, regenerate this map alongside any other generated references.
3. **Keep this file curated**: add new *canonical entry points* as Transformers evolves—don’t mirror the full repo tree.
4. **Security**: if you ship scripts alongside Skills, keep them least-privilege and auditable.

---

## Quick “where is X implemented?” cheat sheet

| User asks about… | Start here | Then follow into… |
|---|---|---|
| `pipeline()` / task pipelines | `src/transformers/pipelines/__init__.py` | `pipelines/base.py` + task file |
| `AutoModel*` / auto dispatch | `src/transformers/models/auto/modeling_auto.py` | `auto_factory.py` + model subpackage |
| `AutoTokenizer` | `src/transformers/models/auto/tokenization_auto.py` | model tokenizer module |
| `AutoProcessor` | `src/transformers/models/auto/processing_auto.py` | model processor module |
| `from_pretrained` (models) | `src/transformers/modeling_utils.py` | then `src/transformers/utils/hub.py` (caching/shards) |
| `from_pretrained` (configs) | `src/transformers/configuration_utils.py` | config subclass in model subpackage |
| `generate()` behavior | `src/transformers/generation/utils.py` | logits/stopping/streamers + beam/candidate helpers |
| stopping criteria / stop strings | `src/transformers/generation/stopping_criteria.py` | called from generation utils |
| KV cache / caching behavior | `src/transformers/cache_utils.py` | used by generation + some models |
| quantization (general) | `src/transformers/quantizers/auto.py` | specific `quantizer_*.py` + `utils/quantization_config.py` |
| bitsandbytes 4-bit/8-bit | `src/transformers/integrations/bitsandbytes.py` | `quantizers/quantizer_bnb_*.py` |
| `Trainer` loop / callbacks | `src/transformers/trainer.py` | `trainer_callback.py`, `trainer_utils.py` |
| schedulers / optim helpers | `src/transformers/optimization.py` | used from Trainer / scripts |
| data collators | `src/transformers/data/data_collator.py` | task-specific collator classes |
| ONNX export | `src/transformers/onnx/convert.py` | `onnx/config.py` + `onnx/features.py` |
| CLI: `transformers chat` | `src/transformers/commands/chat.py` | `commands/transformers_cli.py` |
| CLI: `transformers serve` | `src/transformers/commands/serving.py` | `commands/transformers_cli.py` |
```