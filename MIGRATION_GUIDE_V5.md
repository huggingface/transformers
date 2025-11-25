<!---
Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Version 5 Migration guide

## Library-wide changes with widespread impact

### Removal of TensorFlow and Jax

We're removing the TensorFlow and Jax parts of the library. This will help us focus fully on `torch` 
going forward and will greatly reduce the maintenance cost of models. We are working with tools from 
the Jax ecosystem still (such as MaxText) in order to see how we can remain compatible with their 
tool while keeping `torch` as the only backend for now.

Linked PR: https://github.com/huggingface/transformers/pull/40760

### Dynamic weight loading

We introduce a new weight loading API in `transformers`, which significantly improves on the previous API. This
weight loading API is designed to apply operations to the checkpoints loaded by transformers.

Instead of loading the checkpoint exactly as it is serialized within the model, these operations can reshape, merge,
and split the layers according to how they're defined in this new API. These operations are often a necessity when
working with quantization or parallelism algorithms.

This new API is centered around the new `WeightConverter` class:

```python
class WeightConverter(WeightTransform):
    operations: list[ConversionOps]
    source_keys: Union[str, list[str]]
    target_keys: Union[str, list[str]]
```

The weight converter is designed to apply a list of operations on the source keys, resulting in target keys. A common
operation done on the attention layers is to fuse the query, key, values layers. Doing so with this API would amount
to defining the following conversion:

```python
conversion = WeightConverter(
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],  # The input layers
    "self_attn.qkv_proj",  # The single layer as output
    operations=[Concatenate(dim=0)],
)
```

In this situation, we apply the `Concatenate` operation, which accepts a list of layers as input and returns a single 
layer. 

This allows us to define a mapping from architecture to a list of weight conversions. Applying those weight conversions
can apply arbitrary transformations to the layers themselves. This significantly simplified the `from_pretrained` method
and helped us remove a lot of technical debt that we accumulated over the past few years.

This results in several improvements:
- Much cleaner definition of transformations applied to the checkpoint
- Reversible transformations, so loading and saving a checkpoint should result in the same checkpoint
- Faster model loading thanks to scheduling of tensor materialization
- Enables complex mix of transformations that wouldn't otherwise be possible (such as quantization + MoEs, or TP + MoEs)

While this is being implemented, expect varying levels of support across different release candidates.

Linked PR: https://github.com/huggingface/transformers/pull/41580

## Library-wide changes with lesser impact

### `use_auth_token`

The `use_auth_token` argument/parameter is deprecated in favor of `token` everywhere.
You should be able to search and replace `use_auth_token` with `token` and get the same logic.

Linked PR: https://github.com/huggingface/transformers/pull/41666

We decided to remove some features for the upcoming v5 as they are currently only supported in a few old models and no longer integrated in current model additions. It's recommended to stick to v4.x in case you need them. Following features are affected:
- No more head masking, see #41076. This feature allowed to turn off certain heads during the attention calculation and only worked for eager.
- No more relative positional biases in Bert-like models, see #41170. This feature was introduced to allow relative position scores within attention calculations (similar to T5). However, this feature is barely used in official models and a lot of complexity instead. It also only worked with eager.
- No more head pruning, see #41417 by @gante. As the name suggests, it allowed to prune heads within your attention layers.

### Updates to supported torch APIs

We dropped support for two torch APIs:
- `torchscript` in https://github.com/huggingface/transformers/pull/41688
- `torch.fx` in https://github.com/huggingface/transformers/pull/41683

Those APIs were deprecated by the PyTorch team, and we're instead focusing on the supported APIs `dynamo` and `export`.

## Quantization changes

We clean up the quantization API in transformers, and significantly refactor the weight loading as highlighted
above.

We drop support for two quantization arguments that have been deprecated for some time:
- `load_in_4bit`
- `load_in_8bit`

We remove them in favor of the `quantization_config` argument which is much more complete. As an example, here is how
you would load a 4-bit bitsandbytes model using this argument:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    quantization_config=quantization_config
)
```


## Configuration

- Methods to init a nested config such as `from_xxx_config` are deleted. Configs can be init from the `__init__` method in the same way (https://github.com/huggingface/transformers/pull/41314)

## Processing

### Tokenization

- Slow tokenizer files (aka: `tokenization_<model>.py` ) will be removed in favor of using fast tokenizer files `tokenization_<model>_fast.py` --> will be renamed to `tokenization_<model>.py`.  As fast tokenizers are :hugs:`tokenizers` - backend, they include a wider range of features that are maintainable and reliable. 
- Other backends (sentence piece, tokenizers, etc.) will be supported with a light layer if loading a fast tokenizer fails
- Remove legacy files like special_tokens_map.json and added_tokens.json
- Remove _eventually_correct_t5_max_length 
- `encode_plus` --> `__call__`
- `batch_decode` --> `decode`

`apply_chat_template` by default returns naked `input_ids` rather than a `BatchEncoding` dict. 
This was inconvenient - it should return a `BatchEncoding` dict like `tokenizer.__call__()`, but we were stuck with 
it for backward compatibility. The method now returns a `BatchEncoding`.

Linked PRs: 
- https://github.com/huggingface/transformers/issues/40938
- https://github.com/huggingface/transformers/pull/40936
- https://github.com/huggingface/transformers/pull/41626

### Processing classes

- In processing classes each attribute will be serialized under `processor_config.json` as a nested dict, instead of serializing attributes in their own config files. Loading will be supported for all old format processors (https://github.com/huggingface/transformers/pull/41474)
- `XXXFeatureExtractors` classes are completely removed in favor of `XXXImageProcessor` class for all vision models (https://github.com/huggingface/transformers/pull/41174)
- Minor change: `XXXFastImageProcessorKwargs` is removed in favor of `XXXImageProcessorKwargs` which will be shared between fast and slow processors (https://github.com/huggingface/transformers/pull/40931)

## Modeling

- Some `RotaryEmbeddings` layers will start returning a dict of tuples, in case the model uses several RoPE configurations (Gemma2, ModernBert). Each value will be a tuple of "cos, sin" per RoPE type.
- Config attribute for `RotaryEmbeddings` layer will be unified and accessed via `config.rope_parameters`. Config attr for `rope_theta` might not be accessible anymore for some models, and instead will be in `config.rope_parameters['rope_theta']`. BC will be supported for a while as much as possible, and in the near future we'll gradually move to the new RoPE format  (https://github.com/huggingface/transformers/pull/39847)

### Generate

- Old, deprecated output type aliases were removed (e.g. `GreedySearchEncoderDecoderOutput`). We now only have 4 output classes built from the following matrix: decoder-only vs encoder-decoder, uses beams vs doesn't use beams (https://github.com/huggingface/transformers/pull/40998)
- Removed deprecated classes regarding decoding methods that were moved to the Hub due to low usage (constraints and beam scores) (https://github.com/huggingface/transformers/pull/41223)
- If `generate` doesn't receive any KV Cache argument, the default cache class used is now defined by the model (as opposed to always being `DynamicCache`) (https://github.com/huggingface/transformers/pull/41505)

## Trainer

### Removing arguments without deprecation cycle in `TrainingArguments` due to low usage

- `mp_parameters` -> legacy param that was later on added to sagemaker trainer
- `_n_gpu` -> not intended for users to set, we will initialize it correctly instead of putting it in the `TrainingArguments`
- `overwrite_output_dir` - > replaced by `resume_from_checkpoint` and it was only used in examples script, no impact on Trainer. 
- `logging_dir` -> only used for tensorboard, set `TENSORBOARD_LOGGING_DIR` env var instead
- `jit_mode_eval` -> use `use_torch_compile` instead as torchscript is not recommended anymore
- `tpu_num_cores`-> It is actually better to remove it as it is not recommended to set the number of cores. By default, all tpu cores are used . Set `TPU_NUM_CORES` env var instead
- `past_index` -> it was only used for a very small number of models that have special architecture like transformersxl + it was not documented at all how to train those model
- `ray_scope` -> only for a minor arg for ray integration. Set `RAY_SCOPE` var env instead 
- `warmup_ratio` -> use `warmup_step` instead. We combined both args together by allowing passing float values in `warmup_step`. 

### Removing deprecated arguments in `TrainingArguments`

- `fsdp_min_num_params` and `fsdp_transformer_layer_cls_to_wrap` -> use `fsdp_config`
- `tpu_metrics_debug` -> `debug` 
- `push_to_hub_token` -> `hub_token`
- `push_to_hub_model_id` and `push_to_hub_organization` -> `hub_model_id`
- `include_inputs_for_metrics` -> `include_for_metrics`
- `per_gpu_train_batch_size` -> `per_device_train_batch_size`
- `per_gpu_eval_batch_size` -> `per_device_eval_batch_size`
- `use_mps_device` -> mps will be used by default if detected
- `fp16_backend` and `half_precision_backend` -> we will only rely on torch.amp as everything has been upstream to torch
- `no_cuda` -> `use_cpu`
- ` include_tokens_per_second` -> `include_num_input_tokens_seen`
- `use_legacy_prediction_loop` -> we only use `evaluation_loop` function from now on

### Removing deprecated arguments in `Trainer`

- `tokenizer` in initialization -> `processing_class`
- `model_path` in train() -> `resume_from_checkpoint`

### Removed features for `Trainer`

- sigpot integration for hp search was removed as the library was archived + the api stopped working
- drop support for sagemaker API <1.10
- bump accelerate minimum version to 1.1.0 

###  New defaults for `Trainer`

- `use_cache` in the model config will be set to `False`. You can still change the cache value through `TrainingArguments` `usel_cache` argument if needed. 

## CLI

The deprecated `transformers-cli ...` command was deprecated, `transformers ...` is now the only CLI entry point.

`transformers` CLI has been migrated to `Typer`, making it easier to maintain + adding some nice features out of 
the box (improved `--help` section, autocompletion).

Biggest breaking change is in `transformers chat`. This command starts a terminal UI to interact with a chat model. 
It used to also be able to start a Chat Completion server powered by `transformers` and chat with it. In this revamped 
version, this feature has been removed in favor of `transformers serve`. The goal of splitting `transformers chat` 
and `transformers serve` is to define clear boundaries between client and server code. It helps with maintenance 
but also makes the commands less bloated. The new signature of `transformers chat` is:

```
Usage: transformers chat [OPTIONS] BASE_URL MODEL_ID [GENERATE_FLAGS]...

  Chat with a model from the command line.
```

Example:

```sh
transformers chat https://router.huggingface.co/v1 HuggingFaceTB/SmolLM3-3B
```

Linked PRs: 
- https://github.com/huggingface/transformers/pull/40997
- https://github.com/huggingface/transformers/pull/41487
