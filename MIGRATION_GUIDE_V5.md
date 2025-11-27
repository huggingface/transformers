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




## Tokenization

Just as we moved towards a single backend library for model definition, we want `Tokenizer` to be a lot more intuitive.
With v5, you can now initialize an empty `LlamaTokenizer` and train it directly on your new task! 

Defining a new tokenizer object should be as simple as this:
```python
from transformers import TokenizersBackend, generate_merges
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.model import BPE

class Llama5Tokenizer(TokenizersBackend):
    def __init__(self,        unk_token="<unk>",bos_token="<s>", eos_token="</s>", vocab=None, merges=None ):
        if vocab is None:
            self._vocab = {
                str(unk_token): 0,
                str(bos_token): 1,
                str(eos_token): 2,
            }

        else:
            self._vocab = vocab

        if merges is not None:
            self._merges = merges
        else:
            self._merges = generate_merges(filtered_vocab)

        self._tokenizer = Tokenizer(
            BPE(vocab=self._vocab, merges=self._merges, fuse_unk=True)
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=_get_prepend_scheme(self.add_prefix_space, self), split=False
        )
        super().__init__(
            tokenizer_object=self._tokenizer,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
```

And now if you call `Llama5Tokenizer()` you just get an empty, trainable tokenizer that follows the definition of the authors of `Llama5` (it does not exist yet :wink:).

The above is the main motivation towards refactoring tokenization: we want people to just instantiate a tokenizer like they would a model, empty or not and with exactly what they defined.

### Non-tokenizers
If you tokenizers is not common, or you just don't want to rely on `sentencepiece` nor `tokenizers` you can just import the `PythonBackend` (previousl `PreTrainedTokenzier`) which has all the API and logic for added tokens, encoding and decoding wieht them etc. 

If you want to have en less features, you can use the common `PreTrainedTokenizerBase` mixin, which mostly defines `transformers` tokenizer API: `encode`, `decode`, `vocab_size`, `get_vocab`, `convert_tokens_to_ids`, `convert_ids_to_tokens`, `from_pretrained`, `save_pretrained`, etc.

### Backend Architecture Changes

**Moving away from "slow" vs "fast" tokenizers:**

Previously, transformers maintained two parallel implementations for many tokenizers:
- "Slow" tokenizers (`tokenization_<model>.py`) - Python-based implementations, often using [SentencePiece](https://github.com/google/sentencepiece) as the backend.
- "Fast" tokenizers (`tokenization_<model>_fast.py`) - Rust-based implementations using the 🤗 [tokenizers](https://github.com/huggingface/tokenizers) library.

In v5, we consolidate to a single tokenizer file per model: `tokenization_<model>.py`. This file will use the most appropriate backend available:

1. **TokenizersBackend** (preferred): Rust-based tokenizers from the 🤗 [tokenizers](https://github.com/huggingface/tokenizers) library. In general its performances are better, but it also offers a lot more features that are comonly adopted across the ecosystem, like handling additional tokens, easily update the state of the tokenizer, automatic parallelisation etc. 
2. **SentencePieceBackend**: For models requiring SentencePiece
3. **PythonBackend**: Pure Python implementations
4. **MistralCommonBackend**: Relies on `MistralCommon`'s toknenization library. (Previously `MistralCommonTokenizer`)

The `AutoTokenizer` automatically selects the appropriate backend based on available files and dependencies. This is transparent, you continue to use `AutoTokenizer.from_pretrained()` as before. This allows transformers to be future-proof and modular to easily support future backends.


### API Changes

**1. Direct tokenizer initialization with vocab and merges:**

In v5, you can now initialize tokenizers directly with vocabulary and merges, enabling training custom tokenizers from scratch:

```python
# v5: Initialize a blank tokenizer for training
from transformers import LlamaTokenizer

# Create a tokenizer with custom vocabulary and merges
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}
merges = [("h", "e"), ("l", "l"), ("o", " ")]

tokenizer = LlamaTokenizer(vocab=vocab, merges=merges)

# Or initialize a blank tokenizer to train on your own dataset
tokenizer = LlamaTokenizer()  # Creates a blank Llama-like tokenizer
```
But you can no longer pass a vocab file. As this accounts for `from_pretrained` use-case.

**2. Simplified decoding API:**

The `batch_decode` method has been unified with `decode`. Both single and batch decoding now use the same method:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small") 
inputs = ["hey how are you?", "fine"]
tokenizer.decode(tokenizer.encode(inputs))
```
Gives:
```diff
- 'hey how are you?</s> fine</s>'
+ ['hey how are you?</s>', 'fine</s>']
```

This is mostly because people get `list[list[int]]` out of `generate`, but then they would use `decode` because they use `encode` and would get:
```python
   ...: tokenizer.decode([[1,2], [1,4]])
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[2], line 4
      2 tokenizer = AutoTokenizer.from_pretrained("t5-small") 
      3 inputs = ["hey how are you?", "fine"]
----> 4 tokenizer.decode([[1,2], [1,4]])

File /raid/arthur/transformers/src/transformers/tokenization_utils_base.py:3948, in PreTrainedTokenizerBase.decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs)
   3945 # Convert inputs to python lists
   3946 token_ids = to_py_obj(token_ids)
-> 3948 return self._decode(
   3949     token_ids=token_ids,
   3950     skip_special_tokens=skip_special_tokens,
   3951     clean_up_tokenization_spaces=clean_up_tokenization_spaces,
   3952     **kwargs,
   3953 )

File /raid/arthur/transformers/src/transformers/tokenization_utils_fast.py:682, in PreTrainedTokenizerFast._decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs)
    680 if isinstance(token_ids, int):
    681     token_ids = [token_ids]
--> 682 text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    684 clean_up_tokenization_spaces = (
    685     clean_up_tokenization_spaces
    686     if clean_up_tokenization_spaces is not None
    687     else self.clean_up_tokenization_spaces
    688 )
    689 if clean_up_tokenization_spaces:

TypeError: argument 'ids': 'list' object cannot be interpreted as an integer
```

**3. Unified encoding API:**

The `encode_plus` is deprecated → call directly with `__call__`

**3. `apply_chat_template` returns `BatchEncoding`:**

Previously, `apply_chat_template` returned `input_ids` for backward compatibility. In v5, it now consistently returns a `BatchEncoding` dict like other tokenizer methods:

```python
# v5
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]

# Now returns BatchEncoding with input_ids, attention_mask, etc.
outputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
print(outputs.keys())  # dict_keys(['input_ids', 'attention_mask'])
```

#### Removed legacy configuration file saving:

- `special_tokens_map.json` - special tokens are now stored in `tokenizer_config.json`.
- `added_tokens.json` - added tokens are now stored in `tokenizer.json`.
- `added_tokens_decoder` is only stored when there is no `tokenizer.json`.

When loading older tokenizers, these files are still read for backward compatibility, but new saves use the consolidated format.

### Model-Specific Changes

Several models that had identical tokenizers now import from their base implementation:

- **LayoutLM** → uses BertTokenizer
- **LED** → uses BartTokenizer  
- **Longformer** → uses RobertaTokenizer
- **LXMert** → uses BertTokenizer
- **MT5** → uses T5Tokenizer
- **MVP** → uses BartTokenizer

We're just gonna remove these files at term.

**Removed T5-specific workarounds:**

The internal `_eventually_correct_t5_max_length` method has been removed. T5 tokenizers now handle max length consistently with other models.

### Testing Changes

Model-specific tokenization test files now focus on integration tests.
Common tokenization API tests (e.g., `add_tokens`, `encode`, `decode`) are now centralized and automatically applied across all tokenizers. This reduces test duplication and ensures consistent behavior


For legacy implementations, the original BERT Python tokenizer code (including `WhitespaceTokenizer`, `BasicTokenizer`, etc.) is preserved in `bert_legacy.py` for reference purposes.

**Linked PRs:**
- https://github.com/huggingface/transformers/issues/40938
- https://github.com/huggingface/transformers/pull/40936
- https://github.com/huggingface/transformers/pull/41626


## Library-wide changes with lesser impact

### `use_auth_token`

The `use_auth_token` argument/parameter is deprecated in favor of `token` everywhere.
You should be able to search and replace `use_auth_token` with `token` and get the same logic.

Linked PR: https://github.com/huggingface/transformers/pull/41666

### Attention-related features

We decided to remove some features for the upcoming v5 as they are currently only supported in a few old models and no longer integrated in current model additions. It's recommended to stick to v4.x in case you need them. Following features are affected:
- No more head masking, see [#41076](https://github.com/huggingface/transformers/pull/41076). This feature allowed to turn off certain heads during the attention calculation and only worked for eager.
- No more relative positional biases in Bert-like models, see [#41170](https://github.com/huggingface/transformers/pull/41170). This feature was introduced to allow relative position scores within attention calculations (similar to T5). However, this feature is barely used in official models and a lot of complexity instead. It also only worked with eager.
- No more head pruning, see [#41417](https://github.com/huggingface/transformers/pull/41417) by @gante. As the name suggests, it allowed to prune heads within your attention layers.

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

- Methods to init a nested config such as `from_xxx_config` are deleted. Configs can be init from the `__init__` method in the same way. See [#41314](https://github.com/huggingface/transformers/pull/41314).
- It is no longer possible to load a config class from a URL file. Configs must be loaded from either a local path or a repo on the Hub. See [#42383](https://github.com/huggingface/transformers/pull/42383).
- All parameters for configuring model's rotary embedding are now stored under `mode.rope_parameters`, including the `rope_theta` and `rope_type`. Model's `config.rope_parameters` is a simple dictionaty in most cases, and can also be a nested dict in special cases (i.e. Gemma3 and ModernBert) with different rope parameterization for each layer type. See [#39847](https://github.com/huggingface/transformers/pull/39847)
- Qwen-VL family configuration is in a nested format and trying to access keys directly will throw an error (e.g. `config.vocab_size`). Users are expected to access keys from their respective sub-configs (`config.text_config.vocab_size`).

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
- Vision Language models will not have a shortcut access to its language and vision component from the generative model via `model.language_model`. It is recommended to either access the module with `model.model.language_model` or `model.get_decoder()`. See [#42156](https://github.com/huggingface/transformers/pull/42156/)

### Generate

- Old, deprecated output type aliases were removed (e.g. `GreedySearchEncoderDecoderOutput`). We now only have 4 output classes built from the following matrix: decoder-only vs encoder-decoder, uses beams vs doesn't use beams (https://github.com/huggingface/transformers/pull/40998)
- Removed deprecated classes regarding decoding methods that were moved to the Hub due to low usage (constraints and beam scores) (https://github.com/huggingface/transformers/pull/41223)
- If `generate` doesn't receive any KV Cache argument, the default cache class used is now defined by the model (as opposed to always being `DynamicCache`) (https://github.com/huggingface/transformers/pull/41505)
- Generation parameters are no longer accessible via model's config. If generation paramaters are serialized in `config.json` for any old model, it will be loaded back into model's generation config. Users are expected to access or modify generation parameters only with `model.generation_config.do_sample = True`. 

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

## Pipeline

- Image text to text pipelines will no longer accept images as a separate argument along with conversation chats. Image data has to be embedded in the chat's "content" field. See [#42359](https://github.com/huggingface/transformers/pull/42359)

## PushToHubMixin

- removed deprecated `organization` and `repo_url` from `PushToHubMixin`. You must pass a `repo_id` instead.
- removed `ignore_metadata_errors` from `PushToMixin`. In practice if we ignore errors while loading the model card, we won't be able to push the card back to the Hub so it's better to fail early and not provide the option to fail later.
- `push_to_hub` do not accept `**kwargs` anymore. All accepted parameters are explicitly documented.
- arguments of `push_to_hub` are now keyword-only to avoid confusion. Only `repo_id` can be positional since it's the main arg.
- removed `use_temp_dir` argument from `push_to_hub`. We now use a tmp dir in all cases.

Linked PR: https://github.com/huggingface/transformers/pull/42391.

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


### Removal of the `run` method

The `transformers run` (previously `transformers-cli run`) is an artefact of the past, was not documented nor tested,
and isn't part of any public documentation. We're removing it for now and ask you to please let us know in case
this is a method you are using; in which case we should bring it back with better support.

Linked PR: https://github.com/huggingface/transformers/pull/42447

## Environment variables

- Legacy environment variables like `TRANSFORMERS_CACHE`, `PYTORCH_TRANSFORMERS_CACHE`, and `PYTORCH_PRETRAINED_BERT_CACHE` have been removed. Please use `HF_HOME` instead.
- Constants `HUGGINGFACE_CO_EXAMPLES_TELEMETRY`, `HUGGINGFACE_CO_EXAMPLES_TELEMETRY`, `HUGGINGFACE_CO_PREFIX`, and `HUGGINGFACE_CO_RESOLVE_ENDPOINT` have been removed. Please use `huggingface_hub.constants.ENDPOINT` instead.

Linked PR: https://github.com/huggingface/transformers/pull/42391.

## Requirements update

`transformers` v5 pins the `huggingface_hub` version to `>=1.0.0`. See this [migration guide](https://huggingface.co/docs/huggingface_hub/concepts/migration) to learn more about this major release. Here are to main aspects to know about:
- switched the HTTP backend from `requests` to `httpx`. This change was made to improve performance and to support both synchronous and asynchronous requests the same way. If you are currently catching `requests.HTTPError` errors in your codebase, you'll need to switch to `httpx.HTTPError`.
- related to 1., it is not possible to set proxies from your script. To handle proxies, you must set the `HTTP_PROXY` / `HTTPS_PROXY` environment variables
- `hf_transfer` and therefore `HF_HUB_ENABLE_HF_TRANSFER` have been completed dropped in favor of `hf_xet`. This should be transparent for most users. Please let us know if you notice any downside!

`typer-slim` has been added as required dependency, used to implement both `hf` and `transformers` CLIs.