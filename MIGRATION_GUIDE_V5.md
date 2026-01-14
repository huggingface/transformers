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

> [!NOTE] 
> 👀 Welcome to the migration guide for the first release candidate! Nothing is final and things are still actively in 
> movement. We have a section dedicated to what is planned for future release candidates, yet is known not to work in 
> the RC0. Look for "Disclaimers for the RC0".
> 
> We'll be eagerly awaiting your feedback in our GitHub issues!

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

Just as we moved towards a single backend library for model definition, we want our tokenizers, and the `Tokenizer` object to be a lot more intuitive. With v5, tokenizer definition is much simpler; one can now initialize an empty `LlamaTokenizer` and train it directly on your corpus.

Defining a new tokenizer object should be as simple as this:

```python
from transformers import TokenizersBackend, generate_merges
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.model import BPE

class Llama5Tokenizer(TokenizersBackend):
    def __init__(self, unk_token="<unk>",bos_token="<s>", eos_token="</s>", vocab=None, merges=None ):
        if vocab is None:
            self._vocab = {
                str(unk_token): 0,
                str(bos_token): 1,
                str(eos_token): 2,
            }

        else:
            self._vocab = vocab

        if merges is not None:
            self._merges = merges or []
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

Once the tokenizer is defined as above, you can load it with the following: `Llama5Tokenizer()`. Doing this returns you an empty, trainable tokenizer that follows the definition of the authors of `Llama5` (it does not exist yet :wink:).

The above is the main motivation towards refactoring tokenization: we want tokenizers to behave similarly to models: trained or empty, and with exactly what is defined in their class definition.

### Backend Architecture Changes: moving away from the slow/fast tokenizer separation

Up to now, transformers maintained two parallel implementations for many tokenizers:
- "Slow" tokenizers (`tokenization_<model>.py`) - Python-based implementations, often using [SentencePiece](https://github.com/google/sentencepiece) as the backend.
- "Fast" tokenizers (`tokenization_<model>_fast.py`) - Rust-based implementations using the 🤗 [tokenizers](https://github.com/huggingface/tokenizers) library.

In v5, we consolidate to a single tokenizer file per model: `tokenization_<model>.py`. This file will use the most appropriate backend available:

1. **TokenizersBackend** (preferred): Rust-based tokenizers from the 🤗 [tokenizers](https://github.com/huggingface/tokenizers) library. In general it provides optimal performance, but it also offers a lot more features that are commonly adopted across the ecosystem:
  - handling additional tokens
  - a full python API for setting and updating 
  - automatic parallelization,
  - automatic offsets
  - customization
  - training
2. **SentencePieceBackend**: for tokenizers requiring the `sentencepiece` library. It inherits from `PythonBackend`. 
3. **PythonBackend**: a Python implementations of the features provided by `tokenizers`. Basically allows adding tokens.
4. **MistralCommonBackend**: relies on `MistralCommon`'s tokenization library. (Previously known as the `MistralCommonTokenizer`)

The `AutoTokenizer` automatically selects the appropriate backend based on available files and dependencies. This is transparent, you continue to use `AutoTokenizer.from_pretrained()` as before. This allows transformers to be future-proof and modular to easily support future backends.

### Defining a tokenizers outside of the existing backends

We enable users and tokenizer builders to define their own tokenizers from top to bottom. Tokenizers are usually defined using a backend such as `tokenizers`, `sentencepiece` or `mistral-common`, but we offer the possibility to design the tokenizer at a higher-level, without relying on those backends.

To do so, you can import the `PythonBackend` (which was previously known as `PreTrainedTokenizer`). This class encapsulates all the logic related to added tokens, encoding, and decoding.

If you want something even higher up the stack, then `PreTrainedTokenizerBase` is what `PythonBackend` inherits from. It contains the very basic tokenizer API features: 
- `encode`
- `decode`
- `vocab_size`
- `get_vocab`
- `convert_tokens_to_ids`
- `convert_ids_to_tokens`
- `from_pretrained`
- `save_pretrained`
- among a few others

**Note for implementing new tokenizers:** When creating a tokenizer class that loads from SentencePiece files, you can override the `convert_from_spm` class method in your converter to customize vocabulary structure, normalizers, regexes and anything that you would want to be passed to the tokenizers your are converting. 
This is useful if the model requires specific token ordering or special split regex patterns. See existing converter classes in `convert_slow_tokenizer.py` for examples.

### API Changes

#### 1. Direct tokenizer initialization with vocab and merges

Starting with v5, we now enable initializing blank, untrained `tokenizers`-backed tokenizers:

```py
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer()
```

This tokenizer will therefore follow the definition of the `LlamaTokenizer` as defined in its class definition. It can then be trained on a corpus as can be seen in [the `tokenizers` documentation](https://huggingface.co/docs/tokenizers/training_from_memory).

These tokenizers can also be initialized from vocab and merges (if necessary), like the previous "slow" tokenizers:

```py
from transformers import LlamaTokenizer

vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}
merges = [("h", "e"), ("l", "l"), ("o", " ")]

tokenizer = LlamaTokenizer(vocab=vocab, merges=merges)
```

This tokenizer will behave as a Llama-like tokenizer, with an updated vocabulary. This allows comparing different tokenizer classes with the same vocab; therefore enabling the comparison of different pre-tokenizers, normalizers, etc.

**Simplified file loading:** Support is added for passing`vocab` and `merges` as file paths directly to tokenizer initialization. The tokenizer will automatically detect the format (SentencePiece `.model`, Tekken `tekken.json`, or plain vocab/merges files) for loading. For BPE tokenizers, if a vocab is provided but no merges, merges will be automatically generated (excluding special tokens).

Note: Loading from file paths with `vocab="<path_to_a_file>"`'s primary goal is to allow you to do some quick testing, but for `BPE` models for example we don't check whether you properly passed the merges or not. 

#### 2. Simplified decoding API

The `batch_decode` and `decode` methods have been unified to reflect behavior of the `encode` method. Both single and batch decoding now use the same `decode` method. See an example of the new behavior below:

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

We expect `encode` and `decode` to behave, as two sides of the same coin: `encode`, `process`, `decode`,  should work. 

> [!NOTE]
> A common use-case would be: `encode`, `model.generate`, `decode`.  However, using `generate` would return `list[list[int]]`, which would then be incompatible with `decode`.

#### 3. Unified encoding API

The `encode_plus` method is deprecated in favor of the single `__call__` method.

#### 4. `apply_chat_template` returns `BatchEncoding`

Previously, `apply_chat_template` returned `input_ids` for backward compatibility. Starting with v5, it now consistently returns a `BatchEncoding` dict like other tokenizer methods.

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

#### 5. Removed legacy configuration file saving:

We simplify the serialization of tokenization attributes:

- `special_tokens_map.json` - special tokens are now stored in `tokenizer_config.json`.
- `added_tokens.json` - added tokens are now stored in `tokenizer.json`.
- `added_tokens_decoder` is only stored when there is no `tokenizer.json`.
- `add_bos_token` and `add_eos_token` - these are no longer saved in `tokenizer_config.json`. When a `tokenizer.json` file exists, these settings are defined in the tokenizer class or `tokenizer.json` itself.

**Backend synchronization removed:** The automatic synchronization logic that updated backend tokenizer settings (like `add_prefix_space`, `do_lower_case`, `strip_accents`, `tokenize_chinese_chars`) after initialization has been removed. Tokenizer behavior is now fully determined by the `tokenizer.json` file or class definition at initialization time. 

When loading older tokenizers, these files are still read for backward compatibility, but new saves use the consolidated format. We're gradually moving towards consolidating attributes to fewer files so that other libraries and implementations may depend on them more reliably.

#### 6. Model-Specific Changes

Several models that had identical tokenizers now import from their base implementation:

- **LayoutLM** → uses BertTokenizer
- **LED** → uses BartTokenizer  
- **Longformer** → uses RobertaTokenizer
- **LXMert** → uses BertTokenizer
- **MT5** → uses T5Tokenizer
- **MVP** → uses BartTokenizer
These modules will eventually be removed altogether.

**Removed T5-specific workarounds**

The internal `_eventually_correct_t5_max_length` method has been removed. T5 tokenizers now handle max length consistently with other models.

### Testing Changes

A few testing changes specific to tokenizers have been applied:
- Model-specific tokenization test files now focus on integration tests.
- Common tokenization API tests (e.g., `add_tokens`, `encode`, `decode`) are now centralized and automatically applied across all tokenizers. This reduces test duplication and ensures consistent behavior

For legacy implementations, the original BERT Python tokenizer code (including `WhitespaceTokenizer`, `BasicTokenizer`, etc.) is preserved in `bert_legacy.py` for reference purposes.

#### 7. Deprecated / Modified Features

**Special Tokens Structure:**
- `SpecialTokensMixin`: Merged into `PreTrainedTokenizerBase` to simplify the tokenizer architecture.
- `special_tokens_map`: Now only stores named special token attributes (e.g., `bos_token`, `eos_token`). Use `extra_special_tokens` for additional special tokens (formerly `additional_special_tokens`). `all_special_tokens` includes both named and extra tokens.

```python
# v4
tokenizer.special_tokens_map  # Included 'additional_special_tokens'

# v5
tokenizer.special_tokens_map  # Only named tokens
tokenizer.extra_special_tokens  # Additional tokens
```

- `special_tokens_map_extended` and `all_special_tokens_extended`: Removed. Access `AddedToken` objects directly from `_special_tokens_map` or `_extra_special_tokens` if needed.
- `additional_special_tokens`: Still accepted for backward compatibility but is automatically converted to `extra_special_tokens`.

**Deprecated Methods:**
- `sanitize_special_tokens()`: Already deprecated in v4, removed in v5.
- `prepare_seq2seq_batch()`: Deprecated; use `__call__()` with `text_target` parameter instead.

```python
# v4
model_inputs = tokenizer.prepare_seq2seq_batch(src_texts, tgt_texts, max_length=128)

# v5
model_inputs = tokenizer(src_texts, text_target=tgt_texts, max_length=128, return_tensors="pt")
model_inputs["labels"] = model_inputs.pop("input_ids_target")
```

- `BatchEncoding.words()`: Deprecated; use `word_ids()` instead.

**Removed Methods:**
- `create_token_type_ids_from_sequences()`: Removed from base class. Subclasses that need custom token type ID creation should implement this method directly.
- `clean_up_tokenization()`: Removed from base class. Now defined at model class level for models that need it (e.g., PLBart, CLVP, Wav2Vec2).
- `prepare_for_model()`, `build_inputs_with_special_tokens()`, `truncate_sequences()`: Moved from `tokenization_utils_base.py` to `tokenization_python.py` for `PythonBackend` tokenizers. `TokenizersBackend` provides model-ready input via `tokenize()` and `encode()`, so these methods are no longer needed in the base class.
- `_switch_to_input_mode()`, `_switch_to_target_mode()`, `as_target_tokenizer()`: Removed from base class. Use `__call__()` with `text_target` parameter instead.

```python
# v4
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_texts, ...)

# v5
labels = tokenizer(text_target=tgt_texts, ...)
```

- `parse_response()`: Removed from base class.

## Disclaimers for the RC0

### PEFT + MoE:

Because we are switching from the naive MOE (`nn.ModuleList` for experts) we currently have an issue with MoEs that have adapters. For more details see https://github.com/huggingface/transformers/issues/42491#issuecomment-3591485649. 

_We aim for this to be fixed and released in a following release candidate in the week that follows RC0._

### Tensor parallel and Expert parallel + MoE

We are streamlining the MoE support with vLLM; while this is being implemented, tensor parallelism and expert parallelism aren't working as expected.
This is known and actively being worked on.

_We aim for this to be fixed and released in a following release candidate in the week that follows RC0._

### Remote code incompatibility

A lot of paths were removed and reworked; paths like `transformers.tokenization_utils` and `transformers.tokenization_utils_fast`, which no longer exist.
These now redirect to `transformers.tokenization_utils_sentencepiece` and `transformers.tokenization_utils_tokenizers` respectively; please update imports accordingly.

_We aim for this to be fixed and released in a following release candidate in the week that follows RC0._

### Custom pretrained models:
For anyone inheriting from a `transformers` `PreTrainedModel`, the weights are automatically initialized with the common scheme: 
```python

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if getattr(module, "weight", None) is not None:
                init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if getattr(module, "weight", None) is not None:
                init.normal_(module.weight, mean=0.0, std=std)
                # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
                if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                    init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if hasattr(module, "weight") and module.weight is not None:
                init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)
```

If you want to avoid that, for now you should just do:

```
class CustomModel(Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_head = nn.Linear(1024, 7)
        self.positional_embedding = nn.Parameter(torch.randn(16, 1152))
        self.post_init()
    
    def _init_weights(self, module):
        pass 

```
There is a tracker for that here: https://github.com/huggingface/transformers/issues/42418.

## Library-wide changes with lesser impact

### Drop support for `safe_serialization=False`

Safetensors is a simple format for storing tensors safely (as opposed to pickle) and that is still fast (zero-copy). It is the preferred file format to store transformers's weights. Prior to transformers `v5`, it was still possible to pass `safe_serialization=False` to fall back to torch's default (and unsafe) file format. This is no longer possible in `v5`. The `safe_serialization` parameter has been removed from all `save_pretrained` and `push_to_hub` methods.

If you really want to export weights to another file format, you must save the `model.state_dict()` by yourself.

Linked PR: https://github.com/huggingface/transformers/issues/42556

### 50GB default shard size

The default shard size went up from `5GB` to `50GB`. Main benefit will be to avoid having tens or hundreds of weight files for large models. This change was made possible thanks to the Xet backend allowing us to efficiently serve very large files. Increasing default shard size was a decision that was only taken after *very careful considerations* around optimizations and load speed. Check out the linked PR for benchmark details.

Linked PR: https://github.com/huggingface/transformers/issues/42556

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


### Auto-classes

- `AutoModelWithLMHead` is removed in favor of `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and `AutoModelForSeq2SeqLM` for encoder-decoder models
- `AutoModelForVision2Seq` is removed in favor of `AutoModelForImageTextToText`


## Configuration

- Methods to init a nested config such as `from_xxx_config` are deleted. Configs can be init from the `__init__` method in the same way. See [#41314](https://github.com/huggingface/transformers/pull/41314).
- It is no longer possible to load a config class from a URL file. Configs must be loaded from either a local path or a repo on the Hub. See [#42383](https://github.com/huggingface/transformers/pull/42383).
- All parameters for configuring model's rotary embedding are now stored under `mode.rope_parameters`, including the `rope_theta` and `rope_type`. Model's `config.rope_parameters` is a simple dictionaty in most cases, and can also be a nested dict in special cases (i.e. Gemma3 and ModernBert) with different rope parameterization for each layer type. Trying to get `config.rope_theta` will throw an attribute error from now on. See [#39847](https://github.com/huggingface/transformers/pull/39847) and [#42255](https://github.com/huggingface/transformers/pull/42255)
- Qwen-VL family configuration is in a nested format and trying to access keys directly will throw an error (e.g. `config.vocab_size`). Users are expected to access keys from their respective sub-configs (`config.text_config.vocab_size`).
- Configurations of non-generative models (any model that doesn't call `model.generate()`) will no longer have a `generation_config` and `model.config.generation_config` will throw an attribute error.

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
- Generation parameters are no longer accessible via model's config. If generation parameters are serialized in `config.json` for any old model, it will be loaded back into model's generation config. Users are expected to access or modify generation parameters only with `model.generation_config.do_sample = True`. 

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

## Pipelines

`Text2TextPipeline`, as well as the related `SummarizationPipeline` and `TranslationPipeline`, were deprecated and will now be removed.
`pipeline` classes are intended as a high-level, beginner-friendly API, but for almost all text-to-text tasks, a modern chat model 
and `TextGenerationPipeline` will provide much higher quality output. As a result, we felt it was misleading for beginners to offer the older pipelines.

If you were using these pipelines before, try using `TextGenerationPipeline` with a chat model instead. For example, for summarization:

```python
import torch
from transformers import pipeline

# Any other chat model will also work - if you're low on memory you can use a smaller one
summarizer = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")  
message_history = [
    {
        "role": "user",
        "content": "Summarize the following text:\n\n[TEXT_TO_SUMMARIZE]"
    }
]
print(summarizer(message_history)[0]["generated_text"][-1]["content"])
```

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

It works hand in hand with `transformers serve`, which means that if `transformers serve` is running on its default endpoint, `transformers chat` can be launched as follows:

```sh
transformers chat HuggingFaceTB/SmolLM3-3B
```

It can however use any OpenAI API compatible HTTP endpoint:

```sh
transformers chat HuggingFaceTB/SmolLM3-3B https://router.huggingface.co/v1
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
