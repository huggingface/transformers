<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tokenizers

A tokenizer converts text into tensors, which are the inputs to a model. It normalizes and splits text, applies the tokenization algorithm, adds special tokens, and decodes output ids back into text.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
{
    'input_ids': tensor([[     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

This guide covers loading, encoding, decoding, batch processing, and the available tokenizer backends.

## Load a tokenizer

Load a tokenizer with the [`AutoTokenizer`] class or a model-specific tokenizer class.

<hfoptions id="tokenizers">
<hfoption id="AutoTokenizer">

[`AutoTokenizer.from_pretrained`] reads the model config, resolves the correct tokenizer class, and returns an instance of it. You don't need to know the tokenizer class beforehand. Most tokenizers resolve to a subclass of [`TokenizersBackend`], a fast Rust-based tokenizer from the [Tokenizers](https://huggingface.co/docs/tokenizers/index) library.

Loading with [`AutoTokenizer`] is the recommended approach.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```

</hfoption>
<hfoption id="model-specific tokenizer">

A model-specific tokenization class is a pre-configured [`TokenizersBackend`] that uses the exact tokenization configuration (normalizer, pre-tokenizer, special token conventions, etc.) a model was trained with.

Use a model-specific class to initialize an empty tokenizer for training or to pass model-specific arguments like `vocab` or `merges` (see the [Customizing tokenizers](./custom_tokenizers) guide to learn how). An empty tokenizer is minimal and only contains a model's special tokens like `<pad>`, `<eos>`, or `<bos>`.

```py
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer()
corpus = [
    ["Sphinx of black quartz, judge my vow."],
    ["Pack my box with five dozen liquor jugs."],
    ["How vexingly quick daft zebras jump!"],
]
new_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=1000)
```

</hfoption>
</hfoptions>

## Encode and decode

The [`TokenizersBackend.__call__`] method encodes text or a batch of text into `input_ids`, `attention_mask`, and other model inputs. It also controls padding, truncation, and special token insertion.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
{
    'input_ids': tensor([[     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

[`TokenizersBackend.encode`] is similar but only returns the `input_ids`.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.encode("Sphinx of black quartz, judge my vow.")
[2, 235277, 82913, 576, 2656, 30407, 235269, 11490, 970, 29871, 235265]
```

[`TokenizersBackend.decode`] converts a single sequence or batch of tokenized `input_ids` back to text.

```py
tokenizer.decode(outputs["input_ids"])
['<bos>Sphinx of black quartz, judge my vow.']
```

[`TokenizersBackend.decode`] preserves the exact tokenization spacing. Set `clean_up_tokenization_spaces` to remove spaces before punctuation, and `skip_special_tokens` to strip special tokens from the output.

```py
tokenizer.decode(outputs["input_ids"], skip_special_tokens=True)
['Sphinx of black quartz, judge my vow.']
``` 

## Special tokens

Special tokens mark structural boundaries in a sequence, like the beginning-of-sequence or padding positions. Each model defines its own set of special tokens. The tokenizer adds them when you call it.

```py
tokenizer.encode("Sphinx of black quartz, judge my vow.")
[2, 235277, 82913, 576, 2656, 30407, 235269, 11490, 970, 29871, 235265]
tokenizer.decode(outputs["input_ids"])
['<bos>Sphinx of black quartz, judge my vow.']
```

Register additional named special tokens with the `extra_special_tokens` argument. Multimodal models use them as placeholders for images, video, or audio.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    extra_special_tokens={"image_token": "<image>"}
)
```

## Batch processing

Batch processing tokenizes multiple sequences in a single call. [`TokenizersBackend`] handles large batches faster because its Rust-based backend parallelizes work across threads.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt"
)
```

Batch processing requires all sequences to share the same length. Padding and truncation are strategies to handle varying-length sequences.

### Padding

Padding appends special tokens so shorter sequences match the longest sequence in a batch. The attention mask marks padding positions as `0` so the model ignores them. Set `padding=True` to pad to the longest sequence or pass `max_length` to pad to a fixed size.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt",
    padding=True,
)
{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265],
        [     0,      2,   6519,    970,   3741,    675,   4105,  25955,  42184, 225789, 235265],
        [     0,      2,   2299,  73378,  17844,   4320, 224463,   4949,  48977,  9902, 235341]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}
```

> [!NOTE]
> Large language models pad on the *left* side to avoid disrupting generation, which predicts the next token from the *right* side.

### Truncation

Truncation clips tokens so a sequence fits within a maximum length. Set `truncation=True` and specify `max_length` to enable it.

Padding and truncation work together. Short sequences gain padding tokens while long sequences lose trailing tokens. Together, they produce a packed rectangular tensor.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=5
)
{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656],
        [     2,   6519,    970,   3741,    675],
        [     2,   2299,  73378,  17844,   4320]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
}
```

## Backends

Each model tokenizer is defined in a single file and supports four tokenization backends.

| backend | implementation | description |
|---|---|---|
| [`TokenizersBackend`] | [Tokenizers](https://huggingface.co/docs/tokenizers) | default for most models |
| [`SentencePieceBackend`] | [SentencePiece](https://github.com/google/sentencepiece) | models requiring SentencePiece |
| [`PythonBackend`] | Python | models requiring specialized custom tokenizers |
| [`MistralCommonBackend`] | [mistral-common](https://mistralai.github.io/mistral-common/) | Mistral and Pixtral models |

All backends inherit from [`PreTrainedTokenizerBase`] and share the same APIs for encoding, decoding, padding, truncation, saving, and loading. The difference is which tokenization pipeline runs underneath.

[`AutoTokenizer`] selects the best available backend when you call [`~AutoTokenizer.from_pretrained`].

1. It reads the `tokenizer_config.json` file for the `tokenizer_class` field.
2. The registry matches `tokenizer_class` to a class name. The resolved class inherits from one of the four backends. For example, [`GemmaTokenizer`] inherits from [`TokenizersBackend`], and [`SiglipTokenizer`] inherits from [`SentencePieceBackend`].

    Some models, like GLM, map directly to [`TokenizersBackend`] because the `tokenizer.json` file fully describes the pipeline. [`GemmaTokenizer`] exists as a subclass since it defines additional model-specific settings in Python that `tokenizer.json` doesn't capture.

    ```py
    TOKENIZER_MAPPING_NAMES = OrderedDict([
        ("gemma2", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("glm", "TokenizersBackend" if is_tokenizers_available() else None),
        (
            "mistral",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        ("siglip", "SiglipTokenizer" if is_sentencepiece_available() else None),
        ...
    ]
    ```
    
    When a backend like mistral-common isn't installed, [`AutoTokenizer`] falls back to [`TokenizersBackend`].

Check which backend a tokenizer is using with the `backend` property.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.backend
'tokenizers'
```

## Inspect the tokenizer architecture

Inspect a tokenizer's internal components (normalizer, pre-tokenizer, model, decoder) with the `_tokenizer` attribute.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
print(tokenizer._tokenizer.normalizer)
print(tokenizer._tokenizer.pre_tokenizer)
print(tokenizer._tokenizer.model)
print(tokenizer._tokenizer.decoder)
```

## Resources

- The [Tokenization in Transformers v5](https://huggingface.co/blog/tokenizers) post discusses the motivation behind the new tokenization backends.
- Review the [migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#tokenization) for an overview of the tokenization changes.
