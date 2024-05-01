<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tokenizer

A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most
of the tokenizers are available in two flavors: a full python implementation and a "Fast" implementation based on the
Rust library [🤗 Tokenizers](https://github.com/huggingface/tokenizers). The "Fast" implementations allows:

1. a significant speed-up in particular when doing batched tokenization and
2. additional methods to map between the original string (character and words) and the token space (e.g. getting the
   index of the token comprising a given character or the span of characters corresponding to a given token). 

The base classes [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`]
implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and
"Fast" tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library
(downloaded from HuggingFace's AWS S3 repository). They both rely on
[`~tokenization_utils_base.PreTrainedTokenizerBase`] that contains the common methods, and
[`~tokenization_utils_base.SpecialTokensMixin`].

[`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`] thus implement the main
methods for using all the tokenizers:

- Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and
  encoding/decoding (i.e., tokenizing and converting to integers).
- Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece...).
- Managing special tokens (like mask, beginning-of-sentence, etc.): adding them, assigning them to attributes in the
  tokenizer for easy access and making sure they are not split during tokenization.

[`BatchEncoding`] holds the output of the
[`~tokenization_utils_base.PreTrainedTokenizerBase`]'s encoding methods (`__call__`,
`encode_plus` and `batch_encode_plus`) and is derived from a Python dictionary. When the tokenizer is a pure python
tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by
these methods (`input_ids`, `attention_mask`...). When the tokenizer is a "Fast" tokenizer (i.e., backed by
HuggingFace [tokenizers library](https://github.com/huggingface/tokenizers)), this class provides in addition
several advanced alignment methods which can be used to map between the original string (character and words) and the
token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding
to a given token).


## PreTrainedTokenizer

[[autodoc]] PreTrainedTokenizer
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## PreTrainedTokenizerFast

The [`PreTrainedTokenizerFast`] depend on the [tokenizers](https://huggingface.co/docs/tokenizers) library. The tokenizers obtained from the 🤗 tokenizers library can be
loaded very simply into 🤗 transformers. Take a look at the [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) page to understand how this is done.

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding

[[autodoc]] BatchEncoding
