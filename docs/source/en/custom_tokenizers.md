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

# Customizing tokenizers

Tokenizers are decoupled from their learned vocabularies. This allows you to initialize an empty tokenizer for training or create one directly with your own vocabulary. The underlying tokenization pipeline remains the same (normalizer, pre-tokenizer, tokenization algorithm) so you don't need to recreate it from scratch.

This guide shows how to train and create a custom tokenizer.

## Training a tokenizer

An empty trainable tokenizer replaces the vocabulary with a new target vocabulary. This is useful for adapting to a new domain like finance, a low-resource language, or code.

Create an empty tokenizer and load a dataset.

```py
from datasets import load_dataset
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer()
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
```

Use the [`TokenizersBackend.train_new_from_iterator`] method to train the tokenizer. This method accepts a generator function to return chunks of text from the dataset instead of loading everything into memory at once. The `vocab_size` argument sets the tokenizers vocabulary size.

```py
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["assistant"]

trained_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=32000,
)
encoded = trained_tokenizer("The stock market rallied today.")
print(encoded["input_ids"])
[5866, 11503, 98, 5885, 8617, 13381, 30]
```

Add new special tokens with the `new_special_tokens` argument or use `special_tokens_map` to rename the old special tokens to the new special tokens.

Save the new finance tokenizer with [`~PreTrainedTokenizerBase.save_pretrained`] or save and upload it to the Hub with [`~PreTrainedTokenizerBase.push_to_hub`]. This creates a `tokenizer.json` file that captures the newly trained vocabulary, merge rules, and full pipeline configuration.

```py
trained_tokenizer.save_pretrained("./finance-gemma-tokenizer")
trained_tokenizer.push_to_hub("finance-gemma-tokenizer")
```

## Custom vocabulary

An empty tokenizer supports custom vocabulary with the `vocab` and `merges` arguments.

- `vocab` is the complete set of tokens a tokenizer knows and each entry maps a token to its input id.
- `merges` defines how the BPE algorithm should combine adjacent tokens. 

```py
from transformers import GemmaTokenizer

vocab={
    "<pad>": 0,
    "</s>": 1,
    "<s>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "▁the": 5,
    "▁stock": 6,
    "▁market": 7,
    "▁": 8,
    "r": 9,
    "a": 10,
    "l": 11,
    "i": 12,
    "e": 13,
    "d": 14,
    "ra": 15,
    "li": 16,
    "lie": 17,
    "lied": 18,
    "ral": 19,
    "ralli": 20,
    "rallie": 21,
    "rallied": 22,
}
merges=[
    ("r", "a"),       # r + a → ra
    ("l", "i"),       # l + i → li
    ("li", "e"),      # li + e → lie
    ("lie", "d"),     # lie + d → lied
    ("ra", "l"),      # ra + l → ral
    ("ral", "li"),    # ral + li → ralli
    ("ralli", "e"),   # ralli + e → rallie
    ("rallie", "d"),  # rallie + d → rallied
]

tokenizer = GemmaTokenizer(vocab=vocab, merges=merges)
encoded = tokenizer("the stock market rallied")
print(encoded["input_ids"])
```

## Subclassing TokenizersBackend

Tokenizers supports four different [backends](./fast_tokenizers#backends). Generally, you should use the [`TokenizersBackend`] to define a new tokenizer because it's faster.

> [!TIP]
> The [`PythonBackend`] is a pure Python tokenizer that does not rely on backends like Rust, SentencePiece, or mistral-common. You should only use [`PythonBackend`] if you're building a very specialized tokenizer that can't be expressed by the Rust backend.

1. Subclass the [`TokenizersBackend`] with class attributes like padding side and the tokenization algorithm to use.
2. Define the tokenization pipeline in the `__init__`. This includes the tokenization algorithm to use, how to split the raw text before the algorithm, and how to decode the tokens back to text.

```py
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from transformers import TokenizersBackend

class NewTokenizer(TokenizersBackend):
    padding_side = "left"
    model = BPE

    def __init__(
        self,
        vocab=None,
        merges=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    ):
        self._vocab = vocab or {
            str(unk_token): 0,
            str(bos_token): 1,
            str(eos_token): 2,
            str(pad_token): 3,
        }
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(vocab=self._vocab, merges=self._merges, fuse_unk=True)
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )
```

Train or save the new empty tokenizer.

```py
tokenizer = NewTokenizer()

# train on new corpus
tokenizer.train_new_from_iterator()
# save tokenizer
tokenizer.save_pretrained("./new-tokenizer")
```
