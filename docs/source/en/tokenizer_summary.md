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

# Tokenization algorithms

<Youtube id="zHvTiHr506c"/>

Transformers support three subword tokenization algorithms: Byte pair encoding (BPE), Unigram, and WordPiece. They split text into units between words and characters, keeping the vocabulary compact while still capturing meaningful pieces. Common words stay intact as single tokens, and rare or unknown words decompose into subwords.

For instance, `annoyingly` might be split into `["annoying", "ly"]` or `["annoy", "ing", "ly"]` depending on the vocabulary. Subword splitting lets the model represent unseen words from known subwords.

> [!TIP]
> Subword tokenization is especially useful for languages like Turkish, where you can form long, complex words by stringing subwords together.

## Byte pair encoding (BPE)

<Youtube id="HEikzVL-lZU"/>

[Byte pair encoding](https://huggingface.co/papers/1508.07909) (BPE) is the most popular tokenization algorithm in Transformers, used by models like [Llama](./model_doc/llama), [Gemma](./model_doc/gemma), [Qwen2](./model_doc/qwen2), and more.

1. A pre-tokenizer splits text on whitespace or other rules, producing a set of unique words and their frequencies.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

2. The BPE algorithm creates a base vocabulary, `["b", "g", "h", "n", "p", "s", "u"]`, from all the characters.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

3. BPE starts with individual characters and iteratively merges the most frequent adjacent pair. `"u"` and `"g"` appear together the most in `"hug"`, `"pug"`, and `"hugs"`, so BPE merges them into `"ug"` and adds it to the vocabulary.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

4. The next most common pair is `"u"` and `"n"`, which appear in `"pun"` and `"bun"`, so they merge into `"un"`.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

5. The vocabulary is now `["b", "g", "h", "n", "p", "s", "u", "ug", "un"]`. BPE continues learning merge rules until it reaches the target vocabulary size, which equals the base vocabulary size plus the number of merges. [GPT](model_doc/openai-gpt) uses BPE with a vocabulary size of 40,478 (478 base tokens + 40,000 merges).

Any character not in the base vocabulary maps to an unknown token like `"<unk>"`. In practice, the base vocabulary covers all characters seen during training, so unknown tokens are rare.

### Byte-level BPE

Including all Unicode characters would make the base vocabulary enormous. Byte-level BPE uses 256 byte values as the base vocabulary instead, ensuring every word can be tokenized without the `"<unk>"` token. [GPT-2](./model_doc/gpt2) uses byte-level BPE with a vocabulary size of 50,257 (256 byte tokens + 50,000 merges + special end-of-text token).

## Unigram

<Youtube id="TGZfZVuF9Yc"/>

[Unigram](https://huggingface.co/papers/1804.10959) is the second most popular tokenization algorithm in Transformers, used by models like [T5](./model_doc/t5), [BigBird](./model_doc/big_bird), [Pegasus](./model_doc/pegasus), and more.

1. Unigram starts with a large set of candidate subwords, and each candidate gets a probability score based on how often it appears.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "pu", "bu", "gs", "hug", "pug", "pun", "bun", "ugs", "hugs"]
```

2. Unigram scores how well the current vocabulary tokenizes the training data at each step.
3. For every token, Unigram measures how much removing the token would increase the overall loss. For example, removing `"pu"` barely affects the loss because `"pug"` and `"pun"` can still be tokenized as `["p", "ug"]` and `["p", "un"]`.

    But removing `"ug"` would significantly increase the loss because `"hug"`, `"pug"`, and `"hugs"` all rely on it.

4. Unigram removes the tokens with the lowest loss increase, usually the bottom 10-20%. Base characters always remain so any word can be tokenized. Tokens like `"bu"`, `"pu"`, `"gs"`, `"pug"`, and `"bun"` are removed because they contributed least to the overall likelihood.

```text
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "hug", "pun", "ugs", "hugs"]
```

5. Steps 2-4 repeat until the vocabulary reaches the target size.

During inference, Unigram can tokenize a word in several ways. `"hugs"` could become `["hug", "s"]`, `["h", "ug", "s"]`, or `["h", "u", "g", "s"]`. Unigram picks the highest probability tokenization. Unlike BPE, which is deterministic and based on merge rules, Unigram is probabilistic and can sample different tokenizations during training.

## SentencePiece

[SentencePiece](https://huggingface.co/papers/1808.06226) is a tokenization library that applies BPE or Unigram directly on raw text. Standard BPE and Unigram assume whitespace separates words, which doesn't work for languages like Chinese and Japanese that don't use spaces.

1. SentencePiece treats the input text as a raw byte or character stream and includes the space character, represented as `"▁"`, in the vocabulary.

```text
("▁hug", 10), ("▁pug", 5), ("▁pun", 12), ("▁bun", 4), ("▁hugs", 5)
```

2. SentencePiece then applies BPE or Unigram to the text.

At decoding, SentencePiece concatenates all tokens and replaces `"▁"` with a space.

## WordPiece

<Youtube id="qpv6ms_t_1A"/>

[WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) is the tokenization algorithm for BERT-family models like [DistilBERT](./model_doc/distilbert) and [Electra](./model_doc/electra).

It's similar to [BPE](#byte-pair-encoding-bpe) and iteratively merges pairs from the bottom up, but differs in how it selects pairs.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

WordPiece merges pairs that maximize the likelihood of the training data.

```text
score("u", "g") = frequency("ug") / (frequency("u") × frequency("g"))
```

| pair | frequency | score |
|---|---|---|
| `"u"` + `"g"` | 20 | 20 / (36 × 20) = 0.028 |
| `"u"` + `"n"` | 16 | 16 / (36 × 16) = 0.028 |
| `"h"` + `"u"` | 15 | 15 / (15 × 36) = 0.028 |
| `"g"` + `"s"` | 5 | 5 / (20 × 5) = 0.050 |

The score favors merging `"g"` and `"s"` where the combined token appears more often than expected from the individual token frequencies. BPE simply merges whichever pair appears the most. WordPiece measures how *informative* each merge is. Two tokens that appear together far more than chance predicts get merged first.

## Word-level tokenization

<Youtube id="nhJxYji1aho"/>

Word-level tokenization splits text into tokens by space, punctuation, or language-specific rules.

```text
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

Vocabulary size becomes extremely large because every unique word requires its own token, including all variations (`"love"`, `"loving"`, `"loved"`, `"lovingly"`). The resulting embedding matrix is enormous, increasing memory and compute. Words not in the vocabulary map to an `"<unk>"` token, so the model can't handle new words.

## Character-level tokenization

Character-level tokenization splits text into individual characters.

```text
["D", "o", "n", "'", "t", "y", "o", "u", "l", "o", "v", "e"]
```

The vocabulary is small and every word can be represented, so there's no `"<unk>"` problem. But sequences become much longer. A character like `"l"` carries far less meaning than `"love"`, so performance suffers.

## Resources

- [Chapter 6](https://huggingface.co/learn/llm-course/chapter6/1) of the LLM course teaches you how to train a tokenizer from scratch and explains the differences between the BPE, Unigram, and WordPiece algorithms.
