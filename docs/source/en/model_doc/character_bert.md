<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-10-20 and added to Hugging Face Transformers on 2026-03-01.*

# CharacterBERT

## Overview

CharacterBERT was proposed in [CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary
Representations From Characters][characterbert-paper] by Hicham El Boukkouri, Olivier Ferret, Thomas Lavergne, and
Pierre Zweigenbaum.

CharacterBERT replaces BERT wordpiece embeddings with a character CNN that produces token-level representations from
bytes/characters. This enables open-vocabulary token handling without subword vocabularies while keeping the BERT
encoder stack.

Tips:

- CharacterBERT is token-level but computes token representations from characters/bytes internally.
- Inputs use character IDs per token, so `input_ids` has shape `(batch_size, sequence_length, max_characters_per_token)`.
- The byte-based encoding scheme supports open-vocabulary text without a fixed wordpiece vocabulary.

## Usage example

```python
>>> from transformers import CharacterBertModel, CharacterBertTokenizer

>>> tokenizer = CharacterBertTokenizer()
>>> model = CharacterBertModel.from_pretrained("helboukkouri/character-bert-base-uncased")

>>> inputs = tokenizer("CharacterBERT handles typoooos better", return_tensors="pt")
>>> outputs = model(**inputs)
>>> outputs.last_hidden_state.shape
```

## Masked LM example

CharacterBERT masked-LM logits are indexed with `mlm_vocab.txt` from the checkpoint.

```python
>>> from pathlib import Path
>>> import torch
>>> from huggingface_hub import hf_hub_download
>>> from transformers import AutoModelForMaskedLM, AutoTokenizer

>>> model_id = "helboukkouri/character-bert-base-uncased"
>>> tokenizer = AutoTokenizer.from_pretrained(model_id)
>>> model = AutoModelForMaskedLM.from_pretrained(model_id).eval()

>>> inputs = tokenizer("paris is the capital of [MASK].", return_tensors="pt")
>>> tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
>>> mask_index = tokens.index(tokenizer.mask_token)

>>> with torch.no_grad():
...     probs = model(**inputs).logits[0, mask_index].softmax(dim=-1)

>>> mlm_vocab_path = Path(hf_hub_download(repo_id=model_id, filename="mlm_vocab.txt"))
>>> mlm_vocab = mlm_vocab_path.read_text(encoding="utf-8").splitlines()
>>> top_probs, top_indices = torch.topk(probs, k=5)
>>> [(mlm_vocab[i], round(float(p), 4)) for p, i in zip(top_probs, top_indices)]
```

`pipeline("fill-mask")` is currently not supported for CharacterBERT because the model input format and MLM output
vocabulary differ from standard subword tokenizers.

## CharacterBertConfig

[[autodoc]] CharacterBertConfig

## CharacterBertTokenizer

[[autodoc]] CharacterBertTokenizer

## CharacterBertModel

[[autodoc]] CharacterBertModel
    - forward

## CharacterBertForMaskedLM

[[autodoc]] CharacterBertForMaskedLM
    - forward

## CharacterBertForSequenceClassification

[[autodoc]] CharacterBertForSequenceClassification
    - forward

## CharacterBertForTokenClassification

[[autodoc]] CharacterBertForTokenClassification
    - forward

## CharacterBertForQuestionAnswering

[[autodoc]] CharacterBertForQuestionAnswering
    - forward

[characterbert-paper]: https://aclanthology.org/2020.coling-main.609/
