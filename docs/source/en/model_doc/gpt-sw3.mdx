<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# GPT-Sw3

## Overview

The GPT-Sw3 model was first proposed in
[Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf)
by Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman,
Fredrik Carlsson, Magnus Sahlgren.

Since that first paper the authors have extended their work and trained new models on their new 1.2TB corpora named The Nordic Pile.

GPT-Sw3 is a collection of large decoder-only pretrained transformer language models that were developed by AI Sweden
in collaboration with RISE and the WASP WARA for Media and Language. GPT-Sw3 has been trained on a dataset containing
320B tokens in Swedish, Norwegian, Danish, Icelandic, English, and programming code. The model was pretrained using a
causal language modeling (CLM) objective utilizing the NeMo Megatron GPT implementation.

This model was contributed by [AI Sweden](https://huggingface.co/AI-Sweden).

The implementation uses the [GPT2Model](https://huggingface.co/docs/transformers/model_doc/gpt2) coupled
with our `GPTSw3Tokenizer`. This means that `AutoTokenizer` and `AutoModelForCausalLM` map to our tokenizer
implementation and the corresponding GPT2 model implementation respectively.
*Note that sentencepiece is required to use our tokenizer and can be installed with:* `pip install transformers[sentencepiece]` or `pip install sentencepiece`

Example usage:
```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-356m")
>>> model = AutoModelForCausalLM.from_pretrained("AI-Sweden/gpt-sw3-356m")

>>> input_ids = tokenizer("Träd är fina för att", return_tensors="pt")["input_ids"]

>>> generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=10, do_sample=True)[0]

>>> print(tokenizer.decode(generated_token_ids))
Träd är fina för att de är färgstarka. Men ibland är det fint
```

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## GPTSw3Tokenizer

[[autodoc]] GPTSw3Tokenizer
    - save_vocabulary
