<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

>  Transformers-CFG is a fork of the Hugging Face Transformers library adding support for CFG-based Grammar-Constrained Generation methods.
>  The library is trying to keep up-to-date with the main branch of the 🤗 Transformers library.
>  The library tries to offer a compatible interface to llama-cpp project.

## Installation

```bash
pip install git+https://github.com/saibo-creator/transformers-CFG@dev
```


### QuickStart: Force LLM to generate a valid json object

The below example can be found in `examples/pytorch/text-geenration/grammar_constrained_generation.py`

```python

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.grammar_utils import IncrementalGrammarConstraint
from transformers.generation.logits_process import GrammarConstrainedLogitsProcessor


if __name__ == "__main__":

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Load json grammar
    with open("examples/grammars/json.gbnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)


    # Generate
    prefix1 = "This is a valid json string for http request:"
    prefix2 = "This is a valid json string for shopping cart:"
    input_ids = tokenizer([prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_length=50,
        num_beams=2,
        logits_processor=[grammar_processor],
        repetition_penalty=5.0,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for http request:{ "request": { "method": "GET", "headers": [], "content": "Content","type": "application" }}
    'This is a valid json string for shopping cart:This is a valid json string for shopping cart:{ "name": "MyCart", "price": 0, "value": 1 }
    """

```


## Why should I use transformers-CFG?

- We offer the same grammar interface as llama-cpp project, allowing you to drop-in replace llama-cpp with transformers-CFG.
- We allow you to use any of the models in the 🤗 Transformers library, including the ones that are not supported by llama-cpp.




