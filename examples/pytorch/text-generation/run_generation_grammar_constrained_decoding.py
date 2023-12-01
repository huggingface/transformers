#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 University of Cambridge, Tencent AI Lab, DeepMind and The University of Hong Kong Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.grammar_utils import IncrementalGrammarConstraint
from transformers.generation.logits_process import GrammarConstrainedLogitsProcessor


if __name__ == "__main__":

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Load grammar
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
