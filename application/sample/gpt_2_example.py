# encoding: utf-8
# Copyright 2019 The DeepNlp Authors.
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
# limitations under the License
"""
@file: gpt_2_example.py
@time: 2019/11/6 12:07 上午
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
logging.basicConfig(level=logging.INFO)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = 'Who was Jim Henson ? Jim Henson was a'
indexed_tokens = tokenizer.encode(text)

token_tensor = torch.tensor([indexed_tokens])

model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

if torch.cuda.is_available():
    model.to('cuda')
    token_tensor = token_tensor.to('cuda')
with torch.no_grad():
    outputs = model(token_tensor)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

