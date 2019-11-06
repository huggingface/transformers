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
@file: bert_example.py
@time: 2019/11/5 11:30 下午
"""
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

###
# tokenizer
###
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim',
                          '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0] * 7 + [1] * 7
position_ids = list(range(14))
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
positions_tensor = torch.tensor([position_ids])
print(tokenized_text)

###
# setup model
###

model: BertModel = BertModel.from_pretrained('bert-base-uncased')
if torch.cuda.is_available():
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    positions_tensor = positions_tensor.to('cuda')
    model.to('cuda')
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors, position_ids=positions_tensor)
    encoded_layers = outputs[0]
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)


###
# predict the mask word
###
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
print("done")




