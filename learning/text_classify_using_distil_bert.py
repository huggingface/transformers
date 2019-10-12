# -*- coding: utf-8 -*-
# @Time    : 10/12/19 5:44 PM
# @Author  : hujunchao
# @Email   : hujunchao@163.com
# @File    : text_classify_using_distil_bert.py
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification
import torch
from download_save import save_path_model, save_path_tokenizer

tokenizer = DistilBertTokenizer.from_pretrained(save_path_tokenizer)
model = DistilBertForSequenceClassification.from_pretrained(save_path_model)

encoded = tokenizer.encode('hello world, my name is Tom')
encoded = torch.tensor(encoded).unsqueeze(dim=0)
label = torch.tensor([1]).unsqueeze(dim=0)
result = model(encoded, labels=label)
print()