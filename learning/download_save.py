# -*- coding: utf-8 -*-
# @Time    : 10/12/19 5:30 PM
# @Author  : hujunchao
# @Email   : hujunchao@163.com
# @File    : download_save.py
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import os

save_path_tokenizer = os.path.join('model', 'distilbert-base-uncased-tokenizer')
save_path_model = os.path.join('model', 'distilbert-base-uncased-model')

if not os.path.exists(save_path_tokenizer):
    os.makedirs(save_path_tokenizer)

if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# tokenizer.save_pretrained(save_path_tokenizer)
#
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# model.save_pretrained(save_path_model)