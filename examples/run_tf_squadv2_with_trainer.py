# coding=utf-8

import os
import logging

logging.basicConfig(level=logging.INFO)

import tensorflow as tf
import tensorflow_datasets

import torch

from transformers import TFTrainer, BertForQuestionAnswering, BertTokenizer

kwargs = {
    "pretrained_model_name_or_path": "bert-base-cased",
    "optimizer_name": "adamw",
    "learning_rate": 3e-5,
    "epsilon": 1e-08,
    "loss_name": "squad_loss",
    "batch_size": 8,
    "eval_batch_size": 8,
    "distributed": True,
    "epochs": 2,
    "data_processor_name": "squadv2",
    "architecture": "TFBertForQuestionAnswering",
    "max_len": 384,
    "max_grad_norm": 1.0,
    "doc_stride": 128,
    "max_query_len": 64
}

trainer = TFTrainer(**kwargs)
trainer.setup_training("checkpoints", "logs", data_dir="./data/squadv2")
trainer.train()
trainer.save_model("save")

pytorch_model = BertForQuestionAnswering.from_pretrained("./save", from_tf=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
input_ids = tokenizer.encode(question, text)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
start_scores, end_scores = pytorch_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
print(answer)