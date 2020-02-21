# coding=utf-8

import os
import logging

import tensorflow as tf
import numpy as np

from transformers import TFTrainer, TFBertForQuestionAnswering, BertTokenizer, BertForSequenceClassification


# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False
EPOCHS = 2

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})
logging.getLogger("transformers.trainer_tf").setLevel(logging.INFO)

kwargs = {
    "pretrained_model_name_or_path": "bert-base-multilingual-cased",
    "optimizer_name": "adamw",
    "learning_rate": 5e-5,
    "epsilon": 1e-08,
    "loss_name": "sparse_categorical_crossentropy",
    "batch_size": BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "distributed": False,
    "epochs": EPOCHS,
    "data_processor_name": "xnli",
    "architecture": "TFBertForSequenceClassification",
    "max_len": 128,
    "language": "de",
    "train_language": "en",
    "task": "xnli",
    "max_grad_norm": 1.0,
    "metric_name": "sparse_categorical_accuracy"
}

trainer = TFTrainer(**kwargs)
trainer.setup_training("checkpoints", "logs", data_dir="data")
trainer.train()
trainer.save_model("save")

pytorch_model = BertForSequenceClassification.from_pretrained("./save/", from_tf=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
sentence_0 = "Auf der anderen Seite haben wir vieleWaschbären, Opossums ud Schildkröten gegessen haben."
sentence_1 = "Ich habe ungewöhnliche Tiere wie Schildkröte, Waschbär und Opossum gegessen."
inputs = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors="pt")
pred = pytorch_model(**inputs)[0].argmax().item()

print("sentence_1 is", pytorch_model.config.id2label[pred], "of sentence_0")