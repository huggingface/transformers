# coding=utf-8

import os

import tensorflow as tf
import numpy as np

from transformers import TFTrainer, TFBertForQuestionAnswering, BertTokenizer


# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False
EPOCHS = 2

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

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
    "max_grad_norm": 1.0
}

trainer = TFTrainer(**kwargs)
trainer.setup_training("checkpoints", "logs", data_dir="data")
trainer.train()
trainer.save_model("save")
