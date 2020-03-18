# coding=utf-8

import logging

logging.basicConfig(level=logging.INFO)

import tensorflow as tf
import tensorflow_datasets

from transformers import TFTrainer, BertForSequenceClassification, BertTokenizer


# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False
EPOCHS = 3

TASK = "mrpc"

if TASK == "sst-2":
    TFDS_TASK = "sst2"
elif TASK == "sts-b":
    TFDS_TASK = "stsb"
else:
    TFDS_TASK = TASK

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# Load dataset via TensorFlow Datasets
data = tensorflow_datasets.load(f"glue/{TFDS_TASK}")

kwargs = {
    "pretrained_model_name_or_path": "bert-base-cased",
    "optimizer_name": "adam",
    "learning_rate": 3e-5,
    "epsilon": 1e-08,
    "loss_name": "sparse_categorical_crossentropy",
    "batch_size": BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "distributed": False,
    "epochs": EPOCHS,
    "data_processor_name": "glue",
    "task": TASK,
    "architecture": "TFBertForSequenceClassification",
    "max_len": 128,
    "metric_name": "sparse_categorical_accuracy",
    "max_grad_norm": 1.0
}

trainer = TFTrainer(**kwargs)
trainer.setup_training("checkpoints", "logs", training_data=data["train"], validation_data=data["validation"])
trainer.train()
trainer.save_model("save")
trainer.evaluate()

if TASK == "mrpc":
    # Load the TensorFlow model in PyTorch for inspection
    # This is to demo the interoperability between the two frameworks, you don't have to
    # do this in real life (you can run the inference on the TF model).
    pytorch_model = BertForSequenceClassification.from_pretrained("./save/", from_tf=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
    sentence_0 = "This research was consistent with his findings."
    sentence_1 = "His findings were compatible with this research."
    sentence_2 = "His findings were not compatible with this research."
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors="pt")
    inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors="pt")

    pred_1 = pytorch_model(**inputs_1)[0].argmax().item()
    pred_2 = pytorch_model(**inputs_2)[0].argmax().item()
    print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
    print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")
