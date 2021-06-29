import sys

import numpy as np
import jsonlines as jsonl
from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel
import tensorflow as tf
from tensorflow.keras import metrics 

BATCH_SIZE=1

def get_dataset(fil):
    data = []
    with jsonl.open(fil) as reader:
        for line in reader:
            data.append(line['text'])
    return data

if len(sys.argv) == 1:
    model_size = "Small"
    data_dir = '/dockerx/data/'
else:
    model_size = sys.argv[1]
    data_dir = sys.argv[2]

if model_size == "Small":
    model_name = "gpt2"
    train_file = data_dir+'small-117M.train.jsonl'
    test_file = data_dir+'small-117M.test.jsonl'
elif model_size == "Medium":
    model_name = "gpt2-medium"
    train_file = data_dir+'medium-345M.train.jsonl'
    test_file = data_dir+'medium-345M.test.jsonl'
elif model_size == "Large":
    model_name = "gpt2-large"
    train_file = data_dir+'large-762M.train.jsonl'
    test_file = data_dir+'large-762M.test.jsonl'
elif model_size == "XL":
    model_name = 'gpt2-xl'
    train_file = data_dir+'xl-1542M.train.jsonl'
    test_file = data_dir+'xl-1542M.test.jsonl'
print("Profiling model " + model_name)

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
def tokenize(data):
    data = tokenizer(data[0], return_tensors='tf', padding=True, truncation=True)
    return tf.data.Dataset.from_tensor_slices((dict(data), data['input_ids']))

train_dataset = tokenize(get_dataset(train_file)).batch(BATCH_SIZE)
model = TFGPT2LMHeadModel.from_pretrained(model_name)
#Supresses the past_key_values from being expressed in the progress bar
model.config.use_cache=False
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = metrics.SparseCategoricalAccuracy(name='Accuracy')
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer])
model.fit(train_dataset, batch_size=1, epochs=1)

