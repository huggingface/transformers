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
    num_epochs = 1
    truncate = True
else:
    model_size = sys.argv[1]
    data_dir = sys.argv[2]
    num_epochs = int(sys.argv[3])
    if int(sys.argv[4]) == 1:
        truncate = True
    else:
        truncate = False

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
print("Finetuning model " + model_name)
print("With dataset "+train_file)

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
def tokenize(data, truncate=False):
    if truncate:
        data = tokenizer(data[:1000], return_tensors='tf', padding=True, truncation=True)
    else:
        data = tokenizer(data, return_tensors='tf', padding=True, truncation=True)
    return tf.data.Dataset.from_tensor_slices((dict(data), data['input_ids']))

print("========================= Loading dataset ========================")
train_dataset = tokenize(get_dataset(train_file), truncate).shuffle(1000).batch(BATCH_SIZE)
test_dataset = tokenize(get_dataset(test_file), truncate).batch(BATCH_SIZE)
print("============================ Loading model from pretrained ===========================")
model = TFGPT2LMHeadModel.from_pretrained(model_name)
#Supresses the past_key_values from being expressed in the progress bar
model.config.use_cache=False
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = metrics.SparseCategoricalAccuracy(name='Accuracy')
print("========================= Compiling Model ============================")
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
print("========================= Finetuning Model ==================================")
model.fit(train_dataset, batch_size=64, epochs=num_epochs)
print("========================= Evaluating Model ==================================")
info = model.evaluate(test_dataset, verbose=2)

