import sys
import numpy as np
import jsonlines as jsonl
from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel
import tensorflow as tf
from tensorflow.keras import metrics 


def get_dataset(fil):
    data = []
    with jsonl.open(fil) as reader:
        for line in reader:
            data.append(line['text'])
    return data

if len(sys.argv) == 1:
    model_size = "Small"
    data_dir = '/dockerx/data/tf-gpt-2/data/'
    num_epochs = 1
    num_gpus = len(tf.config.list_physical_devices(device_type='GPU'))
    truncate = True
else:
    model_size = sys.argv[1]
    data_dir = sys.argv[2]
    num_epochs = int(sys.argv[3])
    num_gpus = int(sys.argv[4])
    if int(sys.argv[5]) == 1:
        truncate = True
    else:
        truncate = False

if model_size == "Small":
    model_name = "gpt2"
    train_file = data_dir+'small-117M-k40.train.jsonl'
    valid_file = data_dir+'small-117M-k40.valid.jsonl'
elif model_size == "Medium":
    model_name = "gpt2-medium"
    train_file = data_dir+'medium-345M-k40.train.jsonl'
    valid_file = data_dir+'medium-345M-k40.valid.jsonl'
elif model_size == "Large":
    model_name = "gpt2-large"
    train_file = data_dir+'large-762M-k40.train.jsonl'
    valid_file = data_dir+'large-762M-k40.valid.jsonl'
elif model_size == "XL":
    model_name = 'gpt2-xl'
    train_file = data_dir+'xl-1542M-k40.train.jsonl'
    valid_file = data_dir+'xl-1542M-k40.valid.jsonl'
print("Finetuning model " + model_name)
print("With dataset "+train_file)

def tokenize(data, tokenizer, truncate=False):
    if truncate:
        data = tokenizer(data[:1000], return_tensors='tf', padding=True, truncation=True)
    else:
        data = tokenizer(data, return_tensors='tf', padding=True, truncation=True)
    return tf.data.Dataset.from_tensor_slices((dict(data), data['input_ids']))

print("============================ Creating Distributed Strategy ===========================")
devices = []
for i in range(num_gpus):
    devices.append("GPU:"+str(i))
strategy = tf.distribute.MirroredStrategy(devices=devices)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
print("============================ Loading model from pretrained and compiling ===========================")
with strategy.scope():
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("========================= Loading dataset ========================")
    train_dataset = tokenize(get_dataset(train_file),tokenizer, truncate).batch(num_gpus)
    valid_dataset = tokenize(get_dataset(valid_file),tokenizer, truncate).batch(num_gpus)
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    #Disable past key values
    model.config.use_cache=False
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = metrics.SparseCategoricalAccuracy(name='Accuracy')
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
print("========================= Finetuning Model ==================================")
model.fit(train_dataset, batch_size=64, epochs=num_epochs)
print("========================= Evaluating Model ==================================")
model.evaluate(valid_dataset)

