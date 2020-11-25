
# Training the QARiB model from scratch

## Step 0: Collect corpora

## Step 1: Vocab generation
We use the unsupervised text tokenizer SentencePiece (https://github.com/google/sentencepiece) 
to create a BERT-compatible vocab.

The vocab is created on random 5M sentence/tweets from training corpus.
```bash
python prepare_vocab.py -i data.txt -s 64000
```

Split the preprocessed training corpus into 512K shards using `split -l 512000`.
## Step 2: Split data into shards

```bash
split -a 4 -l 512000  -d data.txt  ./shards/shard_
```

## Step 3: Convert shards
Build pretraining data by converting the raw shards into TFRecords. 
```bash
ls ./shards/  | xargs -n 1 -P 2 -I{}  ./create_pretrain.sh {}
```

## Step 4: Upload data to GStorage
Upload TFRecords pretraining data to Google Bucket.
**Notice**: You must create a Google Storage Bucket first. Please also make
sure that the service user (e.g. `service-<id>@cloud-tpu.iam.gserviceaccount.com`)
has "Storage Administrator" permissions in order to write files to the bucket.
```bash
gsutil -m cp -r pretraining_data gs://<bucket>/
```

## Step 5: Login to Google Colab 
First and foremost, we get the packages required to train the model. The Jupyter environment allows executing bash commands directly from the notebook by using an exclamation mark ‘!’. I will be exploiting this approach to make use of several other bash commands throughout the experiment.

Now, let’s import the packages and authorize ourselves in Google Cloud.
```bash
pip install sentencepiece
pip install tensorflow==1.15
```
Start TPU Session
```python

import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import sentencepiece as spm

from glob import glob
from google.colab import auth, drive
from tensorflow.keras.utils import Progbar

sys.path.append("./bert")
sys.path.append("./bert/repo")

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

auth.authenticate_user()

# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
log.handlers = [sh]

if 'COLAB_TPU_ADDR' in os.environ:
  log.info("Using TPU runtime")
  USE_TPU = True
  TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']

  with tf.Session(TPU_ADDRESS) as session:
    log.info('TPU address is ' + TPU_ADDRESS)
    # Upload credentials to TPU.
    with open('/content/adc.json', 'r') as f:
      auth_info = json.load(f)
    tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
else:
  log.warning('Not connected to TPU runtime')
  USE_TPU = False
```
model configuration
We are almost ready to begin training our model. If you wish to continue an interrupted training run, you may skip steps 2-6 and proceed from here.

Make sure that you have set the BUCKET_NAME here as well.
```python
BUCKET_NAME = "data" #@param {type:"string"}
MODEL_DIR = "bert_model" #@param {type:"string"}
PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
VOC_FNAME = "data.txt.vocab" #@param {type:"string"}
OUTPUT_DIR = "bert_evals"
VOC_SIZE =  64000#@param {type:"integer"}

# Input data pipeline config
TRAIN_BATCH_SIZE = 64 #@param {type:"integer"}
MAX_PREDICTIONS =  20#@param {type:"integer"}
MAX_SEQ_LENGTH = 64 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param
MAX_EVAL_STEPS = 100 #@param {type:"integer"}

# Training procedure config
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 0.8e-5 #1.0e-5 # 1.5e-5
TRAIN_STEPS = 2000000 #@param {type:"integer"}
SAVE_CHECKPOINTS_STEPS =  100000 #@param {type:"integer"}
NUM_TPU_CORES = 8

if BUCKET_NAME:
  BUCKET_PATH = "gs://{}".format(BUCKET_NAME)
else:
  BUCKET_PATH = "."

BERT_GCS_DIR = "{}/{}".format(BUCKET_PATH, MODEL_DIR)
DATA_GCS_DIR = "{}/{}".format(BUCKET_PATH, PRETRAINING_DIR)
EVAL_OUTPUT_DIR = "{}/{}".format(BUCKET_PATH, OUTPUT_DIR)

VOCAB_FILE = os.path.join(BUCKET_PATH, VOC_FNAME)
CONFIG_FILE = os.path.join(BUCKET_PATH, "bert_config.json")

INIT_CHECKPOINT = tf.train.latest_checkpoint(BERT_GCS_DIR)

bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
input_files = tf.gfile.Glob(os.path.join(DATA_GCS_DIR,'*tfrecord'))

log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))
log.info("Using {} data shards".format(len(input_files)))
```

BERT Configuration
```python
model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE,
      num_train_steps=TRAIN_STEPS,
      num_warmup_steps=10,
      use_tpu=USE_TPU,
      use_one_hot_embeddings=True)

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=BERT_GCS_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    keep_checkpoint_max=20, # keep all checkpoints otherwise, n 
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=SAVE_CHECKPOINTS_STEPS,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=MAX_SEQ_LENGTH,
        max_predictions_per_seq=MAX_PREDICTIONS,
        is_training=True)
```

## Step 6: Fire training 🔥

```python
estimator.train(input_fn=train_input_fn, max_steps=TRAIN_STEPS)
```

This will train the models for 2M steps (See TRAIN_STEPS). Checkpoints are saved
after 100k steps (SAVE_CHECKPOINTS_STEPS). The last 20 checkpoints will be kept (keep_checkpoint_max=20).