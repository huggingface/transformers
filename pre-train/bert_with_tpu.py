# encoding: utf-8
# Copyright 2019 The DeepNlp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
@file: bert_with_tpu.py
@time: 2019/11/10 7:07 上午
"""

import sys, os
import tensorflow as tf
import logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

GOOGLE_CLOUD_PROJECT_NAME = "pre-train-bert-sogou" #@param {type: "string" }
BUCKET_NAME = "bert-sogou-pretrain"  #@param {type: "string"}
BASE_MODEL_DIR = "fine_tuning/base_model" #@param {type: "string"}
NEW_MODEL_DIR = "fine_tuning/model" #@param {type: "string"}
MODEL_NAME = "chinese_L-12_H-768_A-12" #@param {type: "string"}
INPUT_DATA_DIR = "fine_tuning/data/zh_wiki_news_2016" #@param {type: "string"}

PROCESSES = 4 #@param {type: "integer"}
DO_LOWER_CASE = True
MAX_SEQ_LENGTH = 128 #@param {type : "integer"}
MASKED_LM_PROB = 0.15 #@param {type: "number" }
# xxxx
MAX_PREDICTIONS = 20 #@param {type: "integer"



#! gcloud config set project pre-train-bert-sogou
base_model_name = "gs://{}/{}/{}".format(BUCKET_NAME, BASE_MODEL_DIR, MODEL_NAME)
fine_tuning_name = "gs://{}/{}/{}".format(BUCKET_NAME, NEW_MODEL_DIR, MODEL_NAME)
#! gsutil rm -rf $fine_tuning_name
#! gsutil cp -r $base_model_name $fine_tuning_name

#
# storage_client = storage.Client()
# bucket = storage_client.get_bucket(BUCKET_NAME)

VOC_FNAME = "gs://{}/{}/{}/vocab.txt".format(BUCKET_NAME, NEW_MODEL_DIR, MODEL_NAME)
TF_RECORD_DIR = "gs://{}/{}_tfrecord".format(BUCKET_NAME, INPUT_DATA_DIR)

file_partitions = [[]]
index = 0


# def list_files(bucketFolder):
#     """List all files in GCP bucket."""
#     files = bucket.list_blobs(prefix=bucketFolder, max_results=1000)
#     fileList = [file.name for file in files]
#     return fileList


print(INPUT_DATA_DIR)
print(TF_RECORD_DIR)
FULL_INPUT_DATA_DIR = "gs://{}/{}".format(BUCKET_NAME, INPUT_DATA_DIR)
# ! gsutil ls $FULL_INPUT_DATA_DIR


# for filename in list_files(INPUT_DATA_DIR):
#     if filename.find("tf") != -1 or filename.endswith("/"):
#         continue
#     if len(file_partitions[index]) == PROCESSES:
#         file_partitions.append([])
#         index += 1
#     file_partitions[index].append("gs://{}/{}".format(BUCKET_NAME, filename))

#! gsutil
#mkdir $TF_RECORD_DIR
#! gsutil
#mkdir
#gs: // bert - sogou - pretrain / fine_tuning / data / zh_wiki_news_2016_tfrecord
#              gs://bert-sogou-pretrain/fine_tuning/data/zh_wiki_news_2016_tfrecord
#! gsutil
#ls $TF_RECORD_DIR

# index = 0
# for partition in file_partitions:
#
#     for filename in partition:
#         print(filename, "----", index)
#     index += 1
#
#     XARGS_CMD = ("gsutil ls {} | "
#                  "awk 'BEGIN{{FS=\"/\"}}{{print $NF}}' | "
#                  "xargs -n 1 -P {} -I{} "
#                  "python3 bert/create_pretraining_data.py "
#                  "--input_file=gs://{}/{}/{} "
#                  "--output_file={}/{}.tfrecord "
#                  "--vocab_file={} "
#                  "--do_lower_case={} "
#                  "--max_predictions_per_seq={} "
#                  "--max_seq_length={} "
#                  "--masked_lm_prob={} "
#                  "--random_seed=34 "
#                  "--dupe_factor=5")
#
#     XARGS_CMD = XARGS_CMD.format(" ".join(partition),
#                                  PROCESSES, '{}', BUCKET_NAME, INPUT_DATA_DIR, '{}',
#                                  TF_RECORD_DIR, '{}',
#                                  VOC_FNAME, DO_LOWER_CASE,
#                                  MAX_PREDICTIONS, MAX_SEQ_LENGTH, MASKED_LM_PROB)
#
#     print(XARGS_CMD)
#
#     # ! $XARGS_CMD
#
#     if index == 2:
#         break
from bert import modeling, optimization, tokenization

# Input data pipeline config
TRAIN_BATCH_SIZE = 128 #@param {type:"integer"}
MAX_PREDICTIONS = 20 #@param {type:"integer"}
MAX_SEQ_LENGTH = 128 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param

# Training procedure config
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
TRAIN_STEPS = 1000000 #@param {type:"integer"}
SAVE_CHECKPOINTS_STEPS = 250 #@param {type:"integer"}
NUM_TPU_CORES = 8



BERT_GCS_DIR = fine_tuning_name+"_latest/"
#BERT_GCS_DIR = fine_tuning_name+"_without_pretrain/"

#! gsutil mkdir $BERT_GCS_DIR

DATA_GCS_DIR = TF_RECORD_DIR

VOCAB_FILE = VOC_FNAME

CONFIG_FILE = "gs://{}/{}/{}/bert_config.json".format(BUCKET_NAME, BASE_MODEL_DIR, MODEL_NAME)


#! gsutil ls $BERT_GCS_DIR

INIT_CHECKPOINT = "{}/bert_model.ckpt".format(base_model_name)
#"gs://bert-sogou-pretrain/fine_tuning/base_model/chinese_L-12_H-768_A-12/bert_model.ckpt"
TMP_INIT_CHECKPOINT = tf.train.latest_checkpoint(BERT_GCS_DIR)
if TMP_INIT_CHECKPOINT is not None:
    INIT_CHECKPOINT = TMP_INIT_CHECKPOINT


bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
input_files = tf.gfile.Glob(os.path.join(DATA_GCS_DIR,'*tfrecord'))

log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))

log.info("Using {} data shards".format(len(input_files)))

#! gsutil ls $INIT_CHECKPOINT*

#INIT_CHECKPOINT = None

import sys

sys.path.append("bert")
from bert.run_pretraining import input_fn_builder, model_fn_builder
from bert import modeling, optimization, tokenization
USE_TPU=True
TPU_ADDRESS = "taey2113"
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

estimator.train(input_fn=train_input_fn, max_steps=TRAIN_STEPS)




