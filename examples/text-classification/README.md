<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## GLUE Benchmark

# Run TensorFlow 2.0 version

Based on the script [`run_tf_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py).

Fine-tuning the library TensorFlow 2.0 Bert model for sequence classification on the  MRPC task of the GLUE benchmark: [General Language Understanding Evaluation](https://gluebenchmark.com/).

This script has an option for mixed precision (Automatic Mixed Precision / AMP) to run models on Tensor Cores (NVIDIA Volta/Turing GPUs) and future hardware and an option for XLA, which uses the XLA compiler to reduce model runtime.
Options are toggled using `USE_XLA` or `USE_AMP` variables in the script.
These options and the below benchmark are provided by @tlkh.

Quick benchmarks from the script (no other modifications):

| GPU    | Mode | Time (2nd epoch) | Val Acc (3 runs) |
| --------- | -------- | ----------------------- | ----------------------|
| Titan V | FP32 | 41s | 0.8438/0.8281/0.8333 |
| Titan V | AMP | 26s | 0.8281/0.8568/0.8411 |
| V100    | FP32 | 35s | 0.8646/0.8359/0.8464 |
| V100    | AMP | 22s | 0.8646/0.8385/0.8411 |
| 1080 Ti | FP32 | 55s | - |

Mixed precision (AMP) reduces the training time considerably for the same hardware and hyper-parameters (same batch size was used).


## Run generic text classification script in TensorFlow

The script [run_tf_text_classification.py](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_text_classification.py) allows users to run a text classification on their own CSV files. For now there are few restrictions, the CSV files must have a header corresponding to the column names and not more than three columns: one column for the id, one column for the text and another column for a second piece of text in case of an entailment classification for example.

To use the script, one as to run the following command line:
```bash
python run_tf_text_classification.py \
  --train_file train.csv \ ### training dataset file location (mandatory if running with --do_train option)
  --dev_file dev.csv \ ### development dataset file location (mandatory if running with --do_eval option)
  --test_file test.csv \ ### test dataset file location (mandatory if running with --do_predict option)
  --label_column_id 0 \ ### which column corresponds to the labels
  --model_name_or_path bert-base-multilingual-uncased \
  --output_dir model \
  --num_train_epochs 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --save_steps 10 \
  --overwrite_output_dir \
  --max_seq_length 128
```

# Run PyTorch version

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune the following models: BERT, XLM, XLNet and RoBERTa.

GLUE is made up of a total of 9 different tasks. We get the following results on the dev set of the benchmark with an
uncased  BERT base model (the checkpoint `bert-base-uncased`). All experiments ran single V100 GPUs with a total train
batch sizes between 16 and 64. Some of these tasks have a small dataset and training can lead to high variance in the results
between different runs. We report the median on 5 runs (with different seeds) for each of the metrics.

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthew's corr               | 49.23       |
| SST-2 | Accuracy                     | 91.97       |
| MRPC  | F1/Accuracy                  | 89.47/85.29 |
| STS-B | Person/Spearman corr.        | 83.95/83.70 |
| QQP   | Accuracy/F1                  | 88.40/84.31 |
| MNLI  | Matched acc./Mismatched acc. | 80.61/81.08 |
| QNLI  | Accuracy                     | 87.46       |
| RTE   | Accuracy                     | 61.73       |
| WNLI  | Accuracy                     | 45.07       |

Some of these results are significantly different from the ones reported on the test set
of GLUE benchmark on the website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the
website.

```bash
export TASK_NAME=MRPC

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file `eval_results.txt` in the specified output_dir.
In case of MNLI, since there are two separate dev sets (matched and mismatched), there will be a separate
output folder called `/tmp/MNLI-MM/` in addition to `/tmp/MNLI/`.

The code has not been tested with half-precision training with apex on any GLUE task apart from MRPC, MNLI,
CoLA, SST-2. The following section provides details on how to run half-precision training with MRPC. With that being
said, there shouldn’t be any issues in running half-precision training with the remaining GLUE tasks as well,
since the data processor for each task inherits from the base class DataProcessor.

## Running on TPUs in PyTorch

Even when running PyTorch, you can accelerate your workloads on Google's TPUs, using `pytorch/xla`. For information on
how to setup your TPU environment refer to the
[pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

For running your GLUE task on MNLI dataset you can run something like the following form the root of the transformers
repo:

```
python examples/xla_spawn.py \
  --num_cores=8 \
  transformers/examples/text-classification/run_glue.py \
  --do_train \
  --do_eval \
  --task_name=mrpc \
  --num_train_epochs=3 \
  --max_seq_length=128 \
  --learning_rate=5e-5 \
  --output_dir=/tmp/mrpc \
  --overwrite_output_dir \
  --logging_steps=5 \
  --save_steps=5 \
  --tpu_metrics_debug \
  --model_name_or_path=bert-base-cased \
  --per_device_train_batch_size=64 \
  --per_device_eval_batch_size=64
```


#### Using Apex and mixed-precision

Using Apex and 16 bit precision, the fine-tuning on MRPC only takes 27 seconds. First install
[apex](https://github.com/NVIDIA/apex), then run the following example:

```bash

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/ \
  --fp16
```

#### Distributed training

Here is an example using distributed training on 8 V100 GPUs. The model used is the BERT whole-word-masking and it
reaches F1 > 92 on MRPC.

```bash

python -m torch.distributed.launch \
    --nproc_per_node 8 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mrpc_output/
```

Training with these hyper-parameters gave us the following results:

```bash
acc = 0.8823529411764706
acc_and_f1 = 0.901702786377709
eval_loss = 0.3418912578906332
f1 = 0.9210526315789473
global_step = 174
loss = 0.07231863956341798
```

### MNLI

The following example uses the BERT-large, uncased, whole-word-masking model and fine-tunes it on the MNLI task.

```bash
export GLUE_DIR=/path/to/glue

python -m torch.distributed.launch \
    --nproc_per_node 8 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output_dir \
```

The results  are the following:

```bash
***** Eval results *****
  acc = 0.8679706601466992
  eval_loss = 0.4911287787382479
  global_step = 18408
  loss = 0.04755385363816904

***** Eval results *****
  acc = 0.8747965825874695
  eval_loss = 0.45516540421714036
  global_step = 18408
  loss = 0.04755385363816904
```

# Run PyTorch version using PyTorch-Lightning

Run `bash run_pl.sh` from the `glue` directory. This will also install `pytorch-lightning` and the requirements in
`examples/requirements.txt`. It is a shell pipeline that will automatically download, preprocess the data and run the
specified models. Logs are saved in `lightning_logs` directory.

Pass `--gpus` flag to change the number of GPUs. Default uses 1. At the end, the expected results are:

```
TEST RESULTS {'val_loss': tensor(0.0707), 'precision': 0.852427800698191, 'recall': 0.869537067011978, 'f1': 0.8608974358974358}
```


# XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_xnli.py).

[XNLI](https://www.nyu.edu/projects/bowman/xnli/) is a crowd-sourced dataset based on [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/). It is an evaluation benchmark for cross-lingual text representations. Pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

#### Fine-tuning on XNLI

This example code fine-tunes mBERT (multi-lingual BERT) on the XNLI dataset. It runs in 106 mins
on a single tesla V100 16GB. The data for XNLI can be downloaded with the following links and should be both saved (and un-zipped) in a
`$XNLI_DIR` directory.

* [XNLI 1.0](https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip)
* [XNLI-MT 1.0](https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip)

```bash
export XNLI_DIR=/path/to/XNLI

python run_xnli.py \
  --model_name_or_path bert-base-multilingual-cased \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --data_dir $XNLI_DIR \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_xnli/ \
  --save_steps -1
```

Training with the previously defined hyper-parameters yields the following results on the **test** set:

```bash
acc = 0.7093812375249501
```
