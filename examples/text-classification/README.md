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
of GLUE benchmark on the website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the webite.

Before running any one of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
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

**Update**: read the more up-to-date [Running on TPUs](../README.md#running-on-tpus) in the main README.md instead.

Even when running PyTorch, you can accelerate your workloads on Google's TPUs, using `pytorch/xla`. For information on how to setup your TPU environment refer to the
[pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

The following are some examples of running the `*_tpu.py` finetuning scripts on TPUs. All steps for data preparation are
identical to your normal GPU + Huggingface setup.

For running your GLUE task on MNLI dataset you can run something like the following:

```
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export GLUE_DIR=/path/to/glue
export TASK_NAME=MNLI

python run_glue_tpu.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir \
  --logging_steps 50 \
  --save_steps 200 \
  --num_cores=8
```

### MRPC

#### Fine-tuning example

The following examples fine-tune BERT on the Microsoft Research Paraphrase Corpus (MRPC) corpus and runs in less
than 10 minutes on a single K-80 and in 27 seconds (!) on single tesla V100 16GB with apex installed.

Before running any one of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
export GLUE_DIR=/path/to/glue

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

Our test ran on a few seeds with [the original implementation hyper-
parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) gave evaluation
results between 84% and 88%.

#### Using Apex and mixed-precision

Using Apex and 16 bit precision, the fine-tuning on MRPC only takes 27 seconds. First install
[apex](https://github.com/NVIDIA/apex), then run the following example:

```bash
export GLUE_DIR=/path/to/glue

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/MRPC/ \
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
export GLUE_DIR=/path/to/glue

python -m torch.distributed.launch \
    --nproc_per_node 8 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/MRPC/ \
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
    --data_dir $GLUE_DIR/MNLI/ \
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

Run `bash run_pl.sh` from the `glue` directory. This will also install `pytorch-lightning` and the requirements in `examples/requirements.txt`. It is a shell pipeline that will automatically download, pre-process the data and run the specified models. Logs are saved in `lightning_logs` directory.

Pass `--n_gpu` flag to change the number of GPUs. Default uses 1. At the end, the expected results are: 

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
* [XNLI-MT 1.0](https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip)

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




