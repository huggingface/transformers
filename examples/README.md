# Examples

In this section a few examples are put together. All of these examples work for several models, making use of the very
similar API between the different models.

| Section                    | Description                                                                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Language Model fine-tuning](#language-model-fine-tuning) | Fine-tuning the library models for language modeling on a text dataset. Causal language modeling for GPT/GPT-2, masked language modeling for BERT/RoBERTa. |
| [Language Generation](#language-generation) | Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL and XLNet.                                         |
| [GLUE](#glue) | Examples running BERT/XLM/XLNet/RoBERTa on the 9 GLUE tasks. Examples feature distributed training as well as half-precision.                              |
| [SQuAD](#squad) | Using BERT for question answering, examples with distributed training.                                                                                  |
| [Multiple Choice](#multiple-choice) | Examples running BERT/XLNet/RoBERTa on the SWAG/RACE/ARC tasks. 

## Language model fine-tuning

Based on the script [`run_lm_finetuning.py`](https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py).

Fine-tuning the library models for language modeling on a text dataset for GPT, GPT-2, BERT and RoBERTa (DistilBERT 
to be added soon). GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa 
are fine-tuned using a masked language modeling (MLM) loss.

Before running the following example, you should get a file that contains text on which the language model will be
fine-tuned. A good example of such text is the [WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

We will refer to two different files: `$TRAIN_FILE`, which contains text for training, and `$TEST_FILE`, which contains
text that will be used for evaluation.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_lm_finetuning.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

### RoBERTa/BERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling. 

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore, converge
slightly slower (over-fitting takes more epochs).

We use the `--mlm` flag so that the script may change its loss function.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_lm_finetuning.py \
    --output_dir=output \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm
```

## Language generation

Based on the script [`run_generation.py`](https://github.com/huggingface/transformers/blob/master/examples/run_generation.py).

Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL and XLNet.
A similar script is used for our official demo [Write With Transfomer](https://transformer.huggingface.co), where you
can try out the different models available in the library.

Example usage:

```bash
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
```

## GLUE

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding 
Evaluation](https://gluebenchmark.com/). This script can fine-tune the following models: BERT, XLM, XLNet and RoBERTa. 

GLUE is made up of a total of 9 different tasks. We get the following results on the dev set of the benchmark with an
uncased  BERT base model (the checkpoint `bert-base-uncased`). All experiments ran on 8 V100 GPUs with a total train
batch size of 24. Some of these tasks have a small dataset and training can lead to high variance in the results
between different runs. We report the median on 5 runs (with different seeds) for each of the metrics.

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthew's corr               | 48.87       |
| SST-2 | Accuracy                     | 91.74       |
| MRPC  | F1/Accuracy                  | 90.70/86.27 |
| STS-B | Person/Spearman corr.        | 91.39/91.04 |
| QQP   | Accuracy/F1                  | 90.79/87.66 |
| MNLI  | Matched acc./Mismatched acc. | 83.70/84.83 |
| QNLI  | Accuracy                     | 89.31       |
| RTE   | Accuracy                     | 71.43       |
| WNLI  | Accuracy                     | 43.66       |

Some of these results are significantly different from the ones reported on the test set
of GLUE benchmark on the website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the webite.

Before running anyone of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
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

### MRPC

#### Fine-tuning example

The following examples fine-tune BERT on the Microsoft Research Paraphrase Corpus (MRPC) corpus and runs in less 
than 10 minutes on a single K-80 and in 27 seconds (!) on single tesla V100 16GB with apex installed.

Before running anyone of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
export GLUE_DIR=/path/to/glue

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
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
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
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
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/MRPC/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
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
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name mnli \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/MNLI/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
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

## Multiple Choice

Based on the script [`run_multiple_choice.py`]().

#### Fine-tuning on SWAG
Download [swag](https://github.com/rowanz/swagaf/tree/master/data) data

```bash
#training on 4 tesla V100(16GB) GPUS
export SWAG_DIR=/path/to/swag_data_dir
python ./examples/run_multiple_choice.py \
--model_type roberta \
--task_name swag \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--do_lower_case \
--data_dir $SWAG_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir models_bert/swag_base \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output
```
Training with the defined hyper-parameters yields the following results:
```
***** Eval results *****
eval_acc = 0.8338998300509847
eval_loss = 0.44457291918821606
```

## SQuAD

Based on the script [`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py).

#### Fine-tuning on SQuAD

This example code fine-tunes BERT on the SQuAD dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large) 
on a single tesla V100 16GB. The data for SQuAD can be downloaded with the following links and should be saved in a 
$SQUAD_DIR directory.

* [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 88.52
exact_match = 81.22
```

#### Distributed training


Here is an example using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD:

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../models/wwm_uncased_finetuned_squad/ \
    --per_gpu_train_batch_size 24 \
    --gradient_accumulation_steps 12
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 93.15
exact_match = 86.91
```

This fine-tuneds model is available as a checkpoint under the reference
`bert-large-uncased-whole-word-masking-finetuned-squad`.

