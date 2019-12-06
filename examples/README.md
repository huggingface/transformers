# Examples

In this section a few examples are put together. All of these examples work for several models, making use of the very
similar API between the different models.

**Important**  
To run the latest versions of the examples, you have to install from source and install some specific requirements for the examples.
Execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install [--editable] .
pip install -r ./examples/requirements.txt
```

| Section                    | Description                                                                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [TensorFlow 2.0 models on GLUE](#TensorFlow-2.0-Bert-models-on-GLUE) | Examples running BERT TensorFlow 2.0 model on the GLUE tasks. 
| [Language Model fine-tuning](#language-model-fine-tuning) | Fine-tuning the library models for language modeling on a text dataset. Causal language modeling for GPT/GPT-2, masked language modeling for BERT/RoBERTa. |
| [Language Generation](#language-generation) | Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL and XLNet.                                         |
| [GLUE](#glue) | Examples running BERT/XLM/XLNet/RoBERTa on the 9 GLUE tasks. Examples feature distributed training as well as half-precision.                              |
| [SQuAD](#squad) | Using BERT/RoBERTa/XLNet/XLM for question answering, examples with distributed training.                                                                                  |
| [Multiple Choice](#multiple-choice) | Examples running BERT/XLNet/RoBERTa on the SWAG/RACE/ARC tasks. 
| [Named Entity Recognition](#named-entity-recognition) | Using BERT for Named Entity Recognition (NER) on the CoNLL 2003 dataset, examples with distributed training.                                                                                  |
| [XNLI](#xnli) | Examples running BERT/XLM on the XNLI benchmark. |
| [Abstractive summarization](#abstractive-summarization) | Fine-tuning the library models for abstractive summarization tasks on the CNN/Daily Mail dataset. |

## TensorFlow 2.0 Bert models on GLUE

Based on the script [`run_tf_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py).

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

Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL, XLNet, CTRL.
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

This fine-tuned model is available as a checkpoint under the reference
`bert-large-uncased-whole-word-masking-finetuned-squad`.

#### Fine-tuning XLNet on SQuAD

This example code fine-tunes XLNet on the SQuAD dataset. See above to download the data for SQuAD .

```bash
export SQUAD_DIR=/path/to/SQUAD

python /data/home/hlu/transformers/examples/run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file /data/home/hlu/notebooks/NLP/examples/question_answering/train-v1.1.json \
    --predict_file /data/home/hlu/notebooks/NLP/examples/question_answering/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --save_steps 5000
```

Training with the previously defined hyper-parameters yields the following results:

```python
{
"exact": 85.45884578997162,
"f1": 92.5974600601065,
"total": 10570,
"HasAns_exact": 85.45884578997162,
"HasAns_f1": 92.59746006010651,
"HasAns_total": 10570
}
```

## Named Entity Recognition

Based on the scripts [`run_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/run_ner.py) for Pytorch and
[`run_tf_ner.py`(https://github.com/huggingface/transformers/blob/master/examples/run_tf_ner.py)] for Tensorflow 2.
This example fine-tune Bert Multilingual on GermEval 2014 (German NER).
Details and results for the fine-tuning provided by @stefan-it.

### Data (Download and pre-processing steps)

Data can be obtained from the [GermEval 2014](https://sites.google.com/site/germeval2014ner/data) shared task page.

Here are the commands for downloading and pre-processing train, dev and test datasets. The original data format has four (tab-separated) columns, in a pre-processing step only the two relevant columns (token and outer span NER annotation) are extracted:

```bash
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp
```

The GermEval 2014 dataset contains some strange "control character" tokens like `'\x96', '\u200e', '\x95', '\xad' or '\x80'`. One problem with these tokens is, that `BertTokenizer` returns an empty token for them, resulting in misaligned `InputExample`s. I wrote a script that a) filters these tokens and b) splits longer sentences into smaller ones (once the max. subtoken length is reached).

```bash
wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
```
Let's define some variables that we need for further pre-processing steps and training the model:

```bash
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
```

Run the pre-processing script on training, dev and test datasets:

```bash
python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt
```

The GermEval 2014 dataset has much more labels than CoNLL-2002/2003 datasets, so an own set of labels must be used:

```bash
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

### Prepare the run

Additional environment variables must be set:

```bash
export OUTPUT_DIR=germeval-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
```

### Run the Pytorch version

To start training, just run:

```bash
python3 run_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

If your GPU supports half-precision training, just add the `--fp16` flag. After training, the model will be both evaluated on development and test datasets.

#### Evaluation

Evaluation on development dataset outputs the following for our example:

```bash
10/04/2019 00:42:06 - INFO - __main__ -   ***** Eval results  *****
10/04/2019 00:42:06 - INFO - __main__ -     f1 = 0.8623348017621146
10/04/2019 00:42:06 - INFO - __main__ -     loss = 0.07183869666975543
10/04/2019 00:42:06 - INFO - __main__ -     precision = 0.8467916366258111
10/04/2019 00:42:06 - INFO - __main__ -     recall = 0.8784592370979806
```

On the test dataset the following results could be achieved:

```bash
10/04/2019 00:42:42 - INFO - __main__ -   ***** Eval results  *****
10/04/2019 00:42:42 - INFO - __main__ -     f1 = 0.8614389652384803
10/04/2019 00:42:42 - INFO - __main__ -     loss = 0.07064602487454782
10/04/2019 00:42:42 - INFO - __main__ -     precision = 0.8604651162790697
10/04/2019 00:42:42 - INFO - __main__ -     recall = 0.8624150210424085
```

#### Comparing BERT (large, cased), RoBERTa (large, cased) and DistilBERT (base, uncased)

Here is a small comparison between BERT (large, cased), RoBERTa (large, cased) and DistilBERT (base, uncased) with the same hyperparameters as specified in the [example documentation](https://huggingface.co/transformers/examples.html#named-entity-recognition) (one run):

| Model | F-Score Dev | F-Score Test
| --------------------------------- | ------- | --------
| `bert-large-cased`            | 95.59 | 91.70
| `roberta-large`                  | 95.96 | 91.87
| `distilbert-base-uncased` | 94.34 | 90.32

### Run the Tensorflow 2 version

To start training, just run:

```bash
python3 run_tf_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
```

Such as the Pytorch version, if your GPU supports half-precision training, just add the `--fp16` flag. After training, the model will be both evaluated on development and test datasets.

#### Evaluation

Evaluation on development dataset outputs the following for our example:
```bash
           precision    recall  f1-score   support

 LOCderiv     0.7619    0.6154    0.6809        52
  PERpart     0.8724    0.8997    0.8858      4057
  OTHpart     0.9360    0.9466    0.9413       711
  ORGpart     0.7015    0.6989    0.7002       269
  LOCpart     0.7668    0.8488    0.8057       496
      LOC     0.8745    0.9191    0.8963       235
 ORGderiv     0.7723    0.8571    0.8125        91
 OTHderiv     0.4800    0.6667    0.5581        18
      OTH     0.5789    0.6875    0.6286        16
 PERderiv     0.5385    0.3889    0.4516        18
      PER     0.5000    0.5000    0.5000         2
      ORG     0.0000    0.0000    0.0000         3

micro avg     0.8574    0.8862    0.8715      5968
macro avg     0.8575    0.8862    0.8713      5968
```

On the test dataset the following results could be achieved:
```bash
           precision    recall  f1-score   support

  PERpart     0.8847    0.8944    0.8896      9397
  OTHpart     0.9376    0.9353    0.9365      1639
  ORGpart     0.7307    0.7044    0.7173       697
      LOC     0.9133    0.9394    0.9262       561
  LOCpart     0.8058    0.8157    0.8107      1150
      ORG     0.0000    0.0000    0.0000         8
 OTHderiv     0.5882    0.4762    0.5263        42
 PERderiv     0.6571    0.5227    0.5823        44
      OTH     0.4906    0.6667    0.5652        39
 ORGderiv     0.7016    0.7791    0.7383       172
 LOCderiv     0.8256    0.6514    0.7282       109
      PER     0.0000    0.0000    0.0000        11

micro avg     0.8722    0.8774    0.8748     13869
macro avg     0.8712    0.8774    0.8740     13869
```

## Abstractive summarization

Based on the script
[`run_summarization_finetuning.py`](https://github.com/huggingface/transformers/blob/master/examples/run_summarization_finetuning.py).

Before running this script you should download **both** CNN and Daily Mail
datasets from [Kyunghyun Cho's website](https://cs.nyu.edu/~kcho/DMQA/)  (the
links next to "Stories") in the same folder. Then uncompress the archives by running:

```bash
tar -xvf cnn_stories.tgz && tar -xvf dailymail_stories.tgz
```

note that the finetuning script **will not work** if you do not download both
datasets. We will refer as `$DATA_PATH` the path to where you uncompressed both
archive.

```bash
export DATA_PATH=/path/to/dataset/

python run_summarization_finetuning.py \
    --output_dir=output \
    --model_type=bert2bert \
    --model_name_or_path=bert2bert \
    --do_train \
    --data_path=$DATA_PATH \
```

## XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/blob/master/examples/run_xnli.py).

[XNLI](https://www.nyu.edu/projects/bowman/xnli/) is crowd-sourced dataset based on [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/). It is an evaluation benchmark for cross-lingual text representations. Pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-ressource language such as English and low-ressource languages such as Swahili).

#### Fine-tuning on XNLI

This example code fine-tunes mBERT (multi-lingual BERT) on the XNLI dataset. It runs in 106 mins
on a single tesla V100 16GB. The data for XNLI can be downloaded with the following links and should be both saved (and un-zipped) in a 
`$XNLI_DIR` directory.

* [XNLI 1.0](https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip)
* [XNLI-MT 1.0](https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip)

```bash
export XNLI_DIR=/path/to/XNLI

python run_xnli.py \
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --data_dir $XNLI_DIR \
  --per_gpu_train_batch_size 32 \
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
