# Examples

In this section a few examples are put together. All of these examples work for several models, making use of the very
similar API between the different models.

**Important**
To run the latest versions of the examples, you have to install from source and install some specific requirements for the examples.
Execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

| Section                    | Description                                                                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------
| [TensorFlow 2.0 models on GLUE](#TensorFlow-2.0-Bert-models-on-GLUE) | Examples running BERT TensorFlow 2.0 model on the GLUE tasks. |
| [Language Model training](#language-model-training) | Fine-tuning (or training from scratch) the library models for language modeling on a text dataset. Causal language modeling for GPT/GPT-2, masked language modeling for BERT/RoBERTa. |
| [Language Generation](#language-generation) | Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL and XLNet. |
| [GLUE](#glue) | Examples running BERT/XLM/XLNet/RoBERTa on the 9 GLUE tasks. Examples feature distributed training as well as half-precision. |
| [SQuAD](#squad) | Using BERT/RoBERTa/XLNet/XLM for question answering, examples with distributed training. |
| [Multiple Choice](#multiple-choice) | Examples running BERT/XLNet/RoBERTa on the SWAG/RACE/ARC tasks. |
| [Named Entity Recognition](https://github.com/huggingface/transformers/tree/master/examples/ner) | Using BERT for Named Entity Recognition (NER) on the CoNLL 2003 dataset, examples with distributed training. |
| [XNLI](#xnli) | Examples running BERT/XLM on the XNLI benchmark. |
| [Adversarial evaluation of model performances](#adversarial-evaluation-of-model-performances) | Testing a model with adversarial evaluation of natural language inference on the Heuristic Analysis for NLI Systems (HANS) dataset (McCoy et al., 2019.) |

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

## Language model training

Based on the script [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py).

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, BERT and RoBERTa (DistilBERT 
to be added soon). GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa 
are fine-tuned using a masked language modeling (MLM) loss.

Before running the following example, you should get a file that contains text on which the language model will be
trained or fine-tuned. A good example of such text is the [WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

We will refer to two different files: `$TRAIN_FILE`, which contains text for training, and `$TEST_FILE`, which contains
text that will be used for evaluation.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
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

python run_language_modeling.py \
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

Before running any one of these GLUE tasks you should download the
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

#### Fine-tuning BERT on SQuAD1.0

This example code fine-tunes BERT on the SQuAD1.0 dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB. The data for SQuAD can be downloaded with the following links and should be saved in a
$SQUAD_DIR directory.

* [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

And for SQuAD2.0, you need to download:

- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
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


Here is an example using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD1.1:

```bash
python -m torch.distributed.launch --nproc_per_node=8 ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./examples/models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 93.15
exact_match = 86.91
```

This fine-tuned model is available as a checkpoint under the reference
`bert-large-uncased-whole-word-masking-finetuned-squad`.

#### Fine-tuning XLNet on SQuAD

This example code fine-tunes XLNet on both SQuAD1.0 and SQuAD2.0 dataset. See above to download the data for SQuAD .

##### Command for SQuAD1.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --save_steps 5000
```

##### Command for SQuAD2.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_gpu_eval_batch_size=2  \
    --per_gpu_train_batch_size=2   \
    --save_steps 5000
```

Larger batch size may improve the performance while costing more memory.

##### Results for SQuAD1.0 with the previously defined hyper-parameters:

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

##### Results for SQuAD2.0 with the previously defined hyper-parameters:

```python
{
"exact": 80.4177545691906,
"f1": 84.07154997729623,
"total": 11873,
"HasAns_exact": 76.73751686909581,
"HasAns_f1": 84.05558584352873,
"HasAns_total": 5928,
"NoAns_exact": 84.0874684608915,
"NoAns_f1": 84.0874684608915,
"NoAns_total": 5945
}
```




## XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/blob/master/examples/run_xnli.py).

[XNLI](https://www.nyu.edu/projects/bowman/xnli/) is crowd-sourced dataset based on [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/). It is an evaluation benchmark for cross-lingual text representations. Pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

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

## MM-IMDb

Based on the script [`run_mmimdb.py`](https://github.com/huggingface/transformers/blob/master/examples/mm-imdb/run_mmimdb.py).

[MM-IMDb](http://lisi1.unal.edu.co/mmimdb/) is a Multimodal dataset with around 26,000 movies including images, plots and other metadata.

### Training on MM-IMDb

```
python run_mmimdb.py \
    --data_dir /path/to/mmimdb/dataset/ \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --output_dir /path/to/save/dir/ \
    --do_train \
    --do_eval \
    --max_seq_len 512 \
    --gradient_accumulation_steps 20 \
    --num_image_embeds 3 \
    --num_train_epochs 100 \
    --patience 5
```

## Adversarial evaluation of model performances

Here is an example on evaluating a model using adversarial evaluation of natural language inference with the Heuristic Analysis for NLI Systems (HANS) dataset [McCoy et al., 2019](https://arxiv.org/abs/1902.01007). The example was gracefully provided by [Nafise Sadat Moosavi](https://github.com/ns-moosavi).

The HANS dataset can be downloaded from [this location](https://github.com/tommccoy1/hans).

This is an example of using test_hans.py:

```bash
export HANS_DIR=path-to-hans
export MODEL_TYPE=type-of-the-model-e.g.-bert-roberta-xlnet-etc
export MODEL_PATH=path-to-the-model-directory-that-is-trained-on-NLI-e.g.-by-using-run_glue.py

python examples/hans/test_hans.py \
        --task_name hans \
        --model_type $MODEL_TYPE \
        --do_eval \
        --do_lower_case \
        --data_dir $HANS_DIR \
        --model_name_or_path $MODEL_PATH \
        --max_seq_length 128 \
        --output_dir $MODEL_PATH \
```

This will create the hans_predictions.txt file in MODEL_PATH, which can then be evaluated using hans/evaluate_heur_output.py from the HANS dataset.

The results of the BERT-base model that is trained on MNLI using batch size 8 and the random seed 42 on the HANS dataset is as follows:

```bash
Heuristic entailed results:
lexical_overlap: 0.9702
subsequence: 0.9942
constituent: 0.9962

Heuristic non-entailed results:
lexical_overlap: 0.199
subsequence: 0.0396
constituent: 0.118
```
