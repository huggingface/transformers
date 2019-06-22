# BERT Model Finetuning using Masked Language Modeling objective

## Introduction

The three example scripts in this folder can be used to **fine-tune** a pre-trained BERT model using the pretraining objective (combination of masked language modeling and next sentence prediction loss). In general, pretrained models like BERT are first trained with a pretraining objective (masked language modeling and next sentence prediction for BERT) on a large and general natural language corpus. A classifier head is then added on top of the pre-trained architecture and the model is quickly fine-tuned on a target task, while still (hopefully) retaining its general language understanding. This greatly reduces overfitting and yields state-of-the-art results, especially when training data for the target task are limited.

The [ULMFiT paper](https://arxiv.org/abs/1801.06146) took a slightly different approach, however, and added an intermediate step in which the model is fine-tuned on text **from the same domain as the target task and using the pretraining objective** before the final stage in which the classifier head is added and the model is trained on the target task itself. This paper reported significantly improved results from this step, and found that they could get high-quality classifications even with only tiny numbers (<1000) of labelled training examples, as long as they had a lot of unlabelled data from the target domain.

Although this wasn't covered in the original BERT paper, domain-specific fine-tuning of Transformer models has [recently been reported by other authors](https://arxiv.org/pdf/1905.05583.pdf), and they report performance improvements as well.

## Input format

The scripts in this folder expect a single file as input, consisting of untokenized text, with one **sentence** per line, and one blank line between documents. The reason for the sentence splitting is that part of BERT's training involves a _next sentence_ objective in which the model must predict whether two sequences of text are contiguous text from the same document or not, and to avoid making the task _too easy_, the split point between the sequences is always at the end of a sentence. The linebreaks in the file are therefore necessary to mark the points where the text can be split.

## Usage

There are two ways to fine-tune a language model using these scripts. The first _quick_ approach is to use [`simple_lm_finetuning.py`](./simple_lm_finetuning.py). This script does everything in a single script, but generates training instances that consist of just two sentences. This is quite different from the BERT paper, where (confusingly) the NextSentence task concatenated sentences together from each document to form two long multi-sentences, which the paper just referred to as _sentences_. The difference between this simple approach and the original paper approach can have a significant effect for long sequences since two sentences will be much shorter than the max sequence length. In this case, most of each training example will just consist of blank padding characters, which wastes a lot of computation and results in a model that isn't really training on long sequences.

As such, the preferred approach (assuming you have documents containing multiple contiguous sentences from your target domain) is to use [`pregenerate_training_data.py`](./pregenerate_training_data.py) to pre-process your data into training examples following the methodology used for LM training in the original BERT paper and repository. Since there is a significant random component to training data generation for BERT, this script includes an option to generate multiple _epochs_ of pre-processed data, to avoid training on the same random splits each epoch. Generating an epoch of data for each training epoch should result a better final model, and so we recommend doing so.

You can then train on the pregenerated data using [`finetune_on_pregenerated.py`](./finetune_on_pregenerated.py), and pointing it to the folder created by [`pregenerate_training_data.py`](./pregenerate_training_data.py). Note that you should use the same `bert_model` and case options for both! Also note that `max_seq_len` does not need to be specified for the [`finetune_on_pregenerated.py`](./finetune_on_pregenerated.py) script, as it is inferred from the training examples.

There are various options that can be tweaked, but they are mostly set to the values from the BERT paper/repository and default values should make sense. The most relevant ones are:

- `--max_seq_len`: Controls the length of training examples (in wordpiece tokens) seen by the model. Defaults to 128 but can be set as high as 512. Higher values may yield stronger language models at the cost of slower and more memory-intensive training.
- `--fp16`: Enables fast half-precision training on recent GPUs.

In addition, if memory usage is an issue, especially when training on a single GPU, reducing `--train_batch_size` from the default 32 to a lower number (4-16) can be helpful, or leaving `--train_batch_size` at the default and increasing `--gradient_accumulation_steps` to 2-8. Changing `--gradient_accumulation_steps` may be preferable as alterations to the batch size may require corresponding changes in the learning rate to compensate. There is also a `--reduce_memory` option for both the `pregenerate_training_data.py` and `finetune_on_pregenerated.py` scripts that spills data to disc in shelf objects or numpy memmaps rather than retaining it in memory, which significantly reduces memory usage with little performance impact.

## Examples

### Simple fine-tuning

```
python3 simple_lm_finetuning.py 
--train_corpus my_corpus.txt 
--bert_model bert-base-uncased 
--do_lower_case 
--output_dir finetuned_lm/
--do_train
```

### Pregenerating training data

```
python3 pregenerate_training_data.py
--train_corpus my_corpus.txt
--bert_model bert-base-uncased
--do_lower_case
--output_dir training/
--epochs_to_generate 3
--max_seq_len 256
```

### Training on pregenerated data

```
python3 finetune_on_pregenerated.py
--pregenerated_data training/
--bert_model bert-base-uncased
--do_lower_case
--output_dir finetuned_lm/
--epochs 3
```
