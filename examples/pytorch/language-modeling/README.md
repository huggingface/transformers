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

# Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2,
ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those
objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

**Note:** The old script `run_language_modeling.py` is still available [here](https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

## GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_clm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```bash
python run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm
```

## RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

To run on your own training and validation files, use the following command:

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_mlm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```bash
python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path roberta-base \
    --output_dir /tmp/test-mlm
```

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.

## Whole word masking

This part was moved to `examples/research_projects/mlm_wwm`.

## XLNet and permutation language modeling

XLNet uses a different training objective, which is permutation language modeling. It is an autoregressive method
to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input
sequence factorization order.

We use the `--plm_probability` flag to define the ratio of length of a span of masked tokens to surrounding
context length for permutation language modeling.

The `--max_span_length` flag may also be used to limit the length of a span of masked tokens used
for permutation language modeling.

Here is how to fine-tune XLNet on wikitext-2:

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

To fine-tune it on your own training and validation file, run:

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.

## BART and denoising language modeling

BART is is an encoder-decoder that is trained on the denoising objective. The input text is corrupted and the model
must reconstruct it. The added noise includes token masking, in-filling (replacing multiple tokens by a single mask),
replacing tokens by random tokens, permuting sentences in a sequence, and so on. The implementation here borrows from
heavily from the original
[fairseq](https://github.com/facebookresearch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py)
implementation and the
[FLAX training script](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_bart_dlm_flax.py)
.

### 1. Train a tokenizer on a dataset on the hub

```shell
python prepare_tokenizer.py \
    oscar \
    --dataset_config_name unshuffled_deduplicated_nl \
    --dataset_split train \
    --dout ./my-bart-model
```

### 2. Prepare a model config file based on an existing model

```shell
python prepare_config.py \
    --pretrained_model_name facebook/bart-base \
    --dout ./my-bart-model
```

### 3. Train the model and specific tokenizer and config

By default, we use the denoising parameters described in
[this post](https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320). 

```shell
python run_bart_dlm.py \
    --config_name ./my-bart-model \
    --tokenizer_name ./my-bart-model \
    --dataset_name oscar \
    --dataset_config_name unshuffled_deduplicated_nl \
    --output_dir ./my-bart-model \
    --do_train \
    --do_eval
```


### Some notes 

#### Sentence splitting

As part of BART, the sentences in a sample may be permuted (reordered). To detect sentences for each sample, we need
sentence splitting. By dfault, we'll use NLTK's English punct sentence splitter but by passing a spaCy model name
to `spacy_model` (e.g. `en_core_web_sm`) you can also rely on spaCy for better (but slower) sentence splitting.
You can also disable sentence splitting completely with `--no_sentence_splitting`. In that case, make sure the
sentences are already split with a padding token between them (`<pad>`).


#### Default values
The defaults are set to the
[given BART args](https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320). This differs from
the Flax defaults in one respect, namely `poisson_lambda`, which is now set to `3.5` instead of `3.0`.


#### HF (Flax), fairseq, and current implementation

There are some differences in implementation between fairseq, the HF FLAX example, and this PyTorch implementation.

- `argwhere` in the Flax example
[in this position](https://github.com/huggingface/transformers/blob/65fb71bc762c46bb067306c1fd083b1cba87a095/examples/flax/language-modeling/run_bart_dlm_flax.py#L319)
is not the same as what is happening in fairseq. [In fairseq](https://github.com/facebookresearch/fairseq/blob/a6a63279422f846a3c2f6c45b9c96d6951cc4b82/fairseq/data/denoising_dataset.py#L230)
we check explicitly that the previous token was not a "full stop" (padding token) but in HF we just check whether the
current token is a full stop. In the current example I also explicitly check that the next token is not a full stop,
in case of padding. (However, in practice that should be a non-issue since all batches/samples should have the
same sequence length and there should not be any padding.)
- I found that the result of sentence permutation was not consistent in terms of where the separating pad token ended
up ([bug report](https://github.com/facebookresearch/fairseq/issues/4695)), so I have reimplemented that method so
that sentences in a sequence are still separated by a padding token, even after permutation.
- In HF FLAX, the token_mask is restricted to [non-special and non-padding tokens](https://github.com/huggingface/transformers/blob/65fb71bc762c46bb067306c1fd083b1cba87a095/examples/flax/language-modeling/run_bart_dlm_flax.py#L361).
In Fairseq, by default, only the first and last tokens are excluded and [all others](https://github.com/facebookresearch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py#L241)
are prone to masking. The HF implementation seems sensible so I follow that. `get_special_tokens_mask` includes the 
padding token, though, so no need to add that separately.
- The Flax example does not include methods to add more noise. I have ported those as well.
- However, I did not adapt `add_insertion_noise` to work well with padded sequences. So the inserted noise may occur
ANYWHERE. It is unclear whether this is intended behavior.


## Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_clm.py --model_type gpt2 --tokenizer_name gpt2 \ --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
[...]
```

This feature is only available in `run_clm.py`, `run_plm.py` and `run_mlm.py`.

