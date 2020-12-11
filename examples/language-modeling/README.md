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

## Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2,
ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those
objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

These scripts leverage the 🤗 Datasets library and the Trainer API. You can easily customize them to your needs if you
need extra processing on your datasets.

**Note:** The old script `run_language_modeling.py` is still available
[here](https://github.com/huggingface/transformers/blob/master/examples/contrib/legacy/run_language_modeling.py).

The following examples, will run on a datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
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
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```


### RoBERTa/BERT/DistilBERT and masked language modeling

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
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.

### Whole word masking

The BERT authors released a new version of BERT using Whole Word Masking in May 2019. Instead of masking randomly
selected tokens (which may be part of words), they mask randomly selected words (masking all the tokens corresponding
to that word). This technique has been refined for Chinese in [this paper](https://arxiv.org/abs/1906.08101).

To fine-tune a model using whole word masking, use the following script:
```bash
python run_mlm_wwm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm-wwm
```

For Chinese models, we need to generate a reference files (which requires the ltp library), because it's tokenized at
the character level.

**Q :** Why a reference file?

**A :** Suppose we have a Chinese sentence like: `我喜欢你` The original Chinese-BERT will tokenize it as
`['我','喜','欢','你']` (character level). But `喜欢` is a whole word. For whole word masking proxy, we need a result
like `['我','喜','##欢','你']`, so we need a reference file to tell the model which position of the BERT original token
should be added `##`.

**Q :** Why LTP ?

**A :** Cause the best known Chinese WWM BERT is [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) by HIT.
It works well on so many Chines Task like CLUE (Chinese GLUE). They use LTP, so if we want to fine-tune their model,
we need LTP.

Now LTP only only works well on `transformers==3.2.0`. So we don't add it to requirements.txt.
You need to create a separate environment with this version of Transformers to run the `run_chinese_ref.py` script that
will create the reference files. The script is in `examples/contrib`. Once in the proper environment, run the
following:


```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export LTP_RESOURCE=/path/to/ltp/tokenizer
export BERT_RESOURCE=/path/to/bert/tokenizer
export SAVE_PATH=/path/to/data/ref.txt

python examples/contrib/run_chinese_ref.py \
    --file_name=path_to_train_or_eval_file \
    --ltp=path_to_ltp_tokenizer \
    --bert=path_to_bert_tokenizer \
    --save_path=path_to_reference_file
```

Then you can run the script like this: 


```bash
python run_mlm_wwm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --train_ref_file path_to_train_chinese_ref_file \
    --validation_ref_file path_to_validation_chinese_ref_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm-wwm
```

**Note:** On TPU, you should the flag `--pad_to_max_length` to make sure all your batches have the same length.

### XLNet and permutation language modeling

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
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.
