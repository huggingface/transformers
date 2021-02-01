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

## Whole Word Mask Language Model


These scripts leverage the 🤗 Datasets library and the Trainer API. You can easily customize them to your needs if you
need extra processing on your datasets.

The following examples, will run on a datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.



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

You could run the following:


```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export LTP_RESOURCE=/path/to/ltp/tokenizer
export BERT_RESOURCE=/path/to/bert/tokenizer
export SAVE_PATH=/path/to/data/ref.txt

python run_chinese_ref.py \
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

**Note1:** On TPU, you should the flag `--pad_to_max_length` to make sure all your batches have the same length.

**Note2:** And if you have any questions or something goes wrong when runing this code, don't hesitate to pin @wlhgtc.