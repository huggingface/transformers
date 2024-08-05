<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Token classification

Fine-tuning the library models for token classification task such as Named Entity Recognition (NER), Parts-of-speech
tagging (POS) or phrase extraction (CHUNKS). The main script `run_ner.py` leverages the [🤗 Datasets](https://github.com/huggingface/datasets) library. You can easily
customize it to your needs if you need extra processing on your datasets.

It will either run on a datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for
training and validation, you might just need to add some tweaks in the data preprocessing.

The following example fine-tunes BERT on CoNLL-2003:

```bash
python run_ner.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner
```

To run on your own training and validation files, use the following command:

```bash
python run_ner.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --train_file path_to_train_file \
  --validation_file path_to_validation_file \
  --output_dir /tmp/test-ner
```

**Note:** This script only works with models that have a fast tokenizer (backed by the [🤗 Tokenizers](https://github.com/huggingface/tokenizers) library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks).
