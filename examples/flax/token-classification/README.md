<!---
Copyright 2021 The Google Flax Team Authors and HuggingFace Team. All rights reserved.

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

# Token classification examples

Fine-tuning the library models for token classification task such as Named Entity Recognition (NER), Parts-of-speech tagging (POS) or phrase extraction (CHUNKS). The main script run_flax_ner.py leverages the 🤗 Datasets library. You can easily customize it to your needs if you need extra processing on your datasets.

It will either run on a datasets hosted on our hub or with your own text files for training and validation, you might just need to add some tweaks in the data preprocessing.

The following example fine-tunes BERT on CoNLL-2003:


```bash
python run_flax_ner.py \
  --model_name_or_path google-bert/bert-base-cased \
  --dataset_name conll2003 \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --output_dir ./bert-ner-conll2003 \
  --eval_steps 300 \
  --push_to_hub
```

Using the command above, the script will train for 3 epochs and run eval after each epoch. 
Metrics and hyperparameters are stored in Tensorflow event files in `--output_dir`.
You can see the results by running `tensorboard` in that directory:

```bash
$ tensorboard --logdir .
```

or directly on the hub under *Training metrics*.

sample Metrics - [tfhub.dev](https://tensorboard.dev/experiment/u52qsBIpQSKEEXEJd2LVYA)