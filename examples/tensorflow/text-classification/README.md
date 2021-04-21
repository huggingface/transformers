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

# Text classification examples

## GLUE tasks

Based on the script [`run_tf_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_tf_glue.py).

Fine-tuning the library TensorFlow 2.0 Bert model for sequence classification on the MRPC task of the GLUE benchmark: [General Language Understanding Evaluation](https://gluebenchmark.com/).

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


## Run generic text classification script in TensorFlow

The script [run_tf_text_classification.py](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_tf_text_classification.py) allows users to run a text classification on their own CSV files. For now there are few restrictions, the CSV files must have a header corresponding to the column names and not more than three columns: one column for the id, one column for the text and another column for a second piece of text in case of an entailment classification for example.

To use the script, one as to run the following command line:
```bash
python run_tf_text_classification.py \
  --train_file train.csv \ ### training dataset file location (mandatory if running with --do_train option)
  --dev_file dev.csv \ ### development dataset file location (mandatory if running with --do_eval option)
  --test_file test.csv \ ### test dataset file location (mandatory if running with --do_predict option)
  --label_column_id 0 \ ### which column corresponds to the labels
  --model_name_or_path bert-base-multilingual-uncased \
  --output_dir model \
  --num_train_epochs 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --save_steps 10 \
  --overwrite_output_dir \
  --max_seq_length 128
```

