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

# Text classification examples

## GLUE tasks

Based on the script [`run_flax_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/flax/text-classification/run_flax_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
export TASK_NAME=mrpc

python run_flax_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of cola, mnli, mnli-mm, mrpc, qnli, qqp, rte, sst2, stsb, wnli.

Using the command above, the script will train for 3 epochs and run eval after each epoch. 
Metrics and hyperparameters are stored in Tensorflow event files in `---output_dir`.
You can see the results by running `tensorboard` in that directory:

```bash
$ tensorboard --logdir .
```

### Accuracy Evaluation

We train five replicas and report mean accuracy and stdev on the dev set below.
We use the settings as in the command above (with an exception for MRPC and
WNLI which are tiny and where we used 5 epochs instead of 3), and we use a total
train batch size of 32 (we train on 8 Cloud v3 TPUs, so a per-device batch size of 4),

On the task other than MRPC and WNLI we train for 3 these epochs because this is the standard,
but looking at the training curves of some of them (e.g., SST-2, STS-b), it appears the models
are undertrained and we could get better results when training longer.

In the Tensorboard results linked below, the random seed of each model is equal to the ID of the run. So in order to reproduce run 1, run the command above with `--seed=1`. The best run used random seed 2, which is the default in the script. The results of all runs are in [this Google Sheet](https://docs.google.com/spreadsheets/d/1zKL_xn32HwbxkFMxB3ftca-soTHAuBFgIhYhOhCnZ4E/edit?usp=sharing).


| Task  | Metric                       | Acc (best run) | Acc (avg/5runs) | Stdev     | Metrics                                                                  |
|-------|------------------------------|----------------|-----------------|-----------|--------------------------------------------------------------------------|
| CoLA  | Matthew's corr               | 59.57          | 58.04           | 1.81      | [tfhub.dev](https://tensorboard.dev/experiment/f4OvQpWtRq6CvddpxGBd0A/)  |
| SST-2 | Accuracy                     | 92.43          | 91.79           | 0.59      | [tfhub.dev](https://tensorboard.dev/experiment/BYFwa49MRTaLIn93DgAEtA/)  |
| MRPC  | F1/Accuracy                  | 89.50/84.8     | 88.70/84.02     | 0.56/0.48 | [tfhub.dev](https://tensorboard.dev/experiment/9ZWH5xwXRS6zEEUE4RaBhQ/)  |
| STS-B | Pearson/Spearman corr.       | 90.00/88.71    | 89.09/88.61     | 0.51/0.07 | [tfhub.dev](https://tensorboard.dev/experiment/mUlI5B9QQ0WGEJip7p3Tng/)  |
| QQP   | Accuracy/F1                  | 90.88/87.64    | 90.75/87.53     | 0.11/0.13 | [tfhub.dev](https://tensorboard.dev/experiment/pO6h75L3SvSXSWRcgljXKA/)  |
| MNLI  | Matched acc.                 | 84.06          | 83.88           | 0.16      | [tfhub.dev](https://tensorboard.dev/experiment/LKwaOH18RMuo7nJkESrpKg/)  |
| QNLI  | Accuracy                     | 91.01          | 90.86           | 0.18      | [tfhub.dev](https://tensorboard.dev/experiment/qesXxNcaQhmKxPmbw1sOoA/)  |
| RTE   | Accuracy                     | 66.80          | 65.27           | 1.07      | [tfhub.dev](https://tensorboard.dev/experiment/Z84xC0r6RjyzT4SLqiAbzQ/)  |
| WNLI  | Accuracy                     | 39.44          | 32.96           | 5.85      | [tfhub.dev](https://tensorboard.dev/experiment/gV73w9v0RIKrqVw32PZbAQ/)  |

Some of these results are significantly different from the ones reported on the test set of GLUE benchmark on the
website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the website.

### Runtime evaluation

We also ran each task once on a single V100 GPU, 8 V100 GPUs, and 8 Cloud v3 TPUs and report the
overall training time below. For comparison we ran Pytorch's [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py) on a single GPU (last column).


| Task  | 8 TPU   | 8 GPU   | 1 GPU      | 1 GPU (Pytorch) |
|-------|---------|---------|------------|-----------------|
| CoLA  |  1m 46s |  1m 26s | 3m 6s      | 4m 6s           |
| SST-2 |  5m 30s |  6m 28s | 22m 6s     | 34m 37s         |
| MRPC  |  1m 32s |  1m 14s | 2m 17s     | 2m 56s          |
| STS-B |  1m 33s |  1m 12s | 2m 11s     | 2m 48s          |
| QQP   | 24m 40s | 31m 48s | 1h 20m 15s | 2h 54m          |
| MNLI  | 26m 30s | 33m 55s | 2h 7m 30s  | 3u 7m 6s        |
| QNLI  |  8m     |  9m 40s | 34m 20s    | 49m 8s          |
| RTE   |  1m 21s |     55s | 1m 8s      | 1m 16s          |
| WNLI  |  1m 12s |     48s | 38s        | 36s             |
