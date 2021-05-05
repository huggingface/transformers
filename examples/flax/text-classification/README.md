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

Based on the script [`run_flax_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/flax/text-classification/run_flax_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
export TASK_NAME=mrpc

python run_flax_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of cola, mnli, mnli-mm, mrpc, qnli, qqp, rte, sst2, stsb, wnli.

Using the command above, the script will train for 3 epochs and run eval after each epoch. Once
training is done, it will run a few predictions on the test set and output the predictions.
Metrics and hyperparameters are stored in Tensorflow event files in `---output_dir`.
You can see the results by running `tensorboard` in that directory:

```bash
$ tensorboard --logdir .
```

### Accuracy Evaluation

We trained five replicas and report mean accuracy and stdev on the dev set below.
We use the same settings as in the command above (with an exception for MRPC and
WNLI which are tiny and where we used 5 epochs instead of 3). 

The random seed of each model is equal to the ID of the run. So in order to reproduce run 1,
run the command above with `--seed=1`.

| Task  | Metric                       | Acc         | Stdev       | Metrics        |
|-------|------------------------------|-------------|-------------|----------------|
| CoLA  | Matthew's corr               | 59.47       |             | [tfhub.dev](https://tensorboard.dev/experiment/NVEMw1kKS9ar1fahvTN0qg/)  |
| SST-2 | Accuracy                     | 91.49       |             | [tfhub.dev](https://tensorboard.dev/experiment/RBdoB3FnTJWzMa6L1ajPFg/)  |
| MRPC  | F1/Accuracy                  | 89.18/84.38 |             | [tfhub.dev](https://tensorboard.dev/experiment/R2XV3HXrRyynxig3hE6Qxw/)  |
| STS-B | Person/Spearman corr.        | 89.11/88.87 |             | [tfhub.dev](https://tensorboard.dev/experiment/nI3MurTWSXaLyb34xVvf1A/)  |
| QQP   | Accuracy/F1                  | 91.00/87.81 |             | [tfhub.dev](https://tensorboard.dev/experiment/QNesGrosRAq4N1ARfJJm8w/)  |
| MNLI  | Matched acc./Mismatched acc. | 83.87/83.92 |             | [tfhub.dev](https://tensorboard.dev/experiment/TWCJqjVZRNuVQj8CGZ7ATA/) / [tfhub.dev](https://tensorboard.dev/experiment/iM7nq7M2R7K0e8tgrabYkg/)  |
| QNLI  | Accuracy                     | 90.67       |             | [tfhub.dev](https://tensorboard.dev/experiment/RsVfbpAARSaJkx46WXYMig/)  |
| RTE   | Accuracy                     | 67.58       |             | [tfhub.dev](https://tensorboard.dev/experiment/bkrDwYQaQcOkDkNlRGLrXw/)  |
| WNLI  | Accuracy                     | 56.25       |             | [tfhub.dev](https://tensorboard.dev/experiment/vqQxIkX5Q3iaKX23jG3BMQ/)  |

Some of these results are significantly different from the ones reported on the test set of GLUE benchmark on the
website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the website.

### Runtime evaluation

We also ran each task once on a single GPU, 8 GPUs, and 8 TPUs and report the
overall training time below. The training time varied very little between runs,
so we do not report mean and stdevs here. We used P100 GPUs and v3 Cloud TPUs.


| Task  | time (1 GPU) | time (8 GPU) | time (8 TPU) |
|-------|--------------|--------------|--------------|
| CoLA  |              |  1m 26s      |              |
| SST-2 |              |  6m 28s      |              |
| MRPC  |              |  1m 14s      |              |
| STS-B |              |  1m 12s      |              |
| QQP   |              | 31m 48s      |              |
| MNLI  |              | 33m 55s      |              |
| QNLI  |              |  9m 40s      |              |
| RTE   |              |     55s      |              |
| WNLI  |              |     48s      |              |