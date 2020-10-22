# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ...file_utils import is_sklearn_available, requires_sklearn


if is_sklearn_available():
    from sklearn.metrics import f1_score, matthews_corrcoef

    from scipy.stats import pearsonr, spearmanr


def simple_accuracy(preds, labels):
    requires_sklearn(simple_accuracy)
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    requires_sklearn(acc_and_f1)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    requires_sklearn(pearson_and_spearman)
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def glue_compute_metrics(task_name, preds, labels):
    requires_sklearn(glue_compute_metrics)
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"mnli/acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"mnli-mm/acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def xnli_compute_metrics(task_name, preds, labels):
    requires_sklearn(xnli_compute_metrics)
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
