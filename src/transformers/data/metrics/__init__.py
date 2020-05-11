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

import re
import string
from collections import defaultdict, Counter
try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    import numpy as np
    from scipy.special import softmax

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels, f1_avg="binary"):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average=f1_avg)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        """ Compute max metric between prediction and each ground truth.
        From official ReCoRD eval script """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _record_f1_score(prediction, ground_truth):
        """ Compute normalized token level F1
        From official ReCoRD eval script """
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _record_em_score(prediction, ground_truth):
        """ Compute normalized exact match
        From official ReCoRD eval script """
        return normalize_answer(prediction) == normalize_answer(ground_truth)

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
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

    def superglue_compute_metrics(task_name, preds, labels, guids=None, answers=None):
        assert len(preds) == len(labels)
        if task_name == "boolq":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "cb":
            return acc_and_f1(preds, labels, f1_avg="macro")
        elif task_name == "copa":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "multirc":
            assert len(guids) == len(preds), "Different number of predictions and IDs!"
            qst2ans = defaultdict(list)
            # iterate over examples and aggregate statistics
            for idx, pred, label in zip(guids, preds, labels):
                qst_idx = f"{idx[0]}-{idx[1]}"
                qst2ans[qst_idx].append((pred, label))

            f1s, ems = [], []
            for qst, preds_and_labels in qst2ans.items():
                preds, labels = zip(*preds_and_labels)
                f1 = f1_score(y_true=labels, y_pred=preds)
                f1s.append(f1)
                em = int(sum([p == l for p, l in preds_and_labels]) == len(preds_and_labels))
                ems.append(em)

            avg_f1 = sum(f1s) / len(f1s)
            avg_em = sum(ems) / len(ems)
            em_and_f1 = (avg_em + avg_f1) / 2
            return {"f1": avg_f1, "em": avg_em, "em_and_f1": em_and_f1}

        elif task_name == "record":
            assert len(guids) == len(preds), "Different number of predictions and IDs!"
            qst2ans = defaultdict(list)
            # iterate over examples and aggregate statistics
            for idx, pred, label in zip(guids, preds, labels):
                qst_idx = (idx[0], idx[1])
                qst2ans[qst_idx].append((idx[2], pred))

            f1s, ems = [], []
            for qst, idxs_and_prds in qst2ans.items():
                cands, golds = answers[qst]

                idxs_and_prds.sort(key=lambda x: x[0])
                logits = np.vstack([i[1] for i in idxs_and_prds])

                # take the most probable choice as the prediction
                pred_idx = softmax(logits, axis=1)[:, -1].argmax().item()
                pred = cands[pred_idx]

                # compute metrics
                f1 = metric_max_over_ground_truths(_record_f1_score, pred, golds)
                em = metric_max_over_ground_truths(_record_em_score, pred, golds)
                f1s.append(f1)
                ems.append(em)

            avg_f1 = sum(f1s) / len(f1s)
            avg_em = sum(ems) / len(ems)
            em_and_f1 = (avg_em + avg_f1) / 2
            return {"f1": avg_f1, "em": avg_em, "em_and_f1": em_and_f1}

        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wic":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wsc":
            return acc_and_f1(preds, labels)
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    superglue_tasks_metrics = {
        "boolq": "acc",
        "cb": "acc_and_f1",
        "copa": "acc",
        "multirc": "em_and_f1",
        "record": "em_and_f1",
        "rte": "acc",
        "wic": "acc",
        "wsc": "acc_and_f1",
    }

