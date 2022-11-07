import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.core.hooks import ModelHooks
from torch import nn


def f1_eval(logits, labels):
    def getpred(result, T1=0.5, T2=0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret.append(r)
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0

        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys / all_sys
        recall = 0 if correct_gt == 0 else correct_sys / correct_gt
        f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    # logits = list(1 / (1 + np.exp(-logits)))

    temp_labels = []
    for l in labels:
        t = []
        for i in range(36):
            if l[i] == 1:
                t += [i]
        if len(t) == 0:
            t = [36]
        temp_labels.append(t)
    assert len(labels) == len(logits)
    labels = temp_labels

    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2 / 100.0)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2 / 100.0

    return dict(f1=bestf_1, T2=bestT2)


def compute_f1(logits, labels):
    n_gold = n_pred = n_correct = 0
    preds = np.argmax(logits, axis=-1)
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if pred != 0 and label != 0 and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {"precision": prec, "recall": recall, "f1": f1}


def acc(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return (preds == labels).mean()


from collections import Counter


def f1_score(output, label, rel_num=42, na_num=13):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()
    if output.shape != label.shape:
        output = np.argmax(output, axis=-1)

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)

    return dict(f1=micro_f1)


class knnLoss(nn.Module):
    def __init__(self):
        super(knnLoss, self).__init__()

    def loss(self, logits, knn_logits, targets, coeff, kwargs):
        loss = F.cross_entropy(logits, targets, reduction="mean")

        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        knn_loss = F.nll_loss(torch.clamp(torch.log(p), min=-100), targets, reduction="mean")

        loss = loss + torch.mul(loss, knn_loss * coeff)
        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, knn_logits, targets, coeff, multihot_targets=False):
        loss = self.loss(pred_logits, knn_logits, targets, coeff, multihot_targets)
        return loss
