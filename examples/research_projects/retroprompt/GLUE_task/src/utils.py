import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data_processor import compute_metrics_mapping
from transformers.trainer_utils import EvalPrediction


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        logits = p.predictions
        preds = np.argmax(logits, axis=1)
        label_ids = p.label_ids

        return compute_metrics_mapping[task_name](task_name, preds, label_ids)

    return compute_metrics_fn


def data_collator_for_cl(features):
    features1 = [vars(f) for f in features]

    first = features1[0]
    batch1 = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch1["labels"] = torch.tensor([f["label"] for f in features1], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch1["labels"] = torch.stack([f["label_ids"] for f in features1])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch1["labels"] = torch.tensor([f["label_ids"] for f in features1], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch1[k] = torch.stack([f[k] for f in features1])
            else:
                batch1[k] = torch.tensor([f[k] for f in features1])

    return batch1


def count_special_tokens_in_template(template, tokenizer, max_len_label_tokens):
    len_special_token_in_template = 0

    special_token_mapping = {
        "cls": tokenizer.cls_token_id,
        "mask": tokenizer.mask_token_id,
        "sep": tokenizer.sep_token_id,
        "sep+": tokenizer.sep_token_id,
        "prompt": tokenizer.pad_token_id,
    }
    for part in template.split("*"):
        if part in special_token_mapping:
            if part == "mask":
                len_special_token_in_template += max_len_label_tokens
            else:
                len_special_token_in_template += 1
        elif "sent" in part:
            continue
        else:
            # Just natural language prompt
            part = part.replace("_", " ")
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                len_special_token_in_template += 1
            else:
                len_special_token_in_template += len(tokenizer.encode(part, add_special_tokens=False))

    return len_special_token_in_template


def map_supports(supports):
    examples_by_class = defaultdict(lambda: list())
    for example in supports:
        examples_by_class[example.label].append(example)

    feature_by_class = {}
    for label in examples_by_class.keys():
        batch = {}

        examples = examples_by_class[label]
        features = [vars(e) for e in examples]
        first = features[0]
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)

        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        feature_by_class[label] = batch

    return feature_by_class


class knnLoss(nn.Module):
    def __init__(self):
        super(knnLoss, self).__init__()

    def loss(self, logits, knn_logits, targets, coeff):
        loss = F.cross_entropy(logits, targets, reduction="mean")

        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        knn_loss = F.nll_loss(torch.clamp(torch.log(p), min=-100), targets, reduction="mean")

        loss = loss + torch.mul(loss, knn_loss * coeff)

        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, knn_logits, targets, coeff):
        loss = self.loss(pred_logits, knn_logits, targets, coeff)
        return loss


class knnFocalLikeLoss(nn.Module):
    def __init__(self):
        super(knnFocalLikeLoss, self).__init__()

    def is_single(self):
        return False

    def loss(self, logits, knn_logits, targets, gamma):
        loss = F.cross_entropy(logits, targets, reduction="none")

        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)
        # modulator = (1 - p_t) ** gamma
        # below is a numerically stable version
        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        p_t = torch.sum(p * targets, -1)
        # a mask of p == 0
        modulator = torch.exp(gamma * torch.log1p(-1 * p_t))

        loss = loss * modulator
        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, knn_logits, targets, coeff):
        loss = self.loss(pred_logits, knn_logits, targets, coeff)
        return loss

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(labels.size(0), nb_classes, device=labels.device).scatter_(1, labels, 1.0)
        return target
