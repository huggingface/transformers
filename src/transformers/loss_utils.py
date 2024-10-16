import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .models.detr.loss_detr import ForObjectDetectionLoss, ForSegmentationLoss
from .models.rt_detr.loss_rt_detr import RtDetrForObjectDetectionLoss


def DefaultCrossEntropyLoss(logits, labels, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, kwargs["vocab_size"])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    num_items = kwargs.pop("num_items", None)

    if num_items is not None:
        # Calculate the CrossEntropyLoss manually when using grad accum
        log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
        loss = -log_probs[range(shift_labels.size(0)), shift_labels]
        loss = loss.sum() / num_items
    else:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    return loss


def ForSequenceClassificationLoss(logits, labels, pooled_logits, **kwargs):
    config = kwargs["config"]
    num_labels = config.num_labels
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    if config.problem_type == "regression":
        loss_fct = MSELoss()
        if num_labels == 1:
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            loss = loss_fct(pooled_logits, labels)
    elif config.problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, num_labels), labels.view(-1))
    elif config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)
    return loss


def ForQuestionAnsweringLoss(start_logits, end_logits, start_positions, end_positions):
    total_loss = None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1).to(start_logits.device)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1).to(end_logits.device)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
    return total_loss


def ForTokenClassification(logits, labels, config, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.view(-1, config.num_labels)
    labels = labels.view(-1)
    logits = logits.float()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    return loss_fct(logits, labels)


LOSS_MAPPING = {
    "ForCausalLM": DefaultCrossEntropyLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
}

LOSS_MAPPING["ForSegmentation"] = ForSegmentationLoss
LOSS_MAPPING["ForObjectDetection"] = ForObjectDetectionLoss
LOSS_MAPPING["RtForObjectDetection"] = RtDetrForObjectDetectionLoss
