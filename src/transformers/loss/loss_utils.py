# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss

from .loss_ctc import ForCTCLoss
from .loss_d_fine import DFineForObjectDetectionLoss
from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
from .loss_deimv2 import Deimv2ForObjectDetectionLoss
from .loss_for_object_detection import ForObjectDetectionLoss, ForSegmentationLoss
from .loss_grounding_dino import GroundingDinoForObjectDetectionLoss
from .loss_lsd import PocketTTSForConditionalGenerationLoss
from .loss_lw_detr import LwDetrForObjectDetectionLoss
from .loss_rf_detr import RfDetrForSegmentationLoss
from .loss_rnnt import ParakeetForRNNTLoss
from .loss_rt_detr import RTDetrForObjectDetectionLoss
from .loss_tdt import ParakeetForTDTLoss


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def loss_share(
    loss: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Returns this batch's share of `loss`, a loss averaged over this batch alone such as the MoE router load balancing
    loss.

    Without `num_items_in_batch`, `loss` is returned as is.
    """
    if num_items_in_batch is None:
        return loss
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss * (target != ignore_index).sum().to(loss.device) / num_items_in_batch


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Accumulate assignment counts and probability sums layer by layer, normalizing at the end,
    # so peak memory stays O(seq_len * num_experts) regardless of the number of layers.
    compute_device = gate_logits[0].device
    tokens_per_expert_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    router_prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    total_rows = 0.0

    if attention_mask is not None:
        # The same flat mask applies to every layer's [batch_size * sequence_length] rows.
        flat_mask = attention_mask.reshape(-1).to(device=compute_device, dtype=torch.float32)

    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate.to(compute_device), dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if attention_mask is None:
            # Count of top-k assignments per expert
            tokens_per_expert_sum = (
                tokens_per_expert_sum + torch.bincount(selected_experts.reshape(-1), minlength=num_experts).float()
            )
            # Sum of routing probabilities per expert
            router_prob_sum = router_prob_sum + routing_weights.float().sum(dim=0)
            total_rows = total_rows + routing_weights.shape[0]
        else:
            # Same reductions, weighted by the attention mask to exclude padding tokens
            tokens_per_expert_sum = tokens_per_expert_sum + torch.zeros(
                num_experts, dtype=torch.float32, device=compute_device
            ).scatter_add_(0, selected_experts.reshape(-1), flat_mask.repeat_interleave(top_k))
            router_prob_sum = router_prob_sum + (routing_weights.float() * flat_mask.unsqueeze(-1)).sum(dim=0)
            total_rows = total_rows + flat_mask.sum()

    tokens_per_expert = tokens_per_expert_sum / total_rows
    router_prob_per_expert = router_prob_sum / total_rows

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    aux_loss: torch.Tensor | float | None = None,
    aux_loss_coef: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)

    if aux_loss is not None:
        aux_loss = loss_share(
            torch.as_tensor(aux_loss, device=loss.device), shift_labels, num_items_in_batch, ignore_index
        )
        loss = loss + aux_loss_coef * aux_loss
    return loss


def ForMaskedLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    aux_loss: torch.Tensor | float | None = None,
    aux_loss_coef: float = 1.0,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)

    if aux_loss is not None:
        aux_loss = loss_share(torch.as_tensor(aux_loss, device=loss.device), labels, num_items_in_batch, ignore_index)
        loss = loss + aux_loss_coef * aux_loss
    return loss


def ForSequenceClassificationLoss(labels: torch.Tensor, pooled_logits: torch.Tensor, config, **kwargs) -> torch.Tensor:
    num_labels = config.num_labels
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    labels = labels.to(pooled_logits.device)
    if config.problem_type == "regression":
        loss_fct = MSELoss()
        if num_labels == 1:
            return loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            return loss_fct(pooled_logits, labels)
    if config.problem_type == "single_label_classification":
        return fixed_cross_entropy(pooled_logits.view(-1, num_labels), labels.view(-1), **kwargs)

    if config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        return loss_fct(pooled_logits, labels)

    raise RuntimeError(f"Invalid problem type: {config.problem_type}")


def ForQuestionAnsweringLoss(start_logits, end_logits, start_positions, end_positions, **kwargs):
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

        start_loss = fixed_cross_entropy(start_logits, start_positions, ignore_index=ignored_index, **kwargs)
        end_loss = fixed_cross_entropy(end_logits, end_positions, ignore_index=ignored_index, **kwargs)
        total_loss = (start_loss + end_loss) / 2
    return total_loss


def ForTokenClassification(logits: torch.Tensor, labels, config, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.view(-1, config.num_labels)
    labels = labels.view(-1).to(logits.device)
    logits = logits.float()
    # Flatten the tokens
    return fixed_cross_entropy(logits, labels, **kwargs)


def ForSemanticSegmentationLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    num_items_in_batch: torch.Tensor | None = None,
    auxiliary_logits: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 0.4,
    **kwargs,
) -> torch.Tensor:
    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    loss = fixed_cross_entropy(
        upsampled_logits, labels, num_items_in_batch=num_items_in_batch, ignore_index=ignore_index
    )
    if auxiliary_logits is not None:
        upsampled_auxiliary_logits = nn.functional.interpolate(
            auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = loss + auxiliary_loss_weight * fixed_cross_entropy(
            upsampled_auxiliary_logits, labels, num_items_in_batch=num_items_in_batch, ignore_index=ignore_index
        )
    return loss


LOSS_MAPPING = {
    "ForSemanticSegmentation": ForSemanticSegmentationLoss,
    "ForCausalLM": ForCausalLMLoss,
    "ForMaskedLM": ForMaskedLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForImageClassification": ForSequenceClassificationLoss,
    "ForVideoClassification": ForSequenceClassificationLoss,
    "ForAudioClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
    "ForSegmentation": ForSegmentationLoss,
    "ForObjectDetection": ForObjectDetectionLoss,
    "ForCTC": ForCTCLoss,
    "ForConditionalGeneration": ForCausalLMLoss,
    "DeformableDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "ConditionalDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "DabDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "GroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "MMGroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "ConditionalDetrForSegmentation": DeformableDetrForSegmentationLoss,
    "RTDetrForObjectDetection": RTDetrForObjectDetectionLoss,
    "RTDetrV2ForObjectDetection": RTDetrForObjectDetectionLoss,
    "DFineForObjectDetection": DFineForObjectDetectionLoss,
    "Deimv2ForObjectDetection": Deimv2ForObjectDetectionLoss,
    "CsmForConditionalGeneration": ForCausalLMLoss,
    "SwitchTransformersForConditionalGeneration": ForMaskedLMLoss,
    "NllbMoeForConditionalGeneration": ForMaskedLMLoss,
    "LwDetrForObjectDetection": LwDetrForObjectDetectionLoss,
    "PocketTTSForConditionalGeneration": PocketTTSForConditionalGenerationLoss,
    "GradiumPhononTTSForConditionalGeneration": PocketTTSForConditionalGenerationLoss,
    "ParakeetForRNNT": ParakeetForRNNTLoss,
    "ParakeetForTDT": ParakeetForTDTLoss,
    "RfDetrForObjectDetection": LwDetrForObjectDetectionLoss,
    "RfDetrForInstanceSegmentation": RfDetrForSegmentationLoss,
}
