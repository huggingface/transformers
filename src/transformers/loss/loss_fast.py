# coding=utf-8
# Copyright 2025 the Fast authors and HuggingFace Inc. team.  All rights reserved.
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
import torch.nn.functional as F


def FastForSceneTextRecognitionLoss(logits: torch.Tensor, labels: torch.Tensor, texts):
    """
    Computes the overall loss for fast.
    """
    gt_texts = labels["gt_texts"]
    gt_kernels = labels["gt_kernels"]
    training_masks = labels["training_masks"]
    gt_instances = labels["gt_instances"]

    kernels = logits[:, 0, :, :]  # 4*640*640
    embs = logits[:, 1:, :, :]  # 4*4*640*640

    selected_masks = ohem_batch(texts, gt_texts, training_masks)
    loss_text = dice_loss_with_masks(texts, gt_texts, selected_masks, reduce=False)
    selected_masks = gt_texts * training_masks
    loss_kernel = dice_loss_with_masks(kernels, gt_kernels, selected_masks, reduce=False)
    loss_kernel = torch.mean(loss_kernel, dim=0)

    loss_emb = emb_loss_batch(embs, gt_instances, gt_kernels, training_masks, reduce=False)
    return torch.mean(loss_text) + torch.mean(loss_kernel) + torch.mean(loss_emb)


def preprocess_inputs(emb: torch.Tensor, instance: torch.Tensor, kernel: torch.Tensor, training_mask: torch.Tensor):
    """
    Preprocess input tensors by applying masks and reshaping them.
    """
    training_mask = (training_mask > 0.5).long()
    kernel = (kernel > 0.5).long()
    instance = instance * training_mask
    instance_kernel = (instance * kernel).view(-1)
    instance = instance.view(-1)
    emb = emb.view(emb.shape[0], -1)
    return emb, instance, instance_kernel


def compute_embedding_means(emb: torch.Tensor, instance_kernel: torch.Tensor, unique_labels: torch.Tensor):
    """
    Compute the mean embedding for each instance.
    """
    num_instance = unique_labels.size(0)
    emb_mean = emb.new_zeros((emb.shape[0], num_instance), dtype=torch.float32)
    for i, label in enumerate(unique_labels):
        if label == 0:
            continue
        indices = instance_kernel == label
        emb_mean[:, i] = torch.mean(emb[:, indices], dim=1)
    return emb_mean


def compute_aggregation_loss(
    emb: torch.Tensor, instance: torch.Tensor, emb_mean: torch.Tensor, unique_labels: torch.Tensor, delta_v: float
):
    """
    Compute the aggregation loss, which ensures that embeddings are close to their instance mean.
    """
    num_instance = unique_labels.size(0)
    aggregation_loss = emb.new_zeros(num_instance, dtype=torch.float32)
    for i, label in enumerate(unique_labels):
        if label == 0:
            continue
        indices = instance == label
        instance_embeddings = emb[:, indices]
        distance = (instance_embeddings - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
        distance = F.relu(distance - delta_v) ** 2
        aggregation_loss[i] = torch.mean(torch.log(distance + 1.0))
    return torch.mean(aggregation_loss[1:])


def compute_discriminative_loss(emb_mean: torch.Tensor, unique_labels: torch.Tensor, delta_d: float, bg_sample: bool):
    """
    Compute the discriminative loss, which ensures that different instances are separated.
    """
    num_instance = unique_labels.size(0)
    if num_instance <= 2:
        return 0

    emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
    emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, emb_mean.shape[0])

    mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, emb_mean.shape[0])
    mask = mask.view(num_instance, num_instance, -1)
    mask[0, :, :] = 0
    mask[:, 0, :] = 0
    mask = mask.view(num_instance * num_instance, -1)

    distance = emb_interleave - emb_band
    distance = distance[mask > 0].view(-1, emb_mean.shape[0]).norm(p=2, dim=1)
    distance = F.relu(2 * delta_d - distance) ** 2
    discriminative_loss = torch.mean(torch.log(distance + 1.0))

    if bg_sample:
        bg_loss = [torch.log(distance + 1.0)]
        return torch.mean(torch.cat(bg_loss))
    return discriminative_loss


def emb_loss(
    emb: torch.Tensor,
    instance: torch.Tensor,
    kernel: torch.Tensor,
    training_mask: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    weights: tuple[float, float] = (1.0, 1.0),
    bg_sample: bool = False,
) -> torch.Tensor:
    """
    Computes the embedding loss based on variance and distance constraints.
    """
    emb, instance, instance_kernel = preprocess_inputs(emb, instance, kernel, training_mask)
    unique_labels, _ = torch.unique(instance_kernel, sorted=True, return_inverse=True)
    num_instance = unique_labels.size(0)

    if num_instance <= 1:
        return 0

    emb_mean = compute_embedding_means(emb, instance_kernel, unique_labels)
    aggregation_loss = compute_aggregation_loss(emb, instance, emb_mean, unique_labels, delta_v)
    discriminative_loss = compute_discriminative_loss(emb_mean, unique_labels, delta_d, bg_sample)

    regularization_loss = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001

    total_loss = (weights[0] * aggregation_loss) + (weights[1] * discriminative_loss) + regularization_loss
    return total_loss


def emb_loss_batch(emb, instance, kernel, training_mask, reduce=True, loss_weight=0.25):
    """
    Computes batch-wise embedding loss.
    """
    loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

    for i in range(loss_batch.size(0)):
        loss_batch[i] = emb_loss(emb[i], instance[i], kernel[i], training_mask[i])

    loss_batch = loss_weight * loss_batch

    if reduce:
        loss_batch = torch.mean(loss_batch)

    return loss_batch


def dice_loss_with_masks(input, target, mask, reduce=True):
    """
    Computes dice loss with masks applied.
    """
    loss_weight = 0.5
    batch_size = input.size(0)
    input = torch.sigmoid(input)

    input = input.contiguous().view(batch_size, -1)
    target = target.contiguous().view(batch_size, -1).float()
    mask = mask.contiguous().view(batch_size, -1).float()

    # we add padding if input or mask size do not match the target size
    if input.size(1) < target.size(1):
        padding_size = target.size(1) - input.size(1)
        input = F.pad(input, (0, padding_size), mode="constant", value=0)

    if mask.size(1) < target.size(1):
        padding_size = target.size(1) - mask.size(1)
        mask = F.pad(mask, (0, padding_size), mode="constant", value=0)

    input = input * mask
    target = target * mask

    intersection = torch.sum(input * target, dim=1)
    input_sum = torch.sum(input * input, dim=1) + 0.001  # prevent division by zero
    target_sum = torch.sum(target * target, dim=1) + 0.001  # prevent division by zero
    dice_score = (2 * intersection) / (input_sum + target_sum)
    dice_loss = 1 - dice_score

    loss = loss_weight * dice_loss

    if reduce:
        loss = torch.mean(loss)

    return loss


def ohem_single(score, gt_text, training_mask):
    """
    Computes Online Hard Example Mining (OHEM) for a single sample.
    """
    pos_num = int(torch.sum(gt_text > 0.5)) - int(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_num = int(torch.sum(gt_text <= 0.5))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    """
    Computes OHEM for a batch of samples.
    """
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))
    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks
