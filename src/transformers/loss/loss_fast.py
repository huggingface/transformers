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
import torch.nn as nn
from typing import Tuple

class FastForSceneTextRecognitionLoss(nn.Module):
    """
    This class computes the embedding loss for Fast. It encapsulates multiple loss components such as embedding loss,
    dice loss, and OHEM loss.

    Args:
        feature_dim (int): Number of feature dimensions.
        delta_v (float): Intra-cluster variance margin.
        delta_d (float): Inter-cluster distance margin.
        weights (Tuple[float, float]): Weights for loss components.
        bg_sample (bool): Whether to sample background pixels.
    """
    
    def __init__(self, feature_dim=4, delta_v=0.5, delta_d=1.5, weights=(1.0, 1.0), bg_sample=False):
        super(FastLoss, self).__init__()
        self.feature_dim = feature_dim
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.weights = weights
        self.bg_sample = bg_sample
    
    def emb_loss(
        emb: torch.Tensor,
        instance: torch.Tensor,
        kernel: torch.Tensor,
        training_mask: torch.Tensor,
        feature_dim: int = 4,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        weights: Tuple[float, float] = (1.0, 1.0),
        bg_sample: bool = False
    ) -> torch.Tensor:
        """
        Computes the embedding loss based on variance and distance constraints.
        """
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(feature_dim, -1)

        unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((feature_dim, num_instance), dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, feature_dim)
            # print(seg_band)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            # print(mask)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * delta_d - dist) ** 2
            l_dis = torch.mean(torch.log(dist + 1.0))

            if bg_sample:
                l_dis = [torch.log(dist + 1.0)]
                emb_bg = emb[:, instance == 0].view(feature_dim, -1)
                if emb_bg.size(1) > 100:
                    rand_ind = np.random.permutation(emb_bg.size(1))[:100]
                    emb_bg = emb_bg[:, rand_ind]
                if emb_bg.size(1) > 0:
                    for i, lb in enumerate(unique_labels):
                        if lb == 0:
                            continue
                        dist = (emb_bg - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
                        dist = F.relu(2 * delta_d - dist) ** 2
                        l_dis_bg = torch.mean(torch.log(dist + 1.0), 0, keepdim=True)
                        l_dis.append(l_dis_bg)
                l_dis = torch.mean(torch.cat(l_dis))
        else:
            l_dis = 0

        l_agg = weights[0] * l_agg
        l_dis = weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss 

    
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

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = loss_weight * loss

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
    
    def forward(self, hidden: torch.Tensor, labels: torch.Tensor):
        """
        Computes the overall loss.
        """
        gt_texts = labels["gt_texts"]
        gt_kernels = labels["gt_kernels"]
        training_masks = labels["training_masks"]
        gt_instances = labels["gt_instances"]

        kernels = hidden[:, 0, :, :]  # 4*640*640
        texts = self._max_pooling(kernels, scale=1)  # 4*640*640
        embs = hidden[:, 1:, :, :]  # 4*4*640*640

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = dice_loss_with_masks(texts, gt_texts, selected_masks, reduce=False)
        selected_masks = gt_texts * training_masks
        loss_kernel = dice_loss_with_masks(kernels, gt_kernels, selected_masks, reduce=False)
        loss_kernel = torch.mean(loss_kernel, dim=0)

        loss_emb = emb_loss_batch(embs, gt_instances, gt_kernels, training_masks, reduce=False)
        return torch.mean(loss_text) + torch.mean(loss_kernel) + torch.mean(loss_emb)
