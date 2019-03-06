# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Utilities for PyTorch Transformer XL model.
    Directly adapted from https://github.com/kimiyoung/transformer-xl.
"""

from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA_MAJOR = int(torch.version.cuda.split('.')[0])
# CUDA_MINOR = int(torch.version.cuda.split('.')[1])

class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target=None, keep_order=False):
        '''
            Params:
                hidden :: [len*bsz x d_proj]
                target :: [len*bsz]
            Return:
                if target is None:
                    out :: [len*bsz] Negative log likelihood
                else:
                    out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary
            We could replace this implementation by the native PyTorch one
            if their's had an option to set bias on all clusters in the native one.
            here: https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        '''

        if target is not None:
            target = target.view(-1)
            if hidden.size(0) != target.size(0):
                raise RuntimeError('Input and target should have the same size '
                                'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            if target is not None:
                output = -F.log_softmax(logit, dim=-1) \
                        .gather(1, target.unsqueeze(1)).squeeze(1)
            else:
                output = F.log_softmax(logit, dim=-1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            if target is None:
                out = hidden.new_empty((head_logit.size(0), self.n_token))
            else:
                out = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                if target is not None:
                    mask_i = (target >= l_idx) & (target < r_idx)
                    indices_i = mask_i.nonzero().squeeze()

                    if indices_i.numel() == 0:
                        continue

                    target_i = target.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = hidden.index_select(0, indices_i)
                else:
                    hidden_i = hidden

                if i == 0:
                    if target is not None:
                        logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1  # No probability for the head cluster
                    if target is not None:
                        logprob_i = head_logprob_i[:, cluster_prob_idx] \
                                + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, cluster_prob_idx, None] + tail_logprob_i
                        out[:, l_idx:r_idx] = logprob_i

                if target is not None:
                    if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                        out.index_copy_(0, indices_i, -logprob_i)
                    else:
                        out[offset:offset+logprob_i.size(0)].copy_(-logprob_i)
                    offset += logprob_i.size(0)

        return out


    def log_prob(self, hidden):
        r""" Computes log probabilities for all :math:`n\_classes`
        From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py
        Args:
            hidden (Tensor): a minibatch of examples
        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= n\_classes`, where :math:`n\_classes` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.
        Shape:
            - Input: :math:`(N, in\_features)`
            - Output: :math:`(N, n\_classes)`
        """
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            return F.log_softmax(logit, dim=-1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)

            out = hidden.new_empty((head_logit.size(0), self.n_token))
            head_logprob = F.log_softmax(head_logit, dim=1)

            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]

                if i == 0:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    tail_logit_i = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob[:, -i] + tail_logprob_i
                    out[:, start_idx, stop_idx] = logprob_i

            return out


class LogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1., range_max+2., 1.).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            # print('P', self.dist.numpy().tolist()[-30:])

            self.log_q = (- (-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples

def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[: -n_sample].view(b1, b2, -1)
    sample_w = all_w[- n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[: -n_sample].view(b1, b2)
    sample_b = all_b[- n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = torch.einsum('ijk,ijk->ij',
        [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl',
        [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

    return logits


# class LogUniformSampler(object):
#     def __init__(self, range_max, unique=False):
#         """
#         Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
#             `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`
#         """
#         self.range_max = range_max
#         log_indices = torch.arange(1., range_max+2., 1.).log_()
#         self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

#         self.unique = unique

#         if self.unique:
#             self.exclude_mask = torch.ByteTensor(range_max).fill_(0)

#     def sample(self, n_sample, labels):
#         pos_sample, new_labels = labels.unique(return_inverse=True)
#         n_pos_sample = pos_sample.size(0)
#         n_neg_sample = n_sample - n_pos_sample

#         if self.unique:
#             self.exclude_mask.index_fill_(0, pos_sample, 1)
#             sample_dist = self.dist.clone().masked_fill_(self.exclude_mask, 0)
#             self.exclude_mask.index_fill_(0, pos_sample, 0)
#         else:
#             sample_dist = self.dist

#         neg_sample = torch.multinomial(sample_dist, n_neg_sample)

#         sample = torch.cat([pos_sample, neg_sample])
#         sample_prob = self.dist[sample]

#         return new_labels, sample, sample_prob


if __name__ == '__main__':
    S, B = 3, 4
    n_vocab = 10000
    n_sample = 5
    H = 32

    labels = torch.LongTensor(S, B).random_(0, n_vocab)

    # sampler = LogUniformSampler(n_vocab, unique=False)
    # new_labels, sample, sample_prob = sampler.sample(n_sample, labels)

    sampler = LogUniformSampler(n_vocab, n_sample)#, unique=True)
    # true_probs, samp_probs, neg_samples = sampler.sample(n_sample, labels)

    # print('true_probs', true_probs.numpy().tolist())
    # print('samp_probs', samp_probs.numpy().tolist())
    # print('neg_samples', neg_samples.numpy().tolist())

    # print('sum', torch.sum(sampler.dist).item())

    # assert torch.all(torch.sort(sample.unique())[0].eq(torch.sort(sample)[0])).item()

    embedding = nn.Embedding(n_vocab, H)
    bias = torch.zeros(n_vocab)
    inputs = torch.Tensor(S, B, H).normal_()

    logits, out_labels = sample_logits(embedding, bias, labels, inputs, sampler, n_sample)
    print('logits', logits.detach().numpy().tolist())
    print('logits shape', logits.size())
    print('out_labels', out_labels.detach().numpy().tolist())
    print('out_labels shape', out_labels.size())

