# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.model.multimodal_encoder.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_emb
)
import numpy as np


class SoundTower(nn.Module):
    def __init__(self, sound_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.sound_tower_name = sound_tower
        self.cfg_only = None
        self.pos_emb = RotaryEmbedding(
                    dim = 256,
                    freqs_for = 'lang',
                    max_time = 1200.
                    )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def forward(self, sounds, mask=None, times=None):
        self.pos_emb = self.pos_emb.to(sounds.device)
        if type(sounds) is list:
            sound_features = []
            for sound in sounds:
                # Calculate attention mask
                audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(mask.sum(-1))
                # for cases where only one window is there for the audio_clip
                batch_size, _, max_mel_seq_len = sound.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                seq_range = (
                        torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                        .unsqueeze(0)
                        .expand(batch_size, max_seq_len)
                    )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                padding_mask = seq_range >= lengths_expand
                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                        batch_size, 1, max_seq_len, max_seq_len
                    )
                audio_attention_mask = audio_attention_mask_.to(
                        dtype=self.sound_tower.conv1.weight.dtype, device=self.sound_tower.conv1.weight.device
                    )
                audio_attention_mask[audio_attention_mask_] = float("-inf")
                # Calculate features
                sound_feature = self.sound_tower(sound, attention_mask=audio_attention_mask)
                sound_feature = sound_feature.to(sound.dtype)
                sound_feature = sound_feature.last_hidden_state
                times = times.to(sound_feature.device)
                freqs = self.pos_emb.get_axial_freqs(times.shape[0], sound_feature.shape[-2]).to(self.sound_tower.conv1.weight.device)
                angle = (-times * 2 * np.pi).to(self.sound_tower.conv1.weight.device)
                angle_expanded = angle.unsqueeze(1).unsqueeze(2)
                angle_expanded = angle_expanded.expand(times.shape[0], sound_feature.shape[-2], freqs.shape[-1])

                freqs = freqs * angle_expanded
                # print(freqs.shape)
                sound_feature = apply_rotary_emb(freqs, sound_feature.unsqueeze(0))

                sound_features.append(sound_feature)
        else:
            # Calculate attention mask
            if len(sounds.shape) == 5:
                sounds = sounds.squeeze(1).squeeze(1)
                mask = mask.squeeze(0)
                
            audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(mask.sum(-1))
            # for cases where only one window is there for the audio_clip
            
            batch_size, _, max_mel_seq_len = sounds.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
            if len(audio_feat_lengths.shape) == 1:
                audio_feat_lengths = audio_feat_lengths.unsqueeze(1)
            elif len(audio_feat_lengths.shape) == 3: # hard-coded, check logic
                audio_feat_lengths = audio_feat_lengths.squeeze(-1)

            lengths_expand = audio_feat_lengths.expand(batch_size, max_seq_len)
            padding_mask = seq_range >= lengths_expand
            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
            audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.sound_tower.conv1.weight.dtype, device=self.sound_tower.conv1.weight.device
                )
            audio_attention_mask[audio_attention_mask_] = float("-inf")
            # Calculate features
            sound_features = self.sound_tower(sounds, attention_mask=audio_attention_mask)
            sound_features = sound_features.last_hidden_state
            times = times.to(sound_features.device)
            freqs = self.pos_emb.get_axial_freqs(times.shape[0], sound_features.shape[-2]).to(self.sound_tower.conv1.weight.device)
            angle = (-times * 2 * np.pi).to(self.sound_tower.conv1.weight.device)
            angle_expanded = angle.unsqueeze(2)
            angle_expanded = angle_expanded.expand(times.shape[0], sound_features.shape[-2], freqs.shape[-1])

            freqs = freqs * angle_expanded
            # print(freqs.shape)
            sound_features = apply_rotary_emb(freqs, sound_features)
            sound_features = sound_features.to(sounds.dtype)

        return sound_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.sound_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.sound_tower.config
        else:
            return self.cfg_only
            
    @property
    def device(self):
        return self.sound_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size


