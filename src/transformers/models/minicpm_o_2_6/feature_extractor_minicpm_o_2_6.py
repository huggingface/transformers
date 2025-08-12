# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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

import math
from typing import List, Optional, Union

from transformers import WhisperFeatureExtractor, AutoFeatureExtractor, AutoTokenizer
import numpy as np
import torch


class MiniCPM_O_2_6FeatureExtractor(WhisperFeatureExtractor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
        audio_parts: Optional[list] = None,
        chunk_input: Optional[bool] = False,
        sampling_rate: Optional[int] = None,
        chunk_length: Optional[int] = 1,
        **kwargs,
    ):
        if isinstance(audios, np.ndarray):
            audios_list = [[audios]]
        elif isinstance(audios[0], np.ndarray):
            audios_list = [audios]
        else:
            audios_list = audios

        if audio_parts is not None:
            assert len(audio_parts) == len(audios_list)
            for parts, audios in zip(audio_parts, audios_list):
                assert len(parts) == len(audios)

        audio_feature_lens_list = []
        audio_ph_list = []

        audio_features_all = []

        # audio placeholder not dependent on audio_parts
        for audios in audios_list:
            if audios:
                audio_ph_list.append([self.get_audio_placeholder(
                    len(a), chunk_input, chunk_length) for a in audios])
            else:
                audio_ph_list.append([])

        for idx, audios in enumerate(audios_list):
            if audio_parts is not None:
                # same audio part merge
                audio_part = audio_parts[idx]
                merge_audio = []
                cur_audio = []
                for aid, (part, audio) in enumerate(zip(audio_part, audios)):
                    if aid == 0 or audio_part[aid] == audio_part[aid - 1]:
                        cur_audio.append(audio)
                    else:
                        merge_audio.append(np.hstack(cur_audio))
                        cur_audio = [audio]
                if cur_audio:
                    merge_audio.append(np.hstack(cur_audio))

            else:
                merge_audio = audios

            audio_feature_lens = []

            # If the audio exceeds 30 seconds, split it into chunks every 30 seconds.
            final_merge_audio = []
            max_audio_inp_len = 30 * sampling_rate
            for audio in merge_audio:
                if len(audio) <= max_audio_inp_len:
                    final_merge_audio.append(audio)
                else:
                    for i in range(math.ceil(len(audio) / max_audio_inp_len)):
                        final_merge_audio.append(
                            audio[i * max_audio_inp_len: (i + 1) * max_audio_inp_len])

            if audios:
                audio_inputs = super().__call__(
                    final_merge_audio,
                    sampling_rate=sampling_rate,
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                    **kwargs,
                )
                audio_feature = audio_inputs["input_features"]
                actual_lens = audio_inputs["attention_mask"].sum(dim=1)

                for feat, lens in zip(audio_feature, actual_lens):
                    audio_features_all.append(feat[:, :lens])
                    audio_feature_lens.append(lens)

                audio_feature_lens = torch.hstack(audio_feature_lens)
                audio_feature_lens_list.append(audio_feature_lens)
            else:
                audio_feature_lens_list.append([])

        if audio_features_all:
            audio_features = [i.permute(1, 0) for i in audio_features_all]
            audio_features = torch.nn.utils.rnn.pad_sequence(
                audio_features, batch_first=True, padding_value=0.0
            ).permute(0, 2, 1)
        else:
            audio_features = []

        return audio_features, audio_feature_lens_list, audio_ph_list

    def get_audio_placeholder(self, audio_lens, chunk_input, chunk_length):
        pool_step = 2
        feature_lens = math.ceil(
            audio_lens / self.hop_length)

        feature_lens = (feature_lens - 1) // 2 + 1
        output_lens = (feature_lens - pool_step) // pool_step + 1

        if chunk_input:
            fbank_feat_in_chunk = int(chunk_length * 100)
            cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
            audio_embeds_in_chunk = (
                cnn_feat_in_chunk - pool_step) // pool_step + 1
            num_audio_chunks = (
                output_lens + audio_embeds_in_chunk - 1) // audio_embeds_in_chunk

            place_holders = ""
            total_unk_len = 0
            for _ in range(num_audio_chunks):
                unk_len = min(audio_embeds_in_chunk,
                              output_lens - total_unk_len)
                place_holders += self.tokenizer.audio_start + \
                    self.tokenizer.unk_token * unk_len + self.tokenizer.audio_end
                total_unk_len += unk_len
            audio_placeholder = place_holders
        else:
            audio_placeholder = self.tokenizer.audio_start + \
                self.tokenizer.unk_token * output_lens + self.tokenizer.audio_end

        return audio_placeholder


AutoFeatureExtractor.register(
    "MiniCPM_O_2_6FeatureExtractor", MiniCPM_O_2_6FeatureExtractor)

__all__ = ["MiniCPM_O_2_6FeatureExtractor"]
