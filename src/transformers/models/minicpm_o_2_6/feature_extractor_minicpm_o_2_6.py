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
from typing import Optional, Union

import numpy as np
import torch

from ..whisper.feature_extraction_whisper import WhisperFeatureExtractor


class MiniCPM_o_2_6FeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_audios(self, audios):
        """
        Normalize audios format to list of list of numpy arrays.

        Args:
            audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]]

        Returns:
            List[List[np.ndarray]]: Normalized audio format
        """
        # in batch inference, it may be [[]]
        if isinstance(audios, np.ndarray):
            return [[audios]]
        elif isinstance(audios[0], np.ndarray):
            return [audios]
        else:
            return audios

    def __call__(
        self,
        audios: Union[np.ndarray, list[np.ndarray], list[list[np.ndarray]]],
        audio_parts: Optional[list] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ):
        audios_list = self.format_audios(audios)

        if audio_parts is not None:
            assert len(audio_parts) == len(audios_list)
            for parts, audios in zip(audio_parts, audios_list):
                assert len(parts) == len(audios)

        audio_feature_lens_list = []
        audio_features_all = []

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
                        final_merge_audio.append(audio[i * max_audio_inp_len: (i + 1) * max_audio_inp_len])

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

        return audio_features, audio_feature_lens_list


__all__ = ["MiniCPM_o_2_6FeatureExtractor"]
