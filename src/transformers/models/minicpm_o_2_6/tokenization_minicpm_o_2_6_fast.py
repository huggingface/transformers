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

from ..qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


class MiniCPM_o_2_6TokenizerFast(Qwen2TokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # image
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"
        self.im_id_start = "<image_id>"
        self.im_id_end = "</image_id>"
        self.image_tag = f"({self.im_start}./{self.im_end})"
        self.image_pattern = "\(<image>./</image>\)"

        # audio
        self.audio_start = "<|audio_start|>"
        self.audio_end = "<|audio_end|>"
        self.spk_start = "<|spk_bos|>"
        self.spk_end = "<|spk_eos|>"
        self.tts_start = "<|tts_bos|>"
        self.tts_end = "<|tts_eos|>"
        self.unk_token = "<unk>"
        self.audio_tag = "(<audio>./</audio>)"
        self.audio_pattern = "\(<audio>./</audio>\)"

        self.split_pattern = f"({self.image_pattern}|{self.audio_pattern})"

        self.terminator_tokens = ["<|im_end|>", "<|endoftext|>", self.tts_end]

    @property
    def eos_id(self):
        return self.eos_token_id

    @property
    def bos_id(self):
        return self.bos_token_id

    @property
    def unk_id(self):
        return self.unk_token_id

    @property
    def terminators(self):
        return self.terminator_tokens

    @property
    def im_start_id(self):
        return self.convert_tokens_to_ids(self.im_start)

    @property
    def im_end_id(self):
        return self.convert_tokens_to_ids(self.im_end)

    @property
    def slice_start_id(self):
        return self.convert_tokens_to_ids(self.slice_start)

    @property
    def slice_end_id(self):
        return self.convert_tokens_to_ids(self.slice_end)

    @property
    def im_id_start_id(self):
        return self.convert_tokens_to_ids(self.im_id_start)

    @property
    def im_id_end_id(self):
        return self.convert_tokens_to_ids(self.im_id_end)

    @property
    def audio_start_id(self):
        return self.convert_tokens_to_ids(self.audio_start)

    @property
    def audio_end_id(self):
        return self.convert_tokens_to_ids(self.audio_end)

    @property
    def spk_start_id(self):
        return self.convert_tokens_to_ids(self.spk_start)

    @property
    def spk_end_id(self):
        return self.convert_tokens_to_ids(self.spk_end)

    @property
    def tts_start_id(self):
        return self.convert_tokens_to_ids(self.tts_start)

    @property
    def tts_end_id(self):
        return self.convert_tokens_to_ids(self.tts_end)

    @property
    def terminator_ids(self):
        return [self.convert_tokens_to_ids(t) for t in self.terminator_tokens]

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text


__all__ = ["MiniCPM_o_2_6TokenizerFast"]
