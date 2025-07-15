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

from transformers import LlamaTokenizerFast


class MiniCPM_V_4TokenizerFast(LlamaTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
    def newline_id(self):
        return self.convert_tokens_to_ids('\n')

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text


__all__ = ["MiniCPM_V_4TokenizerFast"]
