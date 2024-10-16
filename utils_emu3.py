# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
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
""" Logits Processor Helper class for Emu3. """

import torch

class Emu3PrefixConstrainedLogitsHelper:

    def __init__(
        self,
        height,
        width,
        img_token,
        eoi_token,
        eos_token,
        eol_token,
        eof_token,
        pad_token,
        visual_tokens,
    ):
        self.height = height
        self.width = width
        self.img_token = img_token
        self.eoi_token = eoi_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.pad_token = pad_token
        self.visual_tokens = visual_tokens

        self.offset_cache = {}

    def __call__(self, batch_id, input_ids):
        if batch_id not in self.offset_cache:
            position = torch.nonzero(input_ids == self.img_token, as_tuple=True)[0][0]
            self.offset_cache[batch_id] = position

        offset = input_ids.shape[0] - self.offset_cache[batch_id]
        if offset % (self.width + 1) == 0:
            return (self.eol_token, )
        elif offset == (self.width + 1) * self.height + 1:
            return (self.eof_token, )
        elif offset == (self.width + 1) * self.height + 2:
            return (self.eoi_token, )
        elif offset == (self.width + 1) * self.height + 3:
            return (self.eos_token, )
        elif offset > (self.width + 1) * self.height + 3:
            return (self.pad_token, )
        else:
            return self.visual_tokens
