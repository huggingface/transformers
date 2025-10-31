# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""
Image/Text processor class for MiniCPM-O
"""
from defines import *
from omnilmm.train.train_utils import omni_preprocess
import base64

class MiniCPMoProcessing():
    def expand_question_into_multimodal(self,question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
        if '<image>' in question_text[0]['content']:
            question_text[0]['content'] = question_text[0]['content'].replace(
                '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
        else:
            question_text[0]['content'] = im_st_token + im_patch_token * \
                image_token_len + im_ed_token + '\n' + question_text[0]['content']
        return question_text

    def wrap_question_for_omni_lmm(self,question, image_token_len, tokenizer):
        question = self.expand_question_into_multimodal(
            question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

        conversation = question
        data_dict = omni_preprocess(sources=[conversation],
                                    tokenizer=tokenizer,
                                    generation=True)

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])
        return data_dict
    
    def img2base64(file_name):
        with open(file_name, 'rb') as f:
            encoded_string = base64.b64encode(f.read())
            return encoded_string

