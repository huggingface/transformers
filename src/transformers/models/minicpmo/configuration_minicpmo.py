# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""ALIGN model configuration"""

from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging
import torch
import os

from omnilmm.utils import disable_torch_init
from omnilmm.model.omnilmm import OmniLMMForCausalLM
from omnilmm.model.utils import build_transform

from transformers import AutoTokenizer, AutoModel

from defines import *

logger = logging.get_logger(__name__)

class MiniCPM_oConfig(PretrainedConfig):

    model_type = "miniCPM-O"
    # base_config_key = "multimodal_config" didn't add it bc i have no idea if theres a bck that is compatible with mc-o
    


    def init_omni_lmm(model_path):
        torch.backends.cuda.matmul.allow_tf32 = True
        disable_torch_init()
        model_name = os.path.expanduser(model_path)
        print(f'Load omni_lmm model and tokenizer from {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=2048)

        if False:
            # model on multiple devices for small size gpu memory (Nvidia 3090 24G x2) 
            with init_empty_weights():
                model = OmniLMMForCausalLM.from_pretrained(model_name, tune_clip=True, torch_dtype=torch.bfloat16)
            model = load_checkpoint_and_dispatch(model, model_name, dtype=torch.bfloat16, 
                        device_map="auto",  no_split_module_classes=['Eva','MistralDecoderLayer', 'ModuleList', 'Resampler']
            )
        else:
            model = OmniLMMForCausalLM.from_pretrained(
                model_name, tune_clip=True, torch_dtype=torch.bfloat16
            ).to(device='cuda', dtype=torch.bfloat16)

        image_processor = build_transform(
            is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP')

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        assert mm_use_im_start_end

        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                            DEFAULT_IM_END_TOKEN], special_tokens=True)


        vision_config = model.model.vision_config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = model.model.config.num_query

        return model, image_processor, image_token_len, tokenizer



    

__all__ = ["MiniCPM_oConfig"]