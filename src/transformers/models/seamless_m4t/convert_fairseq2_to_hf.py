# coding=utf-8
# Copyright 2023 ylacombe The HuggingFace Inc. team. All rights reserved.
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
""" Converting Meta SeamlessM4T checkpoints from seamless_communication to HF. """


import argparse
import os
from pathlib import Path

import torch

from transformers.utils import logging
from transformers import set_seed, Wav2Vec2ConformerModel, Wav2Vec2ConformerConfig

from seamless_communication.models.inference.translator import Translator

from huggingface_hub import HfApi, login
api = HfApi()

def assert_param_count(model_1, model_2):
    count_1 = sum(p.numel() for p in model_1.parameters())
    count_2 = sum(p.numel() for p in model_2.parameters())
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"

def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

new_layer_name_dict = {
    "c_attn": "att_proj",
    "c_proj": "out_proj",
    "c_fc": "in_proj",
    "transformer.": "",
    "h.": "layers.",
    "ln_1": "layernorm_1",
    "ln_2": "layernorm_2",
    "ln_f": "layernorm_final",
    "wpe": "position_embeds_layer",
    "wte": "input_embeds_layer",
}

wav2vec_convert_dict = {
    "speech_encoder_frontend.model_dim_proj": "feature_projection.projection",
    "speech_encoder_frontend.post_extract_layer_norm": "feature_projection.layer_norm",
    "speech_encoder_frontend.pos_encoder.conv": "encoder.pos_conv_embed.conv",
    
    "speech_encoder.inner.layers": "encoder.layers",
    "inner_proj": "intermediate_dense",
    "out_proj": "output_dense",
    
    "self_attn.k_proj": "self_attn.linear_k",
    "self_attn.v_proj": "self_attn.linear_v",
    "self_attn.q_proj": "self_attn.linear_q",
    
    "self_attn.sdpa.u_bias": "self_attn.pos_bias_u",
    "self_attn.sdpa.v_bias": "self_attn.pos_bias_v",
    "self_attn.output_proj": "self_attn.linear_out",
    "self_attn.sdpa.r_proj": "self_attn.linear_pos",
    
    "conv.pointwise_conv1": "conv_module.pointwise_conv1",
    "conv.pointwise_conv2": "conv_module.pointwise_conv2",
    "conv.depthwise_conv": "conv_module.depthwise_conv",
    "conv.batch_norm": "conv_module.batch_norm",
    "conv_layer_norm": "conv_module.layer_norm",
    

    #"layer_norm": "encoder.layers.*.final_layer_norm",
    #"inner.layer_norm": "encoder.layer_norm",
}


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


def _load_original_model(device):


    unity_hub = Translator(
        "multitask_unity", "vocoder_36langs", device
    )
    
    return unity_hub


def _load_hf_wav2vec(device, config_dict = None):
    if config_dict is None:
        config_dict = {}
        
    
    config = Wav2Vec2ConformerConfig(
        **config_dict,
        hidden_act="swish"
    )


    hf_wav2vec = Wav2Vec2ConformerModel(config).to(device)
    
    return hf_wav2vec
    


def _convert_wav2vec(original_model, device):

    hf_model = _load_hf_wav2vec()

    state_dict = original_model.state_dict()
    
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            # replace part of the key with corresponding layer name in HF implementation
            new_k = k[len(unwanted_prefix) :]
            for old_layer_name in new_layer_name_dict:
                new_k = new_k.replace(old_layer_name, new_layer_name_dict[old_layer_name])

            state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=False)
    n_params = hf_model.num_parameters(exclude_embeddings=True)

    logger.info(f"model loaded: {round(n_params/1e6,1)}M params loss")
    
    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


def load_model(pytorch_dump_folder_path):


    device = _grab_best_device()
    original_model = _load_original_model()
    
    wav2vec = _convert_wav2vec(original_model, device)
    
    
    
    new_model = ...


    if original_model.num_parameters(exclude_embeddings=True) != new_model.get_num_params():
        raise ValueError("initial and new models don't have the same number of parameters")

    # check if same output as the bark model
    batch_size = 5
    sequence_length = 10

    output_new_model = ...
    output_old_model = ...

    # output difference should come from the difference of self-attention implementation design
    if output_new_model.shape != output_old_model.shape:
        raise ValueError("initial and new outputs don't have the same shape")
    if (output_new_model - output_old_model).abs().max().item() > 1e-3:
        raise ValueError("initial and new outputs are not equal")

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    new_model.save_pretrained(pytorch_dump_folder_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("pytorch_dump_folder_path", default="/home/yoach/m4t_weights", type=str, help="Path to the output PyTorch model.")

    args = parser.parse_args()

    load_model(args.pytorch_dump_folder_path)
