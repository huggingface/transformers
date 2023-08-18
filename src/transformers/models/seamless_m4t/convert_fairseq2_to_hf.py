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
from huggingface_hub import HfApi
from seamless_communication.models.inference.translator import Translator

from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel
from transformers.utils import logging

from .modeling_seamless_m4t import SeamlessM4TModel
from .configuration_seamless_m4t import SeamlessM4TConfig


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

# order is important
wav2vec_convert_dict = [
    ("speech_encoder_frontend.model_dim_proj", "feature_projection.projection"),
    ("speech_encoder_frontend.post_extract_layer_norm", "feature_projection.layer_norm"),
    ("speech_encoder_frontend.pos_encoder.conv", "encoder.pos_conv_embed.conv"),
    ("speech_encoder.inner.layers", "encoder.layers"),
    ("speech_encoder.inner_layer_norm", "encoder.layer_norm"),
    ("speech_encoder.adaptor_layers", "adapter.layers"),
    ("inner_proj", "intermediate_dense"),
    ("self_attn.output_proj", "self_attn.linear_out"),
    ("self_attn.output_dense", "self_attn.linear_out"),
    ("output_proj", "output_dense"),
    ("self_attn.k_proj", "self_attn.linear_k"),
    ("self_attn.v_proj", "self_attn.linear_v"),
    ("self_attn.q_proj", "self_attn.linear_q"),
    ("self_attn.sdpa.u_bias", "self_attn.pos_bias_u"),
    ("self_attn.sdpa.v_bias", "self_attn.pos_bias_v"),
    ("self_attn.output_proj", "self_attn.linear_out"),
    ("self_attn.sdpa.r_proj", "self_attn.linear_pos"),
    ("conv.pointwise_conv1", "conv_module.pointwise_conv1"),
    ("conv.pointwise_conv2", "conv_module.pointwise_conv2"),
    ("conv.depthwise_conv", "conv_module.depthwise_conv"),
    ("conv.batch_norm", "conv_module.batch_norm"),
    ("conv_layer_norm", "conv_module.layer_norm"),
    # "layer_norm", "encoder.layers.*.final_layer_norm",
    # "inner.layer_norm", "encoder.layer_norm",
]


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


def _load_original_model(device):
    unity_hub = Translator("multitask_unity", "vocoder_36langs", device)

    return unity_hub


def _load_hf_wav2vec(device, config_dict=None):
    if config_dict is None:
        config_dict = {
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "final_dropout": 0.0,
            "layerdrop": 0.0,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "max_seq_len": 4096,
            "add_adapter": True,
            "num_adapter_layers": 1,
        }

    config = Wav2Vec2ConformerConfig(**config_dict, hidden_act="swish")

    hf_wav2vec = Wav2Vec2ConformerModel(config).to(device)

    return hf_wav2vec


def _convert_model(
    original_model, hf_model, convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
):
    state_dict = original_model.state_dict()

    # filter
    state_dict = dict(filter(lambda x: filter_state_dict in x[0], state_dict.items()))

    for k, v in list(state_dict.items()):
        new_k = k[len(unwanted_prefix) :]
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        # must do it by hand
        if ".layer_norm" in new_k and new_k.split(".layer_norm")[0][-1].isnumeric():
            new_k = new_k.replace("layer_norm", "final_layer_norm")

        state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set(extra_keys)
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set(missing_keys)
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=False)
    n_params = hf_model.num_parameters(exclude_embeddings=True)

    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


def load_model(pytorch_dump_folder_path):
    device = _grab_best_device()
    original_model = _load_original_model(device)
    
    hf_config = SeamlessM4TConfig()
    hf_model = SeamlessM4TModel(hf_config)

    wav2vec = hf_model.speech_encoder
    
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_dict, device, unwanted_prefix="model.", filter_state_dict="speech"
    )



    new_model = hf_model

    if original_model.num_parameters(exclude_embeddings=True) != new_model.get_num_params():
        raise ValueError("initial and new models don't have the same number of parameters")

    # check if same output as the bark model

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

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/yoach/m4t_weights",
        type=str,
        help="Path to the output PyTorch model.",
    )

    args = parser.parse_args()

    load_model(args.pytorch_dump_folder_path)
