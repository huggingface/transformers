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

from transformers.models.seamless_m4t.configuration_seamless_m4t import SeamlessM4TConfig
from transformers.models.seamless_m4t.modeling_seamless_m4t import SeamlessM4TModel
from transformers.utils import logging

import tempfile

api = HfApi()


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# order is important
wav2vec_convert_list = [
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
    ("speech_encoder.proj", "proj"),
    ("speech_encoder.layer_norm", "inner_layer_norm"),
    # "layer_norm", "encoder.layers.*.final_layer_norm",
    # "inner.layer_norm", "encoder.layer_norm",
]

t2u_convert_list = [
    ("t2u_model.final_proj", "lm_head"),
    ("t2u_model.", "model."),
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    ("encoder_decoder_attn", "cross_attention"),
    ("linear_k", "k_proj"),
    ("linear_v", "v_proj"),
    ("linear_q", "q_proj"),
    ("ffn.inner_proj", "ffn.fc1"),
    ("ffn.output_proj", "ffn.fc2"),
    ("output_proj", "out_proj"),
    ("decoder_frontend.embed", "decoder.embed_tokens"),
]

text_convert_list = [
    ("text_encoder.", ""),
    ("text_decoder.", ""),
    ("text_encoder_frontend.embed", "embed_tokens"),
    ("text_decoder_frontend.embed", "embed_tokens"),
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    ("encoder_decoder_attn", "cross_attention"),
    ("linear_k", "k_proj"),
    ("linear_v", "v_proj"),
    ("linear_q", "q_proj"),
    ("ffn.inner_proj", "ffn.fc1"),
    ("ffn.output_proj", "ffn.fc2"),
    ("output_proj", "out_proj"),
    ("final_proj", "lm_head"),
]

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


def _load_original_model(device):
    unity_hub = Translator("multitask_unity", "vocoder_36langs", device)

    return unity_hub


def _convert_model(
    original_model,
    hf_model,
    convert_list,
    device,
    unwanted_prefix="model.",
    filter_state_dict="speech",
    exclude_state_dict=None,
):
    state_dict = original_model.state_dict()

    # filter func
    if isinstance(filter_state_dict, str):
        def filter_func(x):
            return filter_state_dict in x[0]
    else:

        def filter_func(item):
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            for filter_el in filter_state_dict:
                if filter_el in item[0]:
                    return True

            return False

    state_dict = dict(filter(filter_func, state_dict.items()))

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
    missing_keys = set({k for k in missing_keys if "final_logits_bias" not in k})
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=False)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


def load_model(pytorch_dump_folder_path):
    """
    Meta SeamlessM4T is made of 7 main components:
    - speech_encoder (#1) and speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) and text_encoder_frontend (#5)
    - text_decoder (#6) [and text_decoder_frontend (#5) = equals to text_encoder_frontend]
    - final_proj (#7)
    """
    device = _grab_best_device()
    original_model = _load_original_model(device)

    # init model
    hf_config = SeamlessM4TConfig(
    )
    hf_model = SeamlessM4TModel(hf_config)

    # 1. take care of speech encoder
    wav2vec = hf_model.speech_encoder
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # verify same number of parameters speech encoder
    count_1 = param_count(hf_model.speech_encoder)
    count_2 = param_count(original_model.model.speech_encoder_frontend) + param_count(
        original_model.model.speech_encoder
    )

    assert count_1 == count_2, f"Speech Encoder --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 2. take care of t2u

    hf_model.t2u_model = _convert_model(
        original_model,
        hf_model.t2u_model,
        t2u_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict="t2u_model",
    )

    # verify same number of parameters t2u model
    count_1 = param_count(hf_model.t2u_model)
    count_2 = param_count(original_model.model.t2u_model)

    assert count_1 == count_2, f"T2U model --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 3. take care of text encoder
    hf_model.text_encoder = _convert_model(
        original_model,
        hf_model.text_encoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_encoder"],
        exclude_state_dict="t2u_model",
    )

    # verify same number of parameters text_encoder
    count_1 = param_count(hf_model.text_encoder)
    count_2 = param_count(original_model.model.text_encoder) + param_count(original_model.model.text_encoder_frontend)

    assert count_1 == count_2, f"Text encoder model --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 4. take care of text decoder
    hf_model.text_decoder = _convert_model(
        original_model,
        hf_model.text_decoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_decoder"],
        exclude_state_dict="t2u_model",
    )

    # verify same number of parameters text_decoder
    count_1 = param_count(hf_model.text_decoder)
    count_2 = param_count(original_model.model.text_decoder) + param_count(original_model.model.text_decoder_frontend)
    
    with tempfile.TemporaryDirectory() as tmpdirname:

        hf_model.save_pretrained(tmpdirname)
        hf_model = SeamlessM4TModel.from_pretrained(tmpdirname)

    assert count_1 == count_2, f"Text decoder model --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 5. take care of final proj
    hf_model.lm_head = _convert_model(
        original_model,
        hf_model.lm_head,
        [("final_proj.", "")],
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.final_proj"],
        exclude_state_dict="t2u_model",
    )

    # verify same number of parameters final proj
    count_1 = param_count(hf_model.lm_head)
    count_2 = param_count(original_model.model.final_proj)

    assert count_1 == count_2, f"final proj --- Count HF: {count_1} != Count Seamless: {count_2}"

    new_model = hf_model

    # verify that base model have same number of parameters
    assert_param_count(original_model.model, new_model)

    # if not assert_param_count(original_model, new_model):
    #    raise ValueError("initial and new models don't have the same number of parameters")

    # check if same output as the bark model

    # TODO
    hf_model.num_parameters(exclude_embeddings=True)

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
