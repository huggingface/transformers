# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert SigLIP checkpoints from the original repository.

URL: https://github.com/google-research/big_vision/tree/main
"""


import argparse
import collections
import numpy as np

import torch

from numpy import load
from PIL import Image

from transformers import PalmaForConditionalGeneration, PalmaConfig, AutoTokenizer
from transformers.utils import logging
from PIL import Image
import numpy as np

from transformers.image_processing_utils import BatchFeature


import torch

from transformers.image_transforms import to_channel_dimension_format, normalize, rescale
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    infer_channel_dimension_format,

)

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_palma_config():
    config = PalmaConfig()

    vocab_size = 257152
    image_size = 224
    patch_size = 14

    # size of the architecture
    config.vision_config.image_size = image_size
    config.vision_config.patch_size = patch_size
    config.text_config.vocab_size = vocab_size

    #elif "so400m" in model_name:

    config.vision_config.hidden_size = 1152
    config.vision_config.intermediate_size = 4304
    config.vision_config.num_hidden_layers = 27
    config.vision_config.num_attention_heads = 16
    """
    else:
        raise ValueError("Model not supported")
    """
    vocab_size = 257152
    config.text_config.vocab_size = vocab_size
    config.text_config.num_hidden_layers = 18
    return config


def slice_state_dict(state_dict, config):
    # patch embeddings
    state_dict["vision_model.embeddings.patch_embedding.weight"] = state_dict.pop("img/embedding/kernel").transpose(3, 2, 0, 1)
    state_dict["vision_model.embeddings.patch_embedding.bias"] = state_dict.pop("img/embedding/bias")
    # positional embeddings
    state_dict["vision_model.embeddings.position_embedding.weight"] = state_dict.pop("img/pos_embedding").reshape(-1, config.vision_config.hidden_size)


    # fmt: off
    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias")

    encoderblock_mlp_dense0_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel")
    encoderblock_mlp_dense0_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias")
    encoderblock_mlp_dense1_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")
    encoderblock_mlp_dense1_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias")

    encoderblock_attention_0_key_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel")
    encoderblock_attention_0_key_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias")
    encoderblock_attention_0_value_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel")
    encoderblock_attention_0_value_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias")
    encoderblock_attention_0_query_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel")
    encoderblock_attention_0_query_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias")
    encoderblock_attention_0_out_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel")
    encoderblock_attention_0_out_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias")


    for i in range(config.vision_config.num_hidden_layers):
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]

        state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)


    state_dict["vision_model.post_layernorm.weight"] = state_dict.pop("img/Transformer/encoder_norm/scale").transpose()
    state_dict["vision_model.post_layernorm.bias"] = state_dict.pop("img/Transformer/encoder_norm/bias")

    # multimodal projector
    
    state_dict['multi_modal_projector.linear.weight'] = state_dict.pop("img/head/kernel").transpose()
    state_dict['multi_modal_projector.linear.bias'] = state_dict.pop("img/head/bias")

    # text decoder (gemma)

    state_dict["language_model.model.embed_tokens.weight"] = state_dict.pop("llm/embedder/input_embedding")
    state_dict["language_model.lm_head.weight"] = state_dict["language_model.model.embed_tokens.weight"] # weights are tied
    state_dict["language_model.model.norm.weight"] = state_dict.pop("llm/final_norm/scale")
    
    # pop the einsum attention + mlp representations. There are 18 layers in gemma-2b.

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w")
    
    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear")
    # TODO verify correctness of layer norm loading

    llm_input_layernorm = state_dict.pop("llm/layers/pre_attention_norm/scale")
    llm_post_attention_layernorm = state_dict.pop("llm/layers/pre_ffw_norm/scale")


    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = llm_attention_q_einsum[i].reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size).transpose()
        state_dict[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].reshape(config.text_config.num_key_value_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"language_model.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].reshape(config.text_config.num_key_value_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"language_model.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].reshape(config.text_config.hidden_size, config.text_config.num_attention_heads * config.text_config.head_dim).transpose()
        state_dict[f"language_model.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"language_model.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"language_model.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        # matches
        state_dict[f"language_model.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"language_model.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"language_model.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]
    # fmt: on
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)
    return state_dict


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def verify_logits(model):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # First load intermediates given, and test prompt

    # all intermediates activations
    intermediates_path = "/home/pablo/.cache/huggingface/hub/models--gv-hf--test/snapshots/58b24b23afbeb278bfa3aa3b9f0fc1e60b9cbcc5/hf_test_ckpt.cow_beach_1.bv.intermediates.npz"
    
    
    cow_on_beach_path = "/home/pablo/.cache/huggingface/hub/models--gv-hf--test/snapshots/58b24b23afbeb278bfa3aa3b9f0fc1e60b9cbcc5/cow_beach_1.png"

    # test prompt
    prompt = "answer en Where is the cow standing?\n"
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    
    intermediates = np.load(intermediates_path)


    # These steps mimic what should be taken in the processor - for an image we don't resize (given image is already 224px)
    # TODO replace by a proper Processor() call when tests pass
    img = np.array(Image.open(cow_on_beach_path))
    img = img
    img = np.expand_dims(img, 0)
    images = img.astype(np.float32)

    input_data_format = infer_channel_dimension_format(images[0])
    images = [
            rescale(image=image, scale=1/255, input_data_format=input_data_format)
            for image in images
    ]

    images = [
        normalize(image=image, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD, input_data_format=input_data_format)
        for image in images
    ]
    images = [
        to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format) for image in images
    ]

    image_tensor = BatchFeature(data={"pixel_values": images}, tensor_type='pt')

    with torch.inference_mode():
        vision_outputs =  model.vision_model(pixel_values=image_tensor['pixel_values'], output_hidden_states=True).last_hidden_state
        projector_output = model.multi_modal_projector(vision_outputs)
    
    if not np.allclose(projector_output.cpu().numpy()[0], intermediates['img/zimg'][0], rtol=1e-3, atol=5e-3):
        raise ValueError("image activations do not match.")
    
@torch.no_grad()
def convert_palma_checkpoint(checkpoint_path, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """

    # define default SigLIP configuration
    config = get_palma_config()

    # get checkpoint (move to args)
    checkpoint_path = "/home/pablo/.cache/huggingface/hub/models--gv-hf--test/snapshots/58b24b23afbeb278bfa3aa3b9f0fc1e60b9cbcc5/hf_test_ckpt.bv.params.npz" # model_name_to_checkpoint[model_name]
    # load HuggingFace model
    model = PalmaForConditionalGeneration(config).eval()
    # load original state dict
    data = load(checkpoint_path)
    state_dict = flatten_nested_dict(data)

    state_dict_transformers = slice_state_dict(state_dict, config)
    model.load_state_dict(state_dict_transformers)
    model.save_pretrained(pytorch_dump_folder_path, max_shard_size='2GB', safe_serialization=True)

    if verify_logits:
       verify_logits(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="",
        type=str,
        help="Path to the .npz checkpoint",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        help="Whether to verify logits against the original implementation.",
    )

    args = parser.parse_args()
    convert_palma_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.verify_logits)
