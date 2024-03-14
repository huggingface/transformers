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

from transformers import AutoTokenizer, PalmaConfig, PalmaForConditionalGeneration
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import normalize, rescale, to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    infer_channel_dimension_format,
)
from transformers.utils import logging

device = 'cpu'

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

    # elif "so400m" in model_name:

    config.vision_config.hidden_size = 1152
    config.vision_config.intermediate_size = 4304
    config.vision_config.num_hidden_layers = 27
    config.vision_config.num_attention_heads = 16
    """
    else:
        raise ValueError("Model not supported")
    """
    config.text_config.vocab_size = vocab_size
    config.text_config.num_hidden_layers = 18
    config.text_config.num_key_value_heads = 1
    config.text_config.head_dim = 256
    config.text_config.hidden_size = 2048
    return config


def numpy_adapted_permute(w, num_heads, head_dim, hidden_size):
    #reshaped = w.transpose(0, 2, 1).reshape(num_heads, head_dim, hidden_size).reshape(hidden_size, hidden_size)
    #permuted = reshaped.reshape(num_heads, -1, 2, hidden_size).transpose(0, 2, 1, 3).reshape(hidden_size, hidden_size)

    interleaved = np.transpose(w, (1, 0, 2)).reshape(-1, num_heads * head_dim)
    return interleaved


def slice_state_dict(state_dict, config):
    # patch embeddings
    state_dict["vision_model.embeddings.patch_embedding.weight"] = state_dict.pop("img/embedding/kernel").transpose(
        3, 2, 0, 1
    )
    state_dict["vision_model.embeddings.patch_embedding.bias"] = state_dict.pop("img/embedding/bias")
    # positional embeddings
    state_dict["vision_model.embeddings.position_embedding.weight"] = state_dict.pop("img/pos_embedding").reshape(
        -1, config.vision_config.hidden_size
    )

    # fmt: off
    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop("img/Transformer/encoderblock/LayerNorm_0/scale")
    encoderblock_layernorm0_bias = state_dict.pop("img/Transformer/encoderblock/LayerNorm_0/bias")
    encoderblock_layernorm1_scale = state_dict.pop("img/Transformer/encoderblock/LayerNorm_1/scale")
    encoderblock_layernorm1_bias = state_dict.pop("img/Transformer/encoderblock/LayerNorm_1/bias")

    encoderblock_mlp_dense0_kernel= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel")
    encoderblock_mlp_dense0_bias= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias")
    encoderblock_mlp_dense1_kernel= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")
    encoderblock_mlp_dense1_bias= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias")

    encoderblock_attention_0_key_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel")
    encoderblock_attention_0_key_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias")
    encoderblock_attention_0_value_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel")
    encoderblock_attention_0_value_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias")
    encoderblock_attention_0_query_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel")
    encoderblock_attention_0_query_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias")
    encoderblock_attention_0_out_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel")
    encoderblock_attention_0_out_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias")

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

    embedding_vector = state_dict.pop("llm/embedder/input_embedding")
    state_dict["language_model.model.embed_tokens.weight"] = embedding_vector

    # pop the einsum attention + mlp representations. There are 18 layers in gemma-2b.

    llm_attention_attn_vec_einsum = state_dict.pop("llm/layers/attn/attn_vec_einsum/w")
    llm_attention_kv_einsum = state_dict.pop("llm/layers/attn/kv_einsum/w")
    llm_attention_q_einsum = state_dict.pop("llm/layers/attn/q_einsum/w")

    llm_mlp_gating_einsum = state_dict.pop("llm/layers/mlp/gating_einsum")
    llm_mlp_linear = state_dict.pop("llm/layers/mlp/linear")
    # TODO verify correctness of layer norm loading

    llm_input_layernorm = state_dict.pop("llm/layers/pre_attention_norm/scale")
    llm_post_attention_layernorm = state_dict.pop("llm/layers/pre_ffw_norm/scale")

    for i in range(config.text_config.num_hidden_layers):
        """
        GemmaConfig {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "head_dim": 256,
        "hidden_act": "gelu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 16384,
        "max_position_embeddings": 8192,
        "model_type": "gemma",
        "num_attention_heads": 8,
        "num_hidden_layers": 18,
        "num_key_value_heads": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "transformers_version": "4.39.0.dev0",
        "use_cache": true,
        "vocab_size": 257152
        }
        """
        # flash_attention_2
        # llm_attention_q_einsum[i].shape = (8, 2048, 256)
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)

        state_dict[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        # llm_attention_kv_einsum[i, 0, 0].shape = (2048, 256)
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"language_model.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        # llm_attention_kv_einsum[i, 1, 0].shape = (2048, 256)
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"language_model.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        # output projection.

        # llm_attention_attn_vec_einsum[i].shape = (8, 256, 2048) 
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].transpose(2, 0, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)

        state_dict[f"language_model.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        # mlp layers
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"language_model.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"language_model.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"language_model.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"language_model.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"language_model.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    state_dict["language_model.model.norm.weight"] = state_dict.pop("llm/final_norm/scale")
    state_dict["language_model.lm_head.weight"] = embedding_vector # weights are tied.

    # fmt: on
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)
    return state_dict


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    for k, v in params.items():
        print(k)
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def verify_logits(model):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.padding_side = 'right'
    # First load intermediates given, and test prompt

    # all intermediates activations
    intermediates_path = "/home/ubuntu/gvhf/hf_test_ckpt.cow_beach_1.bv.intermediates.npz"
    # intermediates_path = "/home/pablo/Downloads/gvhf/hf_test_ckpt.cow_beach_1.bv.intermediates.npz"
    cow_on_beach_path = "/home/ubuntu/gvhf/cow_beach_1.png"
    # cow_on_beach_path = "/home/pablo/Downloads/gvhf/cow_beach_1.png"
    intermediates = np.load(intermediates_path)

    # These steps mimic what should be taken in the processor - for an image we don't resize (given image is already 224px)
    # TODO replace by a proper Processor() call when tests pass

    img = np.array(Image.open(cow_on_beach_path))
    img = img
    img = np.expand_dims(img, 0)
    images = img.astype(np.float32)

    input_data_format = infer_channel_dimension_format(images[0])
    images = [rescale(image=image, scale=1 / 255, input_data_format=input_data_format) for image in images]

    images = [
        normalize(
            image=image, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD, input_data_format=input_data_format
        )
        for image in images
    ]
    images = [
        to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
        for image in images
    ]

    image_tensor = BatchFeature(data={"pixel_values": images}, tensor_type="pt").to(device)

    with torch.inference_mode():
        vision_outputs = model.vision_model(
            pixel_values=image_tensor["pixel_values"], output_hidden_states=True
        ).last_hidden_state
        projector_output = model.multi_modal_projector(vision_outputs)

    if not np.allclose(projector_output.cpu().numpy()[0], intermediates["img/zimg"][0], rtol=1e-3, atol=5e-3):
        raise ValueError("image activations do not match.")
    else:
        print("Vision activations match.")


    # text logits
    prompt = "answer en Where is the cow standing?\n"
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt", max_length=16, padding='max_length').to(device)
    with torch.inference_mode():
        text_token_embeddings = model.language_model.model.embed_tokens(prompt_input_ids)
        max_length = 16
        unpadded_length = len(tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt", max_length=max_length, padding='do_not_pad')[0])
        attention_mask = torch.cat((torch.ones((1, unpadded_length)), torch.zeros((1, max_length - unpadded_length))), dim=1)
        # z_txt = model.language_model(input_embeds=text_token_embeddings, attention_mask=attention_mask)

    concat_embeddings = torch.cat((projector_output, np.sqrt(2048) * text_token_embeddings), dim=1)
    # This matches exactly the gemma embeddings
    # Verify generation
    with torch.inference_mode():
        max_length = 16
        unpadded_length = len(tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt", max_length=max_length, padding='do_not_pad')[0])

        attention_mask = torch.cat((torch.ones((1, projector_output.shape[1] + unpadded_length)), torch.zeros((1, max_length - unpadded_length))), dim=1)
        attention_mask= attention_mask.bool()
        # check raw outputs before generate
        outputs = model.language_model(attention_mask=attention_mask, inputs_embeds=concat_embeddings, output_hidden_states=True)
        # chck geenrate
        generation = tokenizer.decode(model.language_model.generate(inputs_embeds=concat_embeddings, max_new_tokens=10, attention_mask=attention_mask)[0])
        print(generation)
        if generation != "beach":
            raise ValueError("Generation does not match.")
        else:
            print("Generation matches. You're almost done!")

    #with torch.inference_mode():
    #    outputs_activations_text = model.language_model(prompt_input_ids).logits
    """
    if not np.allclose(outputs_activations_text.cpu().numpy()[0], intermediates['text_logits'][0], rtol=1e-3, atol=5e-3):
        print(outputs_activations_text.cpu().numpy()[0, 0, 0:10], intermediates['text_logits'][0, 0, 0:10])
        raise ValueError("Text activations do not match.")
    else:
        print("Text activations match.")
    """

    # Verify pre logits
    with torch.inference_mode():
        outputs = model.language_model(attention_mask=attention_mask, inputs_embeds=concat_embeddings, output_hidden_states=True)

        for h in outputs.hidden_states:
            print((h.cpu().numpy()-intermediates['llm/pre_logits']).mean())
            # last entry gives a close to 0.000 mean, which is encouraging
        '''
        pre_logits = outputs.hidden_states[-1]

        if not np.allclose(intermediates['llm/pre_logits'], pre_logits, atol=1e-3, rtol=1e-3):
            raise ValueError("concatenated pre logits do not match.")
        else:
            print("Concatenated pre logits match.")
        '''


@torch.no_grad()
def convert_palma_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """
    # define default SigLIP configuration
    config = get_palma_config()
    # get checkpoint (move to args)

    do_convert_weights = False

    if do_convert_weights:
        checkpoint_path = "/home/ubuntu/gvhf/hf_test_ckpt.bv.params.npz" 
        # load original state dict
        print("Loading original jax state dict...")
        data = load(checkpoint_path)
        print("State dict loaded. Flattening...")
        state_dict = flatten_nested_dict(data)
        del data
        print("Flattened state dict. Starting slice and replace...")
        print(state_dict.keys())

        state_dict_transformers = slice_state_dict(state_dict, config)
        del state_dict
        # load HuggingFace model
        print("Instantiating model...")
        model = PalmaForConditionalGeneration(config).to(device).eval()
        print("Model setup done. Loading new weights...")
        # load jax-imported state dictionary to model
        model.load_state_dict(state_dict_transformers)
        del state_dict_transformers
        model.save_pretrained(pytorch_dump_folder_path, max_shard_size="2GB", safe_serialization=True)

    else:
        print("Loading locally transformed weights...")
        model = PalmaForConditionalGeneration.from_pretrained(pytorch_dump_folder_path).eval()
    print("New weights loaded. Verifying logits...")

    do_verify_logits = True

    if do_verify_logits:
        print("Verifying logits...")
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
        "--pytorch_dump_folder_path", default="/home/ubuntu/palma_hf", type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_palma_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)
