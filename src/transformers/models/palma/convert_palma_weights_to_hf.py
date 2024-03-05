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
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image

from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer, PalmaForConditionalGeneration, GemmaConfig, GemmaForCausalLM, PalmaConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_palma_config(model_name):
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
    state_dict["vision_model.embeddings.patch_embedding.weight"] = state_dict.pop("img/embedding/kernel")
    state_dict["vision_model.embeddings.patch_embedding.bias"] = state_dict.pop("img/embedding/bias")
    # positional embeddings
    state_dict["vision_model.embeddings.position_embedding.weight"] = state_dict.pop("img/pos_embedding")


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
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i]
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i]
        state_dict[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i]
        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i]
        state_dict[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]

        state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i]
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i]


    state_dict["vision_model.post_layernorm.weight"] = state_dict.pop("img/Transformer/encoder_norm/scale")
    state_dict["vision_model.post_layernorm.bias"] = state_dict.pop("img/Transformer/encoder_norm/bias")

    # multimodal projector
    
    state_dict['multi_modal_projector.linear.weight'] = state_dict.pop("img/head/kernel")
    state_dict['multi_modal_projector.linear.bias'] = state_dict.pop("img/head/bias")

    # text decoder (gemma)

    state_dict["language_model.model.embed_tokens"] = state_dict.pop("llm/embedder/input_embedding")
    state_dict["language_model.model.norm.weight"] = state_dict.pop("llm/final_norm/scale")
    state_dict["language_model.model.layers.input_layernorm"] = state_dict.pop("llm/layers/pre_attention_norm/scale")

    # pop the einsum attention representations. There are 18 layers in gemma-2b.

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear")
    

    """
    
    llm/layers/pre_attention_norm/scale 0.000   # OK
    (18, 2048)
    llm/layers/pre_ffw_norm/scale 0.000         # OK
    (18, 2048)
    """


    # new keys

    """
    (language_model): GemmaForCausalLM(
        (model): GemmaModel(
        (embed_tokens): Embedding(257152, 2048, padding_idx=0)
        (layers): ModuleList(
            (0): GemmaDecoderLayer(
            (self_attn): GemmaSdpaAttention(
                (q_proj): Linear(in_features=2048, out_features=4096, bias=False)
                (k_proj): Linear(in_features=2048, out_features=4096, bias=False)
                (v_proj): Linear(in_features=2048, out_features=4096, bias=False)
                (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
                (rotary_emb): GemmaRotaryEmbedding()
            )
            (mlp): GemmaMLP(
                (gate_proj): Linear(in_features=2048, out_features=16384, bias=False)
                (up_proj): Linear(in_features=2048, out_features=16384, bias=False)
                (down_proj): Linear(in_features=16384, out_features=2048, bias=False)
                (act_fn): GELUActivation()
            )
            (input_layernorm): GemmaRMSNorm()
            (post_attention_layernorm): GemmaRMSNorm()
            )
        )
        (norm): GemmaRMSNorm()
        )
        (lm_head): Linear(in_features=2048, out_features=257152, bias=False)
        
    """

    # fmt: on
    return state_dict


def rename_key(dct, old, new, config):
    val = dct.pop(old)

    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if "patch_embedding.weight" in new:
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if new.endswith("bias"):
        val = val.reshape(-1)

    dct[new] = torch.from_numpy(val)


def read_in_q_k_v_head(state_dict, config):
    # read in individual input projection layers
    key_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    key_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/bias").reshape(-1)
    value_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    value_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/bias").reshape(-1)
    query_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    query_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/bias").reshape(-1)

    # next, add them to the state dict as a single matrix + vector
    state_dict["vision_model.head.attention.in_proj_weight"] = torch.from_numpy(
        np.concatenate([query_proj_weight, key_proj_weight, value_proj_weight], axis=0)
    )
    state_dict["vision_model.head.attention.in_proj_bias"] = torch.from_numpy(
        np.concatenate([query_proj_bias, key_proj_bias, value_proj_bias], axis=0)
    )



"""
# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image
"""

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


@torch.no_grad()
def convert_siglip_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """

    # define default SigLIP configuration
    config = get_palma_config(model_name)

    # get checkpoint
    checkpoint = "/home/pablo/.cache/huggingface/hub/models--gv-hf--test/snapshots/58b24b23afbeb278bfa3aa3b9f0fc1e60b9cbcc5/hf_test_ckpt.bv.params.npz" # model_name_to_checkpoint[model_name]
    # load HuggingFace model
    model = PalmaForConditionalGeneration(config).eval()
    breakpoint()
    """
    # get vocab file
    if "i18n" in model_name:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/multilingual_vocab/sentencepiece.model"
    else:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/english_vocab/sentencepiece.model"
    """
    # load original state dict
    data = load(checkpoint)
    state_dict = flatten_nested_dict(data)

    state_dict_transformers = slice_state_dict(state_dict, config)
    breakpoint()

    # remove and rename some keys
    # rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest, config)

    # qkv matrices of attention pooling head need special treatment
    read_in_q_k_v_head(state_dict, config)


    model.load_state_dict(state_dict_transformers)
    
    """

    # create processor
    # important: make tokenizer not return attention_mask since original one doesn't require it
    image_size = config.vision_config.image_size
    size = {"height": image_size, "width": image_size}
    image_processor = SiglipImageProcessor(size=size)
    tokenizer = SiglipTokenizer(vocab_file=vocab_file, model_input_names=["input_ids"])
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # verify on dummy images and texts
    url_1 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
    image_1 = Image.open(requests.get(url_1, stream=True).raw).convert("RGB")
    url_2 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
    image_2 = Image.open(requests.get(url_2, stream=True).raw).convert("RGB")
    texts = ["an apple", "a picture of an apple"]

    inputs = processor(images=[image_1, image_2], text=texts, return_tensors="pt", padding="max_length")

    # verify input_ids against original ones
    if image_size == 224:
        filename = "siglip_pixel_values.pt"
    elif image_size == 256:
        filename = "siglip_pixel_values_256.pt"
    elif image_size == 384:
        filename = "siglip_pixel_values_384.pt"
    elif image_size == 512:
        filename = "siglip_pixel_values_512.pt"
    else:
        raise ValueError("Image size not supported")

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename=filename, repo_type="dataset")
    original_pixel_values = torch.load(filepath)
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="siglip_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath)

    if "i18n" not in model_name:
        assert inputs.input_ids.tolist() == original_input_ids.tolist()

    print("Mean of original pixel values:", original_pixel_values.mean())
    print("Mean of new pixel values:", inputs.pixel_values.mean())

    # note: we're testing with original pixel values here since we don't have exact pixel values
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, pixel_values=original_pixel_values)

    # with torch.no_grad():
    #     outputs = model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values)

    print(outputs.logits_per_image[:3, :3])

    probs = torch.sigmoid(outputs.logits_per_image)  # these are the probabilities
    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
    print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")

    if verify_logits:
        if model_name == "siglip-base-patch16-224":
            expected_slice = torch.tensor(
                [[-2.9621, -2.1672], [-0.2713, 0.2910]],
            )
        elif model_name == "siglip-base-patch16-256":
            expected_slice = torch.tensor(
                [[-3.1146, -1.9894], [-0.7312, 0.6387]],
            )
        elif model_name == "siglip-base-patch16-384":
            expected_slice = torch.tensor(
                [[-2.8098, -2.1891], [-0.4242, 0.4102]],
            )
        elif model_name == "siglip-base-patch16-512":
            expected_slice = torch.tensor(
                [[-2.7899, -2.2668], [-0.4295, -0.0735]],
            )
        elif model_name == "siglip-large-patch16-256":
            expected_slice = torch.tensor(
                [[-1.5827, -0.5801], [-0.9153, 0.1363]],
            )
        elif model_name == "siglip-large-patch16-384":
            expected_slice = torch.tensor(
                [[-2.1523, -0.2899], [-0.2959, 0.7884]],
            )
        elif model_name == "siglip-so400m-patch14-384":
            expected_slice = torch.tensor([[-1.2441, -0.6649], [-0.7060, 0.7374]])
        elif model_name == "siglip-base-patch16-256-i18n":
            expected_slice = torch.tensor(
                [[-0.9064, 0.1073], [-0.0299, 0.5304]],
            )

        assert torch.allclose(outputs.logits_per_image[:3, :3], expected_slice, atol=1e-4)
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub(f"nielsr/{model_name}")
        processor.push_to_hub(f"nielsr/{model_name}")
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="palma-base-patch14-224",
        type=str,
        help="Name of the model you'd like to convert.",
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
    convert_siglip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits)
