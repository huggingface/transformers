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
"""Convert PaliGemma checkpoints from the original repository.
"""


import argparse
import collections

import torch
from numpy import load
from PIL import Image

from transformers import (
    AutoTokenizer,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    GemmaTokenizer,
    GemmaTokenizerFast,
    SiglipImageProcessor,
)
from transformers.utils import logging
from transformers.tokenization_utils_base import AddedToken


device = "cuda" #"cpu"

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# TODO add sequence length variations here

PALIGEMMA_VARIANTS = ["2b-test", "2b-224px", "2b-448px", "2b-896px"]

def get_paligemma_config(variant: str):
    config = {
        'image_token_index': None,
        'pad_token_id': 0,
        'bos_token_id': 2,
        'eos_token_id': 1,
        }

    image_sizes = {
        "2b-test": 224,
        "2b-224px": 224,
        "2b-448px": 448,
        "2b-896px": 896
    }

    if variant in PALIGEMMA_VARIANTS:
        image_size = image_sizes[variant]
        patch_size = 14
        num_image_tokens = (image_size ** 2) // (patch_size ** 2)
        
        config['image_token_index'] = 257152 if variant != "2b-test" else 256000
        text_config = {
            'vocab_size': 257152,
            'num_hidden_layers': 18,
            'num_key_value_heads': 1,
            'head_dim': 256,
            'torch_dtype': 'float32',
            'hidden_size': 2048,
            'hidden_act': 'gelu_pytorch_tanh',
            'num_attention_heads': 8,
            'intermediate_size': 16384,
            'is_encoder_decoder': False
        }
        vision_config = {
            'image_size': image_size,
            'patch_size': patch_size,
            'num_image_tokens': num_image_tokens,
            'hidden_size': 1152,
            'intermediate_size': 4304,
            'num_hidden_layers': 27,
            'num_attention_heads': 16,
            'projector_hidden_act': 'gelu_fast',
            'vision_use_head': False
        }
        final_config = PaliGemmaConfig(text_config=text_config, vision_config=vision_config, **config)
    else:
        raise ValueError(f"Identifier {variant} not supported. Available: {PALIGEMMA_VARIANTS}")
    return final_config


def slice_state_dict(state_dict, config):
    # fmt: off
    # patch embeddings
    state_dict["vision_tower.vision_model.embeddings.patch_embedding.weight"] = state_dict.pop("img/embedding/kernel").transpose(
        3, 2, 0, 1
    )
    state_dict["vision_tower.vision_model.embeddings.patch_embedding.bias"] = state_dict.pop("img/embedding/bias")
    # positional embeddings
    state_dict["vision_tower.vision_model.embeddings.position_embedding.weight"] = state_dict.pop("img/pos_embedding").reshape(
        -1, config.vision_config.hidden_size
    )

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
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    state_dict["vision_tower.vision_model.post_layernorm.weight"] = state_dict.pop("img/Transformer/encoder_norm/scale").transpose()
    state_dict["vision_tower.vision_model.post_layernorm.bias"] = state_dict.pop("img/Transformer/encoder_norm/bias")

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
        k = k.removeprefix("params/")
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def verify_logits(model, processor):
    """
    This verification is only valid for the variant "2b-test". There is currently no other verification available.
    """

    cow_on_beach_path = "/home/pablo/cow_beach_1.png"
    list_images = [Image.open(cow_on_beach_path), Image.open(cow_on_beach_path)]
    
    # Test image captioning generation without padding
    image_captioning_inputs_unpad = processor(text="", images=list_images[0], max_length=16, padding="longest", return_tensors="pt").to(device)
    captioning_generation = model.generate(**image_captioning_inputs_unpad, max_new_tokens=10)
    captioning_output = processor.batch_decode(captioning_generation, skip_special_tokens=True)
    if captioning_output[0] != "\ncow standing on the beach":
        raise ValueError(fr"Image captioning without padding should match, got {captioning_output[0]}.")
    else:
        print("Image captioning works without padding.")


    # Test image captioning generation with padding
    image_captioning_inputs = processor(text="", images=list_images[0], max_length=16, padding="max_length", return_tensors="pt").to(device)
    captioning_generation = model.generate(**image_captioning_inputs, max_new_tokens=10)
    captioning_output = processor.batch_decode(captioning_generation, skip_special_tokens=True)
    if captioning_output[0] != "\ncow standing on the beach":
        raise ValueError(fr"Image captioning with should match, got {captioning_output[0]}.")
    else:
        print("Image captioning works with padding.")



    prompt = ["answer en Where is the cow standing?", ""]

    model_inputs = processor(text=prompt, images=list_images, max_length=16, padding="longest", return_tensors="pt").to(device)
    with torch.inference_mode():        
        raw_generation = model.generate(**model_inputs, max_new_tokens=10)
        generated_output = processor.batch_decode(raw_generation, skip_special_tokens=True)

        if generated_output[0] != "answer en Where is the cow standing?\nbeach":
            raise ValueError("Generation does not match.")
        elif generated_output[1] != "\ncow standing on the beach":
            raise ValueError("Image captioning does not match.")
        else:
            print("Generation matches. You're almost done!")



 

    
    print("Verifying that batched generation and single generation works.")
    big_batch_inputs = ["answer en What is that, isn't int a very strange happenstance? Most confusinglicious?", "", "answer en Where is the cow standing?"]

    list_images = [Image.open(cow_on_beach_path), Image.open(cow_on_beach_path), Image.open(cow_on_beach_path)]
    batch_model_inputs = processor(text=big_batch_inputs, images=list_images, padding="longest", return_tensors='pt').to(device)
    single_inputs = [processor(text=t, images=i, padding="do_not_pad", return_tensors="pt").to(device) for t, i in zip(big_batch_inputs, list_images)]
    with torch.inference_mode():
        batched_generation = model.generate(**batch_model_inputs, max_new_tokens=25)
        batched_output = processor.batch_decode(batched_generation, skip_special_tokens=True)
        single_generations = [model.generate(**single_input, max_new_tokens=25) for single_input in single_inputs]
        single_outputs = [processor.batch_decode(gen, skip_special_tokens=True) for gen in single_generations]
        for single, batched in zip(single_outputs, batched_output):
            if single[0] != batched:
                raise ValueError(f"Single-batch generation does not match batched. {single} != {batched}")
    print("All checks completed. Conversion is proper.")


@torch.no_grad()
def convert_paligemma_checkpoint(
    checkpoint_path, tokenizer_model_file, pytorch_dump_folder_path, variant: str, do_verify_logits=True, do_convert_weights=False
):
    """
    Read checkpoints from flax npz files, rename/reshape, send result to state dict and verify logits if needed.
    """
    config = get_paligemma_config(variant)
    if do_convert_weights:
        if variant == "2b-test":
            # for the test model, the vocabulary was smaller
            tokenizer_id = "google/gemma-2b"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        else:
            tokenizer_class = GemmaTokenizer if GemmaTokenizerFast is None else GemmaTokenizerFast
            tokenizer = tokenizer_class(tokenizer_model_file)
        image_token = AddedToken("<image>", normalized=False, special=True)
        tokens_to_add = {
            "additional_special_tokens": [image_token]
        }
        tokenizer.add_special_tokens(tokens_to_add)
        
        # tokenizer.padding_side = 'right' # commented out, activate for testing purposes. 

        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        image_processor.size = {"width": config.vision_config.image_size, "height": config.vision_config.image_size}
        image_processor.image_seq_length = config.vision_config.num_image_tokens

        processor = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        data = load(checkpoint_path)
        state_dict = flatten_nested_dict(data)
        del data
        state_dict_transformers = slice_state_dict(state_dict, config)
        del state_dict

        model = PaliGemmaForConditionalGeneration(config).to(device).eval()
        model.load_state_dict(state_dict_transformers)
        del state_dict_transformers

    else:
        processor = PaliGemmaProcessor.from_pretrained(pytorch_dump_folder_path)
        model = PaliGemmaForConditionalGeneration.from_pretrained(pytorch_dump_folder_path, attn_implementation="sdpa").to(device).eval()
    model.config.text_config._attn_implementation = 'sdpa'
    if do_verify_logits:
        print("Verifying logits...")
        verify_logits(model, processor)

    # model expansion to get random embeds of image tokens
    pad_shape = 64 # for performance reasons
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[257152:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[257152:].shape[0]))),
        dim=0,
    )
    model.language_model.lm_head.weight.data[257152:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[257152:].shape[0]))),
        dim=0,
    )

    #model.save_pretrained(pytorch_dump_folder_path, max_shard_size="2GB", safe_serialization=True)
    #processor.save_pretrained(pytorch_dump_folder_path)



# 




"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default="/home/pablo/gv-hf/hf_test_ckpt.bv.params.npz",
        type=str,
        help="Path to the .npz checkpoint",
    )

    parser.add_argument(
        "--tokenizer_model_file",
        default="/home/pablo/paligemma/paligemma_tokenizer.model",
        type=str,
        help="Path to the sentencepiece model file",
    )

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/pablo/paligemma/paligemma-test-hf ",
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    parser.add_argument(
        "--variant",
        default="2b-test",
        choices=PALIGEMMA_VARIANTS,
        type=str,
        help="String identifier of the paligemma variant to convert."
    )

    parser.add_argument(
        "--do_verify_logits",
        action="store_false",
        help="Whether or not to run checks against original implementation.",
    )
    parser.add_argument(
        "--do_convert_weights", action="store_true", help="Whether or not to reload and convert the weights."
    )

    args = parser.parse_args()
    convert_paligemma_checkpoint(
        checkpoint_path=args.checkpoint_path,
        tokenizer_model_file=args.tokenizer_model_file,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        variant=args.variant,
        do_verify_logits=args.do_verify_logits,
        do_convert_weights=args.do_convert_weights,
    )

"""

"""
python src/transformers/models/paligemma/convert_paligemma_weights_to_hf.py 
--checkpoint_path="/raid/pablo/palma_files/test_model/hf_test_ckpt.bv.params.npz"
 --variant="2b-test" 
 --tokenizer_model_file="/raid/pablo/paligemma/paligemma_tokenizer.model" 
 --do_convert_weights --do_verify_logits 
 --pytorch_dump_folder_path="/raid/pablo/converted_224_test"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default="/raid/pablo/palma_files/test_model/hf_test_ckpt.bv.params.npz",
        type=str,
        help="Path to the .npz checkpoint",
    )

    parser.add_argument(
        "--tokenizer_model_file",
        default="/raid/pablo/paligemma/paligemma_tokenizer.model",
        type=str,
        help="Path to the sentencepiece model file",
    )

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/raid/pablo/converted_224_test/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )

    parser.add_argument(
        "--variant",
        default="2b-test",
        choices=PALIGEMMA_VARIANTS,
        type=str,
        help="String identifier of the paligemma variant to convert."
    )

    parser.add_argument(
        "--do_verify_logits",
        action="store_false",
        help="Whether or not to run checks against original implementation.",
    )
    parser.add_argument(
        "--do_convert_weights", action="store_false", help="Whether or not to reload and convert the weights."
    )

    args = parser.parse_args()
    convert_paligemma_checkpoint(
        checkpoint_path=args.checkpoint_path,
        tokenizer_model_file=args.tokenizer_model_file,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        variant=args.variant,
        do_verify_logits=args.do_verify_logits,
        do_convert_weights=args.do_convert_weights,
    )
