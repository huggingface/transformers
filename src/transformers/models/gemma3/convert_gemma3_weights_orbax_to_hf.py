# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

r"""Utility to convert Gemma models from Orbax to HF Transformers checkpoint.

python -m transformers.models.gemma3.convert_gemma3_weights_orbax_to_hf \
    --variant='gemma3_4b' \
    --tokenizer_path="$HOME/gemma3/tokenizer/gemma3_cleaned_262144_v2.spiece.model" \
    --checkpoint_path="$HOME/gemma3/gemma3_4b_pt_orbax/" \
    --output_path="$HOME/gemma3/gemma3_4b_pt_safetensors/"
"""

from collections.abc import Iterator, Sequence
from typing import Any

import accelerate
import numpy as np
import torch
import tree
from absl import app, flags, logging
from orbax import checkpoint as obc

from transformers import (
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3ImageProcessor,
    Gemma3Processor,
    Gemma3TextConfig,
    GemmaTokenizerFast,
    GenerationConfig,
    SiglipVisionConfig,
)
from transformers.image_utils import PILImageResampling


# ==== Internal Constants and Classes ====


_CHAT_TEMPLATE = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""

_DTYPES = {"float32", "bfloat16", "float16"}

_SIGLIP_BASE = "SigLiPFromPatches_0/siglip_encoder"
_SIGLIP_EMBEDDING = "SigLiPFromPatches_0/siglip_encoder/embedding"
_SIGLIP_TRANSFORMER_ENCODER_BLOCK = "SigLiPFromPatches_0/siglip_encoder/Transformer/encoderblock_"
_SIGLIP_TRANSFORMER_ENCODER_BLOCK_LEN = len(_SIGLIP_TRANSFORMER_ENCODER_BLOCK)
_SIGLIP_TRANSFORMER_ENCODER_NORM = "SigLiPFromPatches_0/siglip_encoder/Transformer/encoder_norm"

_TRANSFORMER_DECODER_BLOCK = "transformer/layer_"
_TRANSFORMER_DECODER_BLOCK_LEN = len(_TRANSFORMER_DECODER_BLOCK)
_TRANSFORMER_EMBEDDER = "transformer/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"
_TRANSFORMER_POST_TRAINING_PREFIX = "rlx_networks/policy_network/"
_TRANSFORMER_POST_TRAINING_PREFIX_LEN = len(_TRANSFORMER_POST_TRAINING_PREFIX)

_VISION_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_channels": 3,
    "image_size": 896,
    "patch_size": 14,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "vision_use_head": False,
}

_VARIANT_GEMMA_3_1B = "gemma3_1b"
_VARIANT_GEMMA_3_4B = "gemma3_4b"
_VARIANT_GEMMA_3_12B = "gemma3_12b"
_VARIANT_GEMMA_3_27B = "gemma3_27b"
_VARIANTS = {
    _VARIANT_GEMMA_3_1B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_144,
            hidden_size=1152,
            intermediate_size=6 * 1152,
            num_attention_heads=4,
            num_hidden_layers=26,
            num_key_value_heads=1,
            head_dim=256,
            sliding_window=512,
            rope_theta=1_000_000,  # used for global RoPE only
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=256,
            max_position_embeddings=32_768,
        ),
        vision_config=None,
    ),
    _VARIANT_GEMMA_3_4B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_208,
            hidden_size=2560,
            intermediate_size=2560 * 8 // 2,
            num_attention_heads=8,
            head_dim=256,
            num_hidden_layers=34,
            num_key_value_heads=4,
            sliding_window=1024,
            rope_scaling={"rope_type": "linear", "factor": 8.0},  # used for global RoPE only
            rope_theta=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=256,
        ),
        vision_config=_VISION_CONFIG,
    ),
    _VARIANT_GEMMA_3_12B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_208,
            hidden_size=30 * 128,
            intermediate_size=30 * 128 * 8 // 2,
            num_attention_heads=16,
            head_dim=256,
            num_hidden_layers=48,
            num_key_value_heads=8,
            sliding_window=1024,
            rope_scaling={"rope_type": "linear", "factor": 8.0},  # used for global RoPE only
            rope_theta=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=256,
        ),
        vision_config=_VISION_CONFIG,
    ),
    _VARIANT_GEMMA_3_27B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_208,
            hidden_size=42 * 128,
            intermediate_size=42 * 128 * 8 // 2,
            num_attention_heads=32,
            num_hidden_layers=62,
            num_key_value_heads=16,
            head_dim=128,
            sliding_window=1024,
            rope_scaling={"rope_type": "linear", "factor": 8.0},  # used for global RoPE only
            rope_theta=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=(42 * 128 // 32),  # 1 / sqrt(hidden_size // num_attention_heads)
        ),
        vision_config=_VISION_CONFIG,
    ),
}

# ==== Flags ====

_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="Path to the Orbax checkpoint.",
    required=True,
)

_INCLUDE_CHAT_TEMPLATE = flags.DEFINE_bool(
    name="include_chat_template", default=False, help="If true, will save the default chat template with the tokenizer"
)

_OUTPUT_PATH = flags.DEFINE_string(
    name="output_path",
    default=None,
    help="Path to store the HF checkpoint.",
    required=True,
)

_TRANSFORMER_DTYPE = flags.DEFINE_enum(
    name="text_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the model.",
    enum_values=_DTYPES,
)

_TOKENIZER_PATH = flags.DEFINE_string(
    name="tokenizer_path",
    default=None,
    help="Path to the SentencePiece model file.",
    required=True,
)

_VARIANT = flags.DEFINE_enum(
    name="variant",
    default=_VARIANT_GEMMA_3_4B,
    help="The model variant to convert.",
    enum_values=set(_VARIANTS.keys()),
)

_VERBOSE = flags.DEFINE_bool(
    name="verbose",
    default=False,
    help="If true, log the path, shape, and dtype of every converted layer.",
)

_VISION_DTYPE = flags.DEFINE_enum(
    name="vision_dtype",
    default="float32",
    help="The floating point precision (aka dtype) of the model.",
    enum_values=_DTYPES,
)


def convert_siglip_weight(
    config: SiglipVisionConfig,
    paths: Sequence[str],
    weights: np.ndarray,
) -> tuple[str, np.ndarray]:
    path, prop = paths
    normalized_path: str = ""
    updated_weights: np.ndarray = None

    if path == _SIGLIP_BASE:
        normalized_path = "vision_tower.vision_model.embeddings.position_embedding.weight"
        updated_weights = weights.reshape(-1, config.hidden_size)
    elif path == _SIGLIP_EMBEDDING:
        if prop == "kernel":
            normalized_path = "vision_tower.vision_model.embeddings.patch_embedding.weight"
            updated_weights = weights.transpose(3, 2, 0, 1)
        elif prop == "bias":
            normalized_path = "vision_tower.vision_model.embeddings.patch_embedding.bias"
            updated_weights = weights
        else:
            raise ValueError(f"Unexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`.")
    elif path.startswith(_SIGLIP_TRANSFORMER_ENCODER_BLOCK):
        encoder_block_path = path[_SIGLIP_TRANSFORMER_ENCODER_BLOCK_LEN:]
        next_path_seperator_idx = encoder_block_path.find("/")
        layer_idx = encoder_block_path[:next_path_seperator_idx]
        encoder_block_path = encoder_block_path[next_path_seperator_idx:]
        normalized_path = f"vision_tower.vision_model.encoder.layers.{layer_idx}"

        if encoder_block_path.startswith("/LayerNorm"):
            normalized_path += ".layer_norm1" if path.endswith("_0") else ".layer_norm2"

            if prop == "scale":
                normalized_path += ".weight"
                updated_weights = weights.transpose()
            elif prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights
            else:
                raise ValueError(f"Unexpected member, `{prop}`, for path `{path}`. Should be `bias` or `scale`.")
        elif encoder_block_path.startswith("/MlpBlock_0"):
            normalized_path += ".mlp.fc1" if "/Dense_0" in encoder_block_path else ".mlp.fc2"

            if prop == "kernel":
                normalized_path += ".weight"
                updated_weights = weights.transpose()
            elif prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights
            else:
                raise ValueError(f"Unexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`.")
        elif encoder_block_path.startswith("/MultiHeadDotProductAttention_0"):
            if encoder_block_path.endswith("/key"):
                normalized_path += ".self_attn.k_proj"
            elif encoder_block_path.endswith("/out"):
                normalized_path += ".self_attn.out_proj"
            elif encoder_block_path.endswith("/query"):
                normalized_path += ".self_attn.q_proj"
            elif encoder_block_path.endswith("/value"):
                normalized_path += ".self_attn.v_proj"
            else:
                raise ValueError(f"Unexpected path `{path}` in SigLIP Transformer MultiHeadDotProductAttention_0.")

            if prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights.reshape(-1, config.hidden_size).reshape(-1)
            elif prop == "kernel":
                normalized_path += ".weight"
                updated_weights = weights.reshape(-1, config.hidden_size).transpose()
            else:
                raise ValueError(f"Unexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`.")
        else:
            raise ValueError(f"Unexpected path `{path}` in SigLIP Transformer Encoder Block.")
    elif path == _SIGLIP_TRANSFORMER_ENCODER_NORM:
        if prop == "scale":
            normalized_path = "vision_tower.vision_model.post_layernorm.weight"
            updated_weights = weights.transpose()
        elif prop == "bias":
            normalized_path = "vision_tower.vision_model.post_layernorm.bias"
            updated_weights = weights
        else:
            raise ValueError(f"Unexpected member, `{prop}`, for path `{path}`. Should be `bias` or `scale`.")
    else:
        raise ValueError(f"Unexpected path `{path}`.")

    return normalized_path, updated_weights


def convert_transformer_weights(
    config: Gemma3TextConfig,
    paths: Sequence[str],
    weights: np.ndarray,
) -> Iterator[tuple[str, np.ndarray]]:
    path, prop = paths

    if path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
        path = path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]

    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    attn_head_dim = config.num_attention_heads * config.head_dim
    kv_head_dim = config.num_key_value_heads * config.head_dim

    if path == _TRANSFORMER_EMBEDDER:
        if prop == "input_embedding":
            # Tied to language_model.lm_head.weight, assigned at the end.
            converted_paths = ["language_model.model.embed_tokens.weight"]

            if _VARIANT.value != _VARIANT_GEMMA_3_1B:
                # Gemma3 model doesn't have image soft token in input and output embeddings, resize to avoid bugs we had with Mllama
                pre_expansion_embeddings = weights
                mu = np.mean(pre_expansion_embeddings, axis=0)
                sigma = np.cov(pre_expansion_embeddings, rowvar=False, bias=True)
                new_embeddings = np.random.multivariate_normal(mu, 1e-5 * sigma, size=64)
                weights = np.vstack([pre_expansion_embeddings, new_embeddings])

            converted_weights = [weights]
        elif _VARIANT.value == _VARIANT_GEMMA_3_1B or prop in ("mm_output_embedding", "mm_input_embedding_extra"):
            return zip([], [])
        else:
            raise ValueError(f"Unexpected member, {prop}, in Embedder.")
    elif path.startswith(f"{_TRANSFORMER_EMBEDDER}/mm"):
        if _VARIANT.value == _VARIANT_GEMMA_3_1B:
            return zip([], [])

        if path.endswith("/mm_input_projection"):
            converted_paths = ["multi_modal_projector.mm_input_projection_weight"]
            converted_weights = [weights]
        elif path.endswith("/mm_soft_embedding_norm"):
            converted_paths = ["multi_modal_projector.mm_soft_emb_norm.weight"]
            converted_weights = [weights]
        else:
            raise ValueError(f"Unexpected subpath, `{path}`, in Embedder.")
    elif path == _TRANSFORMER_FINAL_NORM:
        converted_paths = ["language_model.model.norm.weight"]
        converted_weights = [weights]
    elif path.startswith(_TRANSFORMER_DECODER_BLOCK):
        decoder_block_path = path[_TRANSFORMER_DECODER_BLOCK_LEN:]
        next_path_seperator_idx = decoder_block_path.find("/")
        layer_idx = decoder_block_path[:next_path_seperator_idx]
        decoder_block_path = decoder_block_path[next_path_seperator_idx:]

        base_path = f"language_model.model.layers.{layer_idx}"

        if path.endswith("attn/attn_vec_einsum"):
            converted_paths = [f"{base_path}.self_attn.o_proj.weight"]
            converted_weights = [weights.transpose(2, 0, 1).reshape(config.hidden_size, attn_head_dim)]
        elif path.endswith("attn/_key_norm"):
            converted_paths = [f"{base_path}.self_attn.k_norm.weight"]
            converted_weights = [weights]
        elif path.endswith("attn/kv_einsum"):
            converted_paths = [
                f"{base_path}.self_attn.k_proj.weight",
                f"{base_path}.self_attn.v_proj.weight",
            ]
            k_proj_weights, v_proj_weights = weights
            converted_weights = [
                k_proj_weights.transpose(0, 2, 1).reshape(kv_head_dim, config.hidden_size),
                v_proj_weights.transpose(0, 2, 1).reshape(kv_head_dim, config.hidden_size),
            ]
        elif path.endswith("attn/q_einsum"):
            converted_paths = [f"{base_path}.self_attn.q_proj.weight"]
            converted_weights = [weights.transpose(0, 2, 1).reshape(attn_head_dim, config.hidden_size)]
        elif path.endswith("attn/_query_norm"):
            converted_paths = [f"{base_path}.self_attn.q_norm.weight"]
            converted_weights = [weights]
        elif path.endswith("mlp/gating_einsum"):
            converted_paths = [
                f"{base_path}.mlp.gate_proj.weight",
                f"{base_path}.mlp.up_proj.weight",
            ]
            gate_proj_weight, up_proj_weight = weights
            converted_weights = [gate_proj_weight, up_proj_weight]
        elif path.endswith("mlp/linear"):
            converted_paths = [f"{base_path}.mlp.down_proj.weight"]
            converted_weights = [weights.transpose()]
        elif path.endswith("post_attention_norm"):
            converted_paths = [f"{base_path}.post_attention_layernorm.weight"]
            converted_weights = [weights]
        elif path.endswith("post_ffw_norm"):
            converted_paths = [f"{base_path}.post_feedforward_layernorm.weight"]
            converted_weights = [weights]
        elif path.endswith("pre_attention_norm"):
            converted_paths = [f"{base_path}.input_layernorm.weight"]
            converted_weights = [weights]
        elif path.endswith("pre_ffw_norm"):
            converted_paths = [f"{base_path}.pre_feedforward_layernorm.weight"]
            converted_weights = [weights]
        else:
            raise ValueError(f"Unexpected path `{path}` in Decoder Block.")
    else:
        raise ValueError(f"Unexpected path `{path}`.")

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert(checkpoint_path: str, config: Gemma3Config) -> dict[str, torch.Tensor]:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    checkpointer = obc.PyTreeCheckpointer()
    ckpt = checkpointer.restore(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    def update_tree(path: str, weights: np.ndarray, target_dtype: torch.dtype) -> None:
        hf_tree[path] = torch.from_numpy(weights.astype("float32")).type(target_dtype)
        if _VERBOSE.value:
            logging.info(
                "%s converted shape=%s with dtype=%s",
                path,
                weights.shape,
                target_dtype,
            )

    for paths, value in tree.flatten_with_path(ckpt):
        if paths[0].startswith("SigLiPFromPatches_"):
            if config.vision_config is None:
                continue

            path, weights = convert_siglip_weight(config=config.vision_config, paths=paths, weights=value)
            update_tree(path, weights, config.vision_config.torch_dtype)
        else:
            for path, weights in convert_transformer_weights(config=config.text_config, paths=paths, weights=value):
                if config.vision_config is None:
                    path = path[len("language_model.") :]

                update_tree(path, weights, config.text_config.torch_dtype)

    if config.vision_config is None:
        hf_tree["lm_head.weight"] = hf_tree["model.embed_tokens.weight"]
    else:
        hf_tree["language_model.lm_head.weight"] = hf_tree["language_model.model.embed_tokens.weight"]

    return hf_tree


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value
    variant = _VARIANT.value

    config = _VARIANTS[variant]
    config.text_config.torch_dtype = getattr(torch, _TRANSFORMER_DTYPE.value)

    if variant == _VARIANT_GEMMA_3_1B:
        config.vision_config = None
    else:
        config.vision_config.torch_dtype = getattr(torch, _VISION_DTYPE.value)

    if _INCLUDE_CHAT_TEMPLATE.value:
        # Chat template is included for instruction tuned models, which treat
        # both "<eos>" and "<end_of_turn>" as generation stoppers.
        config.eos_token_id = [1, 106]

    logging.info(
        "Converting Gemma 3 (%s) @ %s (language) and %s (vision)",
        variant,
        _TRANSFORMER_DTYPE.value,
        _VISION_DTYPE.value,
    )
    state_tree = convert(_CHECKPOINT_PATH.value, config)
    logging.info("Converted Gemma 3 (%s) state tree from Orbax to Hugging Face.", variant)

    with accelerate.init_empty_weights():
        if variant == _VARIANT_GEMMA_3_1B:
            model = Gemma3ForCausalLM(config=config.text_config)
        else:
            model = Gemma3ForConditionalGeneration(config)

    model.load_state_dict(state_tree, assign=True, strict=True)
    logging.info(
        "Loaded Gemma 3 (%s) in Hugging Face Transformers as a %s instance.",
        variant,
        type(model).__name__,
    )
    model.save_pretrained(output_path, safe_serialization=True)
    logging.info(
        "Saved Gemma 3 (%s) to SafeTensors in %s using %s",
        variant,
        output_path,
        type(model).__name__,
    )
    del model
    del state_tree

    tokenizer = GemmaTokenizerFast(
        _TOKENIZER_PATH.value,
        add_bos_token=True,
        extra_special_tokens={
            "image_token": "<image_soft_token>",  # Should be ID=262_144
            "boi_token": "<start_of_image>",  # Should be ID=255_999
            "eoi_token": "<end_of_image>",  # Should be ID=256_000
        },
        chat_template=_CHAT_TEMPLATE if _INCLUDE_CHAT_TEMPLATE.value else None,
    )
    tokenizer.save_pretrained(output_path)
    logging.info("Saved GemmaTokenizer for %s to %s", variant, output_path)

    if variant != _VARIANT_GEMMA_3_1B:
        image_processor = Gemma3ImageProcessor(
            image_seq_length=256,
            image_mean=(0.5,) * 3,
            image_std=(0.5,) * 3,
            size={"height": 896, "width": 896},
            resample=PILImageResampling.BILINEAR,
        )
        processor = Gemma3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )
        processor.save_pretrained(output_path)
        logging.info("Saved Gemma3Processor for %s to %s", variant, output_path)
        del processor

    del tokenizer

    generation_config = GenerationConfig(
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        cache_implementation="hybrid",
        temperature=1.0,
        do_sample=True,
        top_k=64,
        top_p=0.95,
    )
    generation_config.save_pretrained(output_path)


if __name__ == "__main__":
    app.run(main)
