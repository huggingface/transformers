r"""Utility to convert Gemma models from Orbax to HF Transformers checkpoint.

python -m transformers.models.shieldgemma2.convert_shieldgemma2_weights_orbax_to_hf \
    --tokenizer_path="$HOME/gemma3/tokenizer/gemma3_cleaned_262144_v2.spiece.model" \
    --checkpoint_path_gemma="$HOME/gemma3/gemma3_4b_pt_orbax/" \
    --checkpoint_path_shieldgemma="$HOME/shieldgemma2/shieldgemma-2_4b_orbax/" \
    --output_path="$HOME/shieldgemma2/shieldgemma2_4b_pt_safetensors/" \
    --precision='bfloat16'
"""

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import accelerate
import numpy as np
import torch
import tree
from absl import app, flags, logging
from orbax import checkpoint as obc

from ...image_utils import PILImageResampling
from ..gemma import GemmaTokenizerFast
from ..gemma3 import Gemma3ImageProcessor, Gemma3TextConfig
from ..siglip import SiglipVisionConfig
from . import ShieldGemma2Config, ShieldGemma2ForImageClassification, ShieldGemma2Processor


# ==== Internal Constants and Classes ====

_CHAT_TEMPLATE = """{{ bos_token }}
{%- for message in messages -%}
    {%- if message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {{ '<start_of_turn>user\n' }}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ 'You are a policy expert trying to help determine whether a generated or real image is in' }}
                {{ ' violation of the safety policy below. Your decision must very clearly follow the specific'}}
                {{ ' language used in the safety policy and should not include any interpretation outside of the' }}
                {{ ' safety policy.\n\n' }}
                {{ '<BEGIN_SAFETY_POLICY>\n' }}
                {{ item['text'] + '\n' }}
                {{ '</END_SAFETY_POLICY>\n\n' }}
                {{ 'Does the image violate the above policy? Your Answer must start with \\'Yes\\' or \\'No\\'.' }}
                {{ '<end_of_turn>\n' }}
            {%- endif -%}
        {%- endfor -%}
        {{'<start_of_turn>model\n'}}
    {%- else -%}
        {{ raise_exception("Conversation messages must contain iterable content containing images and policy definitions in text.") }}
    {%- endif -%}
{%- endfor -%}
"""

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_SHIELDGEMMA2_POLICIES: Mapping[str, str] = {
    "dangerous": (
        "No Dangerous Content: The image shall not contain content that"
        " facilitates or encourages activities that could cause real-world harm"
        " (e.g., building firearms and explosive devices, promotion of"
        " terrorism, instructions for suicide)."
    ),
    "sexual": (
        "No Sexually Explicit content: The image shall not contain content that"
        " depicts explicit or graphic sexual acts (e.g., pornography, erotic"
        " nudity, depictions of rape or sexual assault)."
    ),
    "violence": (
        "No Violence/Gore content: The image shall not contain content that"
        " depicts shocking, sensational, or gratuitous violence (e.g.,"
        " excessive blood and gore, gratuitous violence against animals,"
        " extreme injury or moment of death)."
    ),
}

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

# ==== Flags ====

_GEMMA_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path_gemma",
    default=None,
    help="Path to the Orbax checkpoint containing the vision weights.",
    required=True,
)

_SHIELDGEMMA_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path_shieldgemma",
    default=None,
    help="Path to the Orbax checkpoint containing the language model weights.",
    required=True,
)

OUTPUT_PATH = flags.DEFINE_string(
    name="output_path",
    default=None,
    help="Path to store the HF checkpoint.",
    required=True,
)

PRECISION = flags.DEFINE_enum(
    name="precision",
    default=None,
    help="The floating point precision (aka dtype) of the model.",
    enum_values=set(_DTYPES.keys()),
    required=True,
)

TOKENIZER_PATH = flags.DEFINE_string(
    name="tokenizer_path",
    default=None,
    help="Path to the SentencePiece model file.",
    required=True,
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

    if "vision" in normalized_path:
        print(normalized_path)
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
            # Gemma3 model doesn't have image soft token in input and output embeddings, resize to avoid bugs we had with Mllama
            pre_expansion_embeddings = weights
            mu = np.mean(pre_expansion_embeddings, axis=0)
            sigma = np.cov(pre_expansion_embeddings, rowvar=False, bias=True)
            new_embeddings = np.random.multivariate_normal(mu, 1e-5 * sigma, size=64)
            weights = np.vstack([pre_expansion_embeddings, new_embeddings])
            converted_weights = [weights]
        else:
            raise ValueError(f"Unexpected member, {prop}, in Embedder.")
    elif path.startswith(f"{_TRANSFORMER_EMBEDDER}/mm"):
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


def transpose_reshape(x: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2]).contiguous()


@dataclasses.dataclass(frozen=True)
class ConversionResult:
    state_tree: dict[str, torch.Tensor]
    config: ShieldGemma2Config


def convert(
    shieldgemma_checkpoint_path: str,
    gemma_checkpoint_path: str,
    config: ShieldGemma2Config,
    target_dtype: torch.dtype,
) -> ConversionResult:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    checkpointer = obc.PyTreeCheckpointer()

    sg2_ckpt = checkpointer.restore(shieldgemma_checkpoint_path)
    g3_ckpt = checkpointer.restore(gemma_checkpoint_path)

    hf_tree: dict[str, torch.Tensor] = {}

    def update_tree(path: str, weights: np.ndarray) -> None:
        torch_tensor = torch.from_numpy(weights.astype("float32")).type(target_dtype)
        logging.info(
            "%s converted shape=%s with dtype=%s",
            path,
            weights.shape,
            torch_tensor.dtype,
        )
        hf_tree[f"model.{path}"] = torch_tensor

    for paths, value in tree.flatten_with_path(g3_ckpt):
        if paths[0].startswith("SigLiPFromPatches_"):
            path, weights = convert_siglip_weight(config=config.vision_config, paths=paths, weights=value)
            update_tree(path, weights)

    for paths, value in tree.flatten_with_path(sg2_ckpt):
        for path, weights in convert_transformer_weights(config=config.text_config, paths=paths, weights=value):
            update_tree(path, weights)

    hf_tree["model.language_model.lm_head.weight"] = hf_tree["model.language_model.model.embed_tokens.weight"]

    return ConversionResult(state_tree=hf_tree, config=config)


def main(*args):
    del args

    dtype = getattr(torch, PRECISION.value)
    output_path = OUTPUT_PATH.value

    tokenizer = GemmaTokenizerFast(
        TOKENIZER_PATH.value,
        extra_special_tokens={
            "image_token": "<image_soft_token>",  # Should be ID=262_144
            "boi_token": "<start_of_image>",  # Should be ID=255_999
            "eoi_token": "<end_of_image>",  # Should be ID=256_000
        },
    )

    yes_token_index, no_token_index = torch.tensor(tokenizer(["Yes", "No"])["input_ids"])[:, 1].numpy()

    config = ShieldGemma2Config(
        yes_token_index=int(yes_token_index),
        no_token_index=int(no_token_index),
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
            max_position_embeddings=8192,
        ),
        vision_config={
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
        },
    )

    config.save_pretrained(output_path)

    image_processor = Gemma3ImageProcessor(
        image_seq_length=256,
        image_mean=(0.5,) * 3,
        image_std=(0.5,) * 3,
        size={"height": 896, "width": 896},
        resample=PILImageResampling.BILINEAR,
    )
    processor = ShieldGemma2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        policy_definitions=_SHIELDGEMMA2_POLICIES,
    )
    tokenizer.chat_template = _CHAT_TEMPLATE
    processor.chat_template = _CHAT_TEMPLATE

    processor.save_pretrained(output_path)
    logging.info("Saved Shieldgemma2Processor to %s", output_path)
    del processor
    del tokenizer

    logging.info("Converting Shieldgemma2 @ %s", dtype)
    result = convert(_SHIELDGEMMA_CHECKPOINT_PATH.value, _GEMMA_CHECKPOINT_PATH.value, config, dtype)
    logging.info("Converted Shieldgemma2 state tree from Orbax to Hugging Face.")

    with accelerate.init_empty_weights():
        model = ShieldGemma2ForImageClassification(config=config)

    model.load_state_dict(result.state_tree, assign=True, strict=True)
    model.config.dtype = dtype
    logging.info("Loaded Shieldgemma2 in Hugging Face Transformers.")
    model.save_pretrained(output_path, safe_serialization=True)
    logging.info("Saved Shieldgemma2 to SafeTensors in %s", output_path)
    del model
    del result


if __name__ == "__main__":
    app.run(main)
