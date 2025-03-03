r"""Utility to convert Gemma models from Orbax to HF Transformers checkpoint.

python3 -m transformers.models.gemma3.convert_gemma3_weights_orbax_to_hf \
    --variant='gemma3_4b' \
    --tokenizer_path="$HOME/gemma3/tokenizer.model" \
    --checkpoint_path="$HOME/gemma3/gemma3_4B_pt_orbax/" \
    --output_path="$HOME/gemma3/gemma3_4B_pt_safetensors/" \
    --precision='bfloat16'

This requires Transformers v4.38+. It was written in a way that it can be run as
a regular non-Google3 script. So you can either `blaze run` it, or `python3` it.

NOTE: The internal HF implementation does not currently support writing
SafeTensors to CNS or saving tokenizers with sentencepiece vocab files loaded
from CNS. If running with blaze, use local paths for both sentencepiece_path and
output_path.
See b/390224023

Table stakes:
- FP32
- BF16
- INT8
- INT4
"""

from collections.abc import Iterator, Sequence
import dataclasses
from typing import Any

from absl import app
from absl import flags
from absl import logging
import accelerate
import numpy as np
from orbax import checkpoint as obc
import torch
import tree

from ..gemma import GemmaTokenizerFast
from ..siglip import SiglipImageProcessor
from .configuration_gemma3 import (
    DEFAULT_ATTENION_PATTERN,
    Gemma3Config,
    Gemma3TextConfig,
    Gemma3VisionConfig,
)
from . import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
)

# ==== Internal Constants and Classes ====

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_SIGLIP_BASE = "SigLiPFromPatches_0/siglip_encoder"
_SIGLIP_EMBEDDING = "SigLiPFromPatches_0/siglip_encoder/embedding"
_SIGLIP_TRANSFORMER_ENCODER_BLOCK = (
    "SigLiPFromPatches_0/siglip_encoder/Transformer/encoderblock_"
)
_SIGLIP_TRANSFORMER_ENCODER_BLOCK_LEN = len(_SIGLIP_TRANSFORMER_ENCODER_BLOCK)
_SIGLIP_TRANSFORMER_ENCODER_NORM = (
    "SigLiPFromPatches_0/siglip_encoder/Transformer/encoder_norm"
)

_TRANSFORMER_DECODER_BLOCK = "transformer/layer_"
_TRANSFORMER_DECODER_BLOCK_LEN = len(_TRANSFORMER_DECODER_BLOCK)
_TRANSFORMER_EMBEDDER = "transformer/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"
_TRANSFORMER_POST_TRAINING_PREFIX = "rlx_networks/policy_network/"
_TRANSFORMER_POST_TRAINING_PREFIX_LEN = len(_TRANSFORMER_POST_TRAINING_PREFIX)

_VARIANT_GEMMA_3_1B = "gemma3_1b"
_VARIANT_GEMMA_3_4B = "gemma3_4b"
_VARIANT_GEMMA_3_12B = "gemma3_12b"
_VARIANT_GEMMA_3_27B = "gemma3_27b"
_VARIANTS = {
    _VARIANT_GEMMA_3_1B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_144,
            hidden_size=1152,
            intermediate_size=6912,
            num_attention_heads=4,
            num_hidden_layers=26,
            num_key_value_heads=1,
            attention_pattern=DEFAULT_ATTENION_PATTERN,
            sliding_window=1024,
            rope_global_base_freq=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=256 ** -0.5,
            max_position_embeddings=32_768,
        ),
        vision_config=None,
    ),
    _VARIANT_GEMMA_3_4B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_144,
            hidden_size=2560,
            intermediate_size=10_240,
            num_attention_heads=8,
            num_hidden_layers=34,
            num_key_value_heads=4,
            attention_pattern=DEFAULT_ATTENION_PATTERN,
            sliding_window=1024,
            rope_global_base_freq=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=256 ** -0.5,
        ),
        vision_config=Gemma3VisionConfig(),
    ),
    _VARIANT_GEMMA_3_12B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_144,
            hidden_size=3840,
            intermediate_size=3840 * 8 // 2,
            num_attention_heads=16,
            num_hidden_layers=48,
            num_key_value_heads=8,
            attention_pattern=DEFAULT_ATTENION_PATTERN,
            sliding_window=1024,
            rope_global_base_freq=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=256 ** -0.5,
        ),
        vision_config=Gemma3VisionConfig(),
    ),
    _VARIANT_GEMMA_3_27B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_144,
            hidden_size=5376,
            intermediate_size=5376 * 8 // 2,
            num_attention_heads=32,
            num_hidden_layers=62,
            num_key_value_heads=16,
            head_dim=128,
            attention_pattern=DEFAULT_ATTENION_PATTERN,
            sliding_window=1024,
            rope_global_base_freq=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
            query_pre_attn_scalar=5376 // 32,   # hidden_size // num_attention_heads
        ),
        vision_config=Gemma3VisionConfig(),
    ),
}

# ==== Flags ====

CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="Path to the Orbax checkpoint.",
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

_TEXT_ONLY = flags.DEFINE_bool(
    name="text_only",
    default=False,
    help=(
        "If True, the model is loaded and saved as a Gemma3ForCausalLM, "
        "otherwise model saed as Gemma3ForConditionalGeneration."
    ),
)

TOKENIZER_PATH = flags.DEFINE_string(
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


def convert_siglip_weight(
    config: Gemma3VisionConfig,
    paths: Sequence[str],
    weights: np.ndarray,
) -> tuple[str, np.ndarray]:
    path, prop = paths
    normalized_path: str = ""
    updated_weights: np.ndarray = None

    if path == _SIGLIP_BASE:
        normalized_path = (
            "vision_model.vision_model.embeddings.position_embedding.weight"
        )
        updated_weights = weights.reshape(-1, config.hidden_size)
    elif path == _SIGLIP_EMBEDDING:
        if prop == "kernel":
            normalized_path = (
                "vision_model.vision_model.embeddings.patch_embedding.weight"
            )
            updated_weights = weights.transpose(3, 2, 0, 1)
        elif prop == "bias":
            normalized_path = (
                "vision_model.vision_model.embeddings.patch_embedding.bias"
            )
            updated_weights = weights
        else:
            raise ValueError(
                f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`."
            )
    elif path.startswith(_SIGLIP_TRANSFORMER_ENCODER_BLOCK):
        encoder_block_path = path[_SIGLIP_TRANSFORMER_ENCODER_BLOCK_LEN:]
        next_path_seperator_idx = encoder_block_path.find("/")
        layer_idx = encoder_block_path[:next_path_seperator_idx]
        encoder_block_path = encoder_block_path[next_path_seperator_idx:]
        normalized_path = f"vision_model.vision_model.encoder.layers.{layer_idx}"

        if encoder_block_path.startswith("/LayerNorm"):
            normalized_path += ".layer_norm1" if path.endswith("_0") else ".layer_norm2"

            if prop == "scale":
                normalized_path += ".weight"
                updated_weights = weights.transpose()
            elif prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights
            else:
                raise ValueError(
                    f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `scale`."
                )
        elif encoder_block_path.startswith("/MlpBlock_0"):
            normalized_path += (
                ".mlp.fc1" if "/Dense_0" in encoder_block_path else ".mlp.fc2"
            )

            if prop == "kernel":
                normalized_path += ".weight"
                updated_weights = weights.transpose()
            elif prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights
            else:
                raise ValueError(
                    f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`."
                )
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
                raise ValueError(
                    f"Upexpected path `{path}` in SigLIP Transformer MultiHeadDotProductAttention_0."
                )

            if prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights.reshape(-1, config.hidden_size).reshape(-1)
            elif prop == "kernel":
                normalized_path += ".weight"
                updated_weights = weights.reshape(-1, config.hidden_size).transpose()
            else:
                raise ValueError(
                    f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`."
                )
        else:
            raise ValueError(
                f"Upexpected path `{path}` in SigLIP Transformer Encoder Block."
            )
    elif path == _SIGLIP_TRANSFORMER_ENCODER_NORM:
        if prop == "scale":
            normalized_path = "vision_model.vision_model.post_layernorm.weight"
            updated_weights = weights.transpose()
        elif prop == "bias":
            normalized_path = "vision_model.vision_model.post_layernorm.bias"
            updated_weights = weights
        else:
            raise ValueError(
                f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `scale`."
            )
    else:
        raise ValueError(f"Upexpected path `{path}`.")

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
            converted_paths = [
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            ]
            converted_weights = [weights, weights]
        elif _TEXT_ONLY.value or prop == "mm_output_embedding":
            return zip([], [])
        elif prop == "mm_input_embedding_extra":
            converted_paths = ["mm_input_embedding_extra.weight"]
            converted_weights = [weights]
        else:
            raise ValueError(f"Upexpected member, {prop}, in Embedder.")
    elif path.startswith(f"{_TRANSFORMER_EMBEDDER}/mm"):
        if _TEXT_ONLY.value:
            return zip([], [])

        if path.endswith("/mm_input_projection"):
            converted_paths = ["mm_input_projection.weight"]
            converted_weights = [weights]
        elif path.endswith("/mm_soft_embedding_norm"):
            converted_paths = ["mm_soft_emb_norm.weight"]
            converted_weights = [weights]
        else:
            raise ValueError(f"Upexpected subpath, `{path}`, in Embedder.")
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
            converted_weights = [
                weights.transpose(2, 0, 1).reshape(config.hidden_size, attn_head_dim)
            ]
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
                k_proj_weights.transpose(0, 2, 1).reshape(
                    kv_head_dim, config.hidden_size
                ),
                v_proj_weights.transpose(0, 2, 1).reshape(
                    kv_head_dim, config.hidden_size
                ),
            ]
        elif path.endswith("attn/q_einsum"):
            converted_paths = [f"{base_path}.self_attn.q_proj.weight"]
            converted_weights = [
                weights.transpose(0, 2, 1).reshape(attn_head_dim, config.hidden_size)
            ]
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
            raise ValueError(f"Upexpected path `{path}` in Decoder Block.")
    else:
        raise ValueError(f"Upexpected path `{path}`.")

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
    config: Gemma3Config


def convert(
    checkpoint_path: str,
    config: Gemma3Config,
    target_dtype: torch.dtype,
) -> ConversionResult:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    checkpointer = obc.PyTreeCheckpointer()
    ckpt = checkpointer.restore(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    def update_tree(path: str, weights: np.ndarray) -> None:
        torch_tensor = torch.from_numpy(weights.astype("float32")).type(target_dtype)
        logging.info(
            "%s converted shape=%s with dtype=%s",
            path,
            weights.shape,
            torch_tensor.dtype,
        )
        hf_tree[path] = torch_tensor

    for paths, value in tree.flatten_with_path(ckpt):
        if paths[0].startswith("SigLiPFromPatches_"):
            if config.vision_config is None:
                continue

            path, weights = convert_siglip_weight(
                config=config.vision_config, paths=paths, weights=value
            )
            update_tree(path, weights)
        else:
            for path, weights in convert_transformer_weights(
                config=config.text_config, paths=paths, weights=value
            ):
                if config.vision_config is None:
                    path = path[len("language_model."):]

                update_tree(path, weights)

    if not _TEXT_ONLY.value:
        extra_embs = hf_tree.pop("mm_input_embedding_extra.weight").float()
        soft_emb_norm = hf_tree["mm_soft_emb_norm.weight"].float()
        soft_emb_proj = hf_tree["mm_input_projection.weight"].float()
        lm_embs = hf_tree["language_model.model.embed_tokens.weight"].float()

        extra_embs_expanded = extra_embs.unsqueeze(0)
        extra_embs_norm = extra_embs_expanded * torch.rsqrt(
            extra_embs_expanded.pow(2).mean(-1, keepdim=True) + config.text_config.rms_norm_eps
        )
        extra_embs_norm = extra_embs_norm * (1.0 + soft_emb_norm)
        extra_embs_proj = torch.einsum('btm,md->btd', extra_embs_norm, soft_emb_proj)
        extra_embs_proj = extra_embs_proj.squeeze()
        embs = torch.cat([lm_embs, extra_embs_proj], dim=0)

        hf_tree["language_model.model.embed_tokens.weight"] = embs
        hf_tree["language_model.lm_head.weight"] = embs.clone()

    return ConversionResult(state_tree=hf_tree, config=config)


def main(*args):
    del args

    variant = _VARIANT.value
    dtype = getattr(torch, PRECISION.value)
    config = _VARIANTS[variant]

    if variant == _VARIANT_GEMMA_3_1B:
        flags.FLAGS.set_default(_TEXT_ONLY.name, True)

    tokenizer = GemmaTokenizerFast(TOKENIZER_PATH.value)

    if _TEXT_ONLY.value:
        config.text_config.mm_vocab_size = 0
        config.vision_config = None
        output_path = f"{OUTPUT_PATH.value}_textonly"
        tokenizer.save_pretrained(output_path)
        logging.info("Saved GemmaTokenizer for %s to %s", variant, output_path)
        del tokenizer
    else:
        output_path = OUTPUT_PATH.value
        processor = Gemma3Processor(
            image_processor=SiglipImageProcessor(image_seq_length=1024),
            tokenizer=tokenizer
        )
        config.text_config.mm_soft_token_id = processor.image_soft_token_id
        processor.save_pretrained(output_path)
        logging.info("Saved Gemma3Processor for %s to %s", variant, output_path)
        del processor
        del tokenizer

    logging.info("Gemma 3 (%s) configured as: %s", variant, config)
    logging.info("Converting Gemma 3 (%s) @ %s", variant, dtype)
    result = convert(CHECKPOINT_PATH.value, config, dtype)
    logging.info("Converted Gemma 3 (%s) state tree from Orbax to Hugging Face.", variant)

    with accelerate.init_empty_weights():
        if config.vision_config is None:
            model = Gemma3ForCausalLM(config=config.text_config)
        else:
            model = Gemma3ForConditionalGeneration(config)

    model.load_state_dict(result.state_tree, assign=True, strict=True)
    model.config.torch_dtype = dtype
    logging.info(
        "Loaded Gemma 3 (%s) in Hugging Face Transformers as a %s instance.",
        variant,
        type(model).__name__
    )
    model.save_pretrained(output_path, safe_serialization=True)
    logging.info(
        "Saved Gemma 3 (%s) to SafeTensors in %s using %s",
        variant,
        output_path,
        type(model).__name__,
    )
    del model
    del result


if __name__ == "__main__":
    app.run(main)
