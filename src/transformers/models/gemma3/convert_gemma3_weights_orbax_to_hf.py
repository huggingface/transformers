r"""Utility to convert Gemma models from Orbax to HF Transformers checkpoint.

python3 -m transformers.models.gemma3.convert_gemma3_weights_orbax_to_hf \
    --variant="gemma3_4b" \
    --tokenizer_path='/usr/local/google/home/$USER/gemini_bpe_256k_v5_no_tags.model' \
    --checkpoint_path='/cns/ge-d/home/gdm-g-mini-team/gemma/checkpoints/orbax/gemma2_2b_pt' \
    --output_path='/usr/local/google/home/$USER/gemma/2b/' \
    --alsologtostderr

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

from collections.abc import Sequence
import dataclasses
import enum
import re
from typing import Any

from absl import app
from absl import flags
from absl import logging
import accelerate
import numpy as np
from orbax import checkpoint as obc
import torch
import tree

from ..gemma.tokenization_gemma import GemmaTokenizer
from .configuration_gemma3 import (
    DEFAULT_ATTENION_PATTERN,
    Gemma3Config,
    Gemma3TextConfig,
    Gemma3VisionConfig,
)

# ==== Internal Constants and Classes ====

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_SIGLIP_BASE = 'SigLiPFromPatches_0/siglip_encoder'
_SIGLIP_EMBEDDING = "SigLiPFromPatches_0/siglip_encoder/embedding"
_SIGLIP_TRANSFORMER_ENCODER_BLOCK = "SigLiPFromPatches_0/siglip_encoder/Transformer/encoderblock_"
_SIGLIP_TRANSFORMER_ENCODER_BLOCK_LEN = len(_SIGLIP_TRANSFORMER_ENCODER_BLOCK)
_SIGLIP_TRANSFORMER_ENCODER_NORM = "SigLiPFromPatches_0/siglip_encoder/Transformer/encoder_norm"

_VARIANT_GEMMA_3_1B = "gemma3_1b"
_VARIANT_GEMMA_3_4B = "gemma3_4b"
_VARIANT_GEMMA_3_12B = "gemma3_12b"
_VARIANT_GEMMA_3_27B = "gemma3_27b"
_VARIANTS = {
    _VARIANT_GEMMA_3_1B: Gemma3Config(
        text_config=Gemma3TextConfig(),
        vision_config=None,
    ),
    _VARIANT_GEMMA_3_4B: Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=262_144,
            num_hidden_layers=34,
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=2_560,
            intermediate_size=10_240,
            attention_pattern=DEFAULT_ATTENION_PATTERN,
            sliding_window=1_024,
            rope_global_base_freq=1_000_000,
            rope_local_base_freq=10_000,
            attn_logit_softcapping=None,
        ),
        vision_config=Gemma3VisionConfig(),
    ),
    _VARIANT_GEMMA_3_12B: Gemma3Config(
        text_config=Gemma3TextConfig(),
        vision_config=Gemma3VisionConfig(),
    ),
    _VARIANT_GEMMA_3_27B: Gemma3Config(
        text_config=Gemma3TextConfig(),
        vision_config=Gemma3VisionConfig(),
    ),
}

# ==== Flags ====

_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="Path to the Orbax checkpoint.",
    required=True,
)

_DEVICE = flags.DEFINE_string(
    name="device",
    default="cpu",
    help="Torch device to use for conversion.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    name="output_path",
    default=None,
    help="Path to store the HF checkpoint.",
    required=True,
)

_PRECISION = flags.DEFINE_enum(
    name="precision",
    default=None,
    help="The floating point precision (aka dtype) of the model.",
    enum_values=set(_DTYPES.keys()),
    required=True,
)

_TOKENIZER_PATH = flags.DEFINE_string(
    name="tokenizer_path",
    default=None,
    help="Path to the SentencePiece model file.",
    required=True,
)

_VARIANT = flags.DEFINE_enum(
    name="variant",
    default=None,
    help="The model variant to convert.",
    enum_values=set(_VARIANTS.keys()),
    required=True,
)


def _convert_siglip_weight(
      config: Gemma3VisionConfig,
      paths: Sequence[str],
      weights: Any,
) -> tuple[str, Any]:
    path, prop = paths
    normalized_path: str = ''
    updated_weights: Any = None

    if path == _SIGLIP_BASE:
        normalized_path = 'vision_tower.vision_model.embeddings.position_embedding.weight'
        updated_weights = weights.reshape(-1, config.hidden_size)
    elif path == _SIGLIP_EMBEDDING:
        if prop == "kernel":
            normalized_path = "vision_tower.vision_model.embeddings.patch_embedding.weight"
            updated_weights = weights.transpose(3, 2, 0, 1)
        elif prop == "bias":
            normalized_path = "vision_tower.vision_model.embeddings.patch_embedding.bias"
            updated_weights = weights
        else:
            raise ValueError(f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`.")
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
                raise ValueError(f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `scale`.")
        elif encoder_block_path.startswith("/MlpBlock_0"):
            normalized_path += ".mlp.fc1" if "/Dense_0" in encoder_block_path else ".mlp.fc2"

            if prop == "kernel":
                normalized_path += ".weight"
                updated_weights = weights.transpose()
            elif prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights
            else:
                raise ValueError(f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`.")
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
                raise ValueError(f"Upexpected path `{path}` in SigLIP Transformer MultiHeadDotProductAttention_0.")

            if prop == "bias":
                normalized_path += ".bias"
                updated_weights = weights.reshape(-1, config.hidden_size).reshape(-1)
            elif prop == "kernel":
                normalized_path += ".weight"
                updated_weights = weights.reshape(-1, config.hidden_size).transpose()
            else:
                raise ValueError(f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `kernel`.")
        else:
            raise ValueError(f"Upexpected path `{path}` in SigLIP Transformer Encoder Block.")
    elif path == _SIGLIP_TRANSFORMER_ENCODER_NORM:
        if prop == "scale":
            normalized_path = "vision_tower.vision_model.post_layernorm.weight"
            updated_weights = weights.transpose()
        elif prop == "bias":
            normalized_path = "vision_tower.vision_model.post_layernorm.bias"
            updated_weights = weights
        else:
            raise ValueError(f"Upexpected member, `{prop}`, for path `{path}`. Should be `bias` or `scale`.")
    else:
        raise ValueError(f"Upexpected path `{path}`.")

    return normalized_path, updated_weights


layer_names = """
"('transformer/embedder', 'input_embedding')"
"('transformer/embedder', 'mm_input_embedding_extra')"
"('transformer/embedder', 'mm_output_embedding')"
"('transformer/embedder/mm_input_projection', 'w')"
"('transformer/embedder/mm_soft_embedding_norm', 'scale')"
"('transformer/final_norm', 'scale')"
"('transformer/layer_i/', 'w')"
"('transformer/layer_i/', 'scale')"
"('transformer/layer_i/', 'w')"
"('transformer/layer_i/', 'w')"
"('transformer/layer_i/', 'scale')"
"('transformer/layer_i/', 'w')"
"('transformer/layer_i/', 'w')"
"('transformer/layer_i/', 'scale')"
"('transformer/layer_i/', 'scale')"
"('transformer/layer_i/', 'scale')"
"('transformer/layer_i/', 'scale')"
"""

_TRANSFORMER_DECODER_BLOCK = "transformer/layer_"
_TRANSFORMER_DECODER_BLOCK_LEN = len(_TRANSFORMER_DECODER_BLOCK)
_TRANSFORMER_EMBEDDER = "transformer/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"


def _convert_transformer_weights(
    config: Gemma3TextConfig,
    paths: Sequence[str],
    weights: Any
) -> tuple[str, Any]:
    path, prop = paths
    normalized_path: str = ''
    updated_weights: Any = None

    if path == _TRANSFORMER_EMBEDDER:
        if prop == "inpup_embedding":
            normalized_path = "language_model.model.embed_tokens.weight"
            updated_weights = weights
        else:
            pass
    elif _TRANSFORMER_EMBEDDER in path:
        pass
    elif path == _TRANSFORMER_FINAL_NORM:
        normalized_path = "language_model.model.norm.weight"
        updated_weights = weights
    elif path.startswith(_TRANSFORMER_DECODER_BLOCK):
        decoder_block_path = path[_TRANSFORMER_DECODER_BLOCK_LEN:]
        next_path_seperator_idx = decoder_block_path.find("/")
        layer_idx = decoder_block_path[:next_path_seperator_idx]
        decoder_block_path = decoder_block_path[next_path_seperator_idx:]

        normalized_path = f"vision_tower.vision_model.encoder.layers.{layer_idx}"

        if path.endswith("attn/attn_vec_einsum"):
            pass
        elif path.endswith("attn/_key_norm"):
            pass
        elif path.endswith("attn/kv_einsum"):
            pass
        elif path.endswith("attn/q_einsum"):
            pass
        elif path.endswith("attn/_query_norm"):
            pass
        elif path.endswith("mlp/gating_einsum"):
            pass
        elif path.endswith("mlp/linear"):
            pass
        elif path.endswith("post_attention_norm"):
            pass
        elif path.endswith("post_ffw_norm"):
            pass
        elif path.endswith("pre_attention_norm"):
            pass
        elif path.endswith("pre_ffw_norm"):
            pass
        else:
            raise ValueError(f"Upexpected path `{path}` in Decoder Block.")
    else:
        raise ValueError(f"Upexpected path `{path}`.")

    return normalized_path, updated_weights


def transpose_reshape(x: torch.Tensor) -> torch.Tensor:
  x = x.transpose(1, 2)
  return x.reshape(x.shape[0] * x.shape[1], x.shape[2]).contiguous()


@dataclasses.dataclass(frozen=True)
class ConversionResult:
  state_tree: dict[str, torch.Tensor]
  config: Gemma3Config


def convert(
    variant: str,
    checkpoint_path: str,
    config: Gemma3Config,
    target_dtype: torch.dtype,
    tokenizer: GemmaTokenizer,
) -> ConversionResult:
  """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
  checkpointer = obc.PyTreeCheckpointer()
  ckpt = checkpointer.restore(checkpoint_path)
  hf_tree: dict[str, torch.Tensor] = {}

  for paths, value in tree.flatten_with_path(ckpt):
    if paths[0].startswith('SigLiPFromPatches_'):
        path, weights = _convert_siglip_weight(config=config.vision_config, paths=paths, weights=value)
    else:
        path, weights = _convert_transformer_weights(config=config.text_config, paths=paths, weights=value)

    if path and weights is not None:
        logging.info(
            "%s converted from shape=%s to shape=%s with dtype=%s",
            path,
            value.shape,
            weights.shape,
            weights.dtype,
        )
        hf_tree[path] = weights

  return ConversionResult(state_tree=hf_tree, config=config)


def validate(
    path: str,
    gemma_model_cls: Any
):
  """Runs some sample prompts through the model."""
  model = gemma_model_cls.from_pretrained(path)
  tokenizer = GemmaTokenizer.from_pretrained(path)
  prompts = [
      "import sys; sys.",
      (
          '<|fim_prefix|>import <|fim_suffix|>\nif __name__ == "__main__":\n '
          " sys.exit(0)\n<|fim_middle|>"
      ),
  ]
  for prompt in prompts:
    logging.info("Prompt: %s", prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    logging.info("Inputs: %s", inputs)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=30)
    logging.info("Generates: %s", generate_ids)
    r = tokenizer.batch_decode(generate_ids, skip_special_token=True)[0]
    logging.info("All together: %s", r)


def main(*args):
  del args

  variant = _VARIANT.value
  dtype = _PRECISION.value
  logging.info('Converting Gemma 3 (%s) @ %s', variant, dtype)
  tokenizer = GemmaTokenizer(_TOKENIZER_PATH.value)

  config = _VARIANTS[variant]
  logging.info('Gemma 3 (%s) configured as: %s', variant, config)
  result = convert(
      variant, _CHECKPOINT_PATH.value, config, getattr(torch, dtype), tokenizer
  )
  hf_tree = result.state_tree
  config = result.config

  # with accelerate.init_empty_weights():
  #   model = gemma_model_cls(config)
  # model.load_state_dict(hf_tree, assign=True, strict=True)
  # model.config.torch_dtype = dtype
  # del model.config._name_or_path  # pylint: disable=protected-access

  # model.save_pretrained(_OUTPUT_PATH.value, safe_serialization=True)
  # del hf_tree
  # del model

  tokenizer.save_pretrained(_OUTPUT_PATH.value)
  del tokenizer

  # validate(_OUTPUT_PATH.value, gemma_model_cls)


if __name__ == "__main__":
  app.run(main)
