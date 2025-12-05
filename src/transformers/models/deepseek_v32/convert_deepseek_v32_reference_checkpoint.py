from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping

import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from ...modeling_utils import init_empty_weights
from ...utils import logging
from .configuration_deepseek_v32 import DeepseekV32Config
from .modular_deepseek_v32 import DeepseekV32ForCausalLM


logger = logging.get_logger(__name__)


def _load_state_dict(path: str) -> OrderedDict[str, torch.Tensor]:
    """
    Loads a state dict from a `.pt`/`.bin` or `.safetensors` file.

    Args:
        path (`str`): Path to the shard file.
    """

    if path.endswith(".safetensors"):
        tensors = safetensors_load_file(path, device="cpu")
        return OrderedDict((k, v) for k, v in tensors.items())

    state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    return OrderedDict(state)


def _normalize_dtype(tensor: torch.Tensor, target_dtype: torch.dtype | None) -> torch.Tensor:
    """Casts floating tensors to `target_dtype` when requested."""
    if target_dtype is None:
        return tensor
    if tensor.dtype == target_dtype:
        return tensor
    if tensor.is_floating_point():
        return tensor.to(target_dtype)
    return tensor


def _merge_slices(
    slices: list[torch.Tensor],
    target_shape: torch.Size,
    target_dtype: torch.dtype | None,
    world_size: int,
    key: str,
) -> torch.Tensor:
    """
    Reconstructs a dense tensor from tensor-parallel shards.

    The heuristics follow the DeepSeek reference layout:

    - Column parallel: concatenate along dim 0.
    - Row parallel: concatenate along dim 1.
    - Identical tensors across shards: keep a single copy.
    """

    if len(slices) == 1:
        return _normalize_dtype(slices[0], target_dtype)

    first = slices[0]
    same_shapes = all(t.shape == first.shape for t in slices)
    if same_shapes and first.shape == target_shape:
        # Parameters that are replicated on all ranks (e.g., biases, norms).
        # Validate consistency to avoid silent errors.
        if not all(torch.equal(first, t) for t in slices[1:]):
            raise ValueError(
                f"Inconsistent replicas for parameter `{key}`. "
                "Expected identical tensors across ranks when shapes already match the dense layout."
            )
        return _normalize_dtype(first, target_dtype)

    if first.dim() >= 1 and first.shape[0] * world_size == target_shape[0] and first.shape[1:] == target_shape[1:]:
        return _normalize_dtype(torch.cat(slices, dim=0), target_dtype)

    if (
        first.dim() >= 2
        and first.shape[0] == target_shape[0]
        and first.shape[1] * world_size == target_shape[1]
        and first.shape[2:] == target_shape[2:]
    ):
        return _normalize_dtype(torch.cat(slices, dim=1), target_dtype)

    raise ValueError(
        f"Unable to merge parameter `{key}`. "
        f"Sharded shape: {first.shape}, target shape: {target_shape}, world size: {world_size}."
    )


def _maybe_strip_prefix(state_dict: MutableMapping[str, torch.Tensor]) -> None:
    """
    Handles checkpoints saved from `DeepseekV32ForCausalLM` or `DeepseekV32Model`.
    """

    if any(k.startswith("module.") for k in state_dict):
        for key in list(state_dict.keys()):
            tensor = state_dict.pop(key)
            new_key = key.replace("module.", "", 1)
            state_dict[new_key] = tensor


def convert_reference_shards_to_dense(
    shard_paths: Iterable[str],
    config: DeepseekV32Config,
    output_path: str | None = None,
    dtype: torch.dtype | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Converts DeepSeek V3.2 reference tensor-parallel shards into a dense Hugging Face state dict.

    Args:
        shard_paths (`Iterable[str]`):
            Paths to per-rank checkpoints. All ranks must be provided and in rank order.
        config (`DeepseekV32Config`):
            Target configuration that defines the dense parameter shapes.
        output_path (`str`, *optional*):
            When provided, the merged weights will be saved as a `.safetensors` file.
        dtype (`torch.dtype`, *optional*):
            Optionally overrides floating-point tensors to a desired dtype (e.g., `torch.bfloat16`).

    Returns:
        `OrderedDict[str, torch.Tensor]`: Dense state dict ready to be loaded via `from_pretrained`.
    """

    shard_paths = list(shard_paths)
    if len(shard_paths) == 0:
        raise ValueError("`shard_paths` must contain at least one checkpoint shard.")

    logger.info("Loading %d tensor-parallel shards ...", len(shard_paths))
    shard_state_dicts = []
    for path in shard_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard `{path}` does not exist.")
        shard_state_dict = _load_state_dict(path)
        _maybe_strip_prefix(shard_state_dict)
        shard_state_dicts.append(shard_state_dict)

    world_size = len(shard_state_dicts)
    with init_empty_weights():
        model = DeepseekV32ForCausalLM(config)
        reference_state_dict = model.state_dict()

    merged_state_dict = OrderedDict()
    missing_keys = []

    for key, reference_tensor in reference_state_dict.items():
        slices = []
        for shard in shard_state_dicts:
            tensor = shard.get(key)
            if tensor is not None:
                slices.append(tensor)
        if not slices:
            missing_keys.append(key)
            continue

        merged_state_dict[key] = _merge_slices(
            slices=slices,
            target_shape=reference_tensor.shape,
            target_dtype=dtype or reference_tensor.dtype,
            world_size=world_size,
            key=key,
        )

    if missing_keys:
        raise ValueError(
            "The following parameters were not found in any shard: "
            + ", ".join(missing_keys[:20])
            + (" ..." if len(missing_keys) > 20 else "")
        )

    if output_path is not None:
        logger.info("Saving merged weights to %s", output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        safetensors_save_file(merged_state_dict, output_path)

    return merged_state_dict


def _parse_dtype_override(dtype_str: str | None) -> torch.dtype | None:
    if dtype_str is None:
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype override `{dtype_str}`. Choose from {list(mapping.keys())}.")
    return mapping[dtype_str]


def _load_config(config_path: str) -> DeepseekV32Config:
    if os.path.isdir(config_path):
        return DeepseekV32Config.from_pretrained(config_path)
    if config_path.endswith(".json"):
        return DeepseekV32Config.from_json_file(config_path)
    # Fall back to repo id / HF Hub reference.
    return DeepseekV32Config.from_pretrained(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepSeek V3.2 tensor-parallel reference shards into a dense Hugging Face checkpoint."
        )
    )
    parser.add_argument(
        "--shard_paths",
        nargs="+",
        required=True,
        help="Paths to tensor-parallel shards. Provide one path per rank in order.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the DeepSeek V3.2 config JSON or directory containing it.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination `.safetensors` file for the merged weights.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Optional dtype override for floating-point tensors (default: bfloat16).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(args.config)
    dtype = _parse_dtype_override(args.dtype)
    convert_reference_shards_to_dense(
        shard_paths=args.shard_paths,
        config=config,
        output_path=args.output,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()

