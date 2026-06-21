# Copyright 2026 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
"""
Convert OLMo Hybrid model checkpoints (with FLA layers) to HuggingFace format.

This script handles OLMo Hybrid models that mix standard attention layers with
linear attention (GatedDeltaNet) layers.

UPDATED: Now aligned with the OLMo-core conversion script, including support for:
- Configurable dtype (defaults to bfloat16)
- Configurable max_sequence_length via CLI
- Device selection

Sample usage:

```bash
TRUST_REMOTE_CODE=True python src/transformers/models/olmo_hybrid/convert_olmo_hybrid_weights_to_hf.py \
    --input_dir /path/to/downloaded/olmo_hybrid/weights \
    --output_dir /output/path
```

Thereafter, models can be loaded via:

```python
from transformers import OlmoHybridForCausalLM, AutoTokenizer

model = OlmoHybridForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import pickle
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.futures import Future

from transformers import AutoTokenizer, OlmoHybridConfig


# Mapping from string dtype names to torch dtypes
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def strtobool(val):
    """
    Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'.
    False values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    if isinstance(val, bool):
        return val
    val = str(val).lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value {val!r}")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def normalize_path(path: Path | str) -> str:
    return str(path).rstrip("/").replace("file://", "")


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_bytes_range(path: Path | str, bytes_start: int, num_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(bytes_start)
        return f.read(num_bytes)


def _narrow_tensor_by_index(tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


class RemoteFileSystemReader(dist_cp.StorageReader):
    """
    A :class:`~torch.distributed.checkpoint.StorageReader` based on :class:`~torch.distributed.checkpoint.FileSystemReader`
    that can read data directly from cloud storage as well as a local directory.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        thread_count: int | None = None,
        pre_download: bool = False,
        work_dir: Path | str | None = None,
    ):
        super().__init__()
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        self.path = normalize_path(path)
        self.thread_count = thread_count or 1
        self.pre_download = pre_download
        self.work_dir = normalize_path(work_dir) if work_dir is not None else None
        self.storage_data: dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = generate_uuid()
        self._metadata: Metadata | None = None

    def _get_bytes(self, relative_path: str, offset: int, length: int) -> bytes:
        full_path = f"{self.path}/{relative_path}"
        return get_bytes_range(full_path, offset, length)

    def _get_content_for_read(self, read_item: ReadItem) -> tuple[ReadItem, bytes]:
        sinfo = self.storage_data[read_item.storage_index]
        content = self._get_bytes(sinfo.relative_path, sinfo.offset, sinfo.length)
        return (read_item, content)

    def reset(self, checkpoint_id: Path | str | None = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = normalize_path(checkpoint_id)
        self.load_id = generate_uuid()

    def read_data(self, plan: dist_cp.LoadPlan, planner: dist_cp.LoadPlanner) -> Future[None]:
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            read_item_content_futures = []
            for read_item in plan.items:
                read_item_content_futures.append(executor.submit(self._get_content_for_read, read_item))
            read_item_content_results = []
            for f in as_completed(read_item_content_futures):
                try:
                    read_item_content_results.append(f.result())
                except BaseException:
                    raise RuntimeError(f"Original error:\n{traceback.format_exc()}")

        for read_item, content in read_item_content_results:
            bytes_io = io.BytesIO(content)
            bytes_io.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes_io)
            else:
                tensor = cast(torch.Tensor, torch.load(bytes_io, map_location="cpu", weights_only=False))
                tensor = _narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)
                target_tensor = planner.resolve_tensor(read_item).detach()

                assert target_tensor.size() == tensor.size(), (
                    f"req {read_item.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                )
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        if self._metadata is None:
            try:
                if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
                    raise ValueError(
                        "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                        "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                        "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                        "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
                    )
                with (Path(self.path) / ".metadata").open("rb") as metadata_file:
                    metadata = restricted_load(metadata_file)
            except FileNotFoundError as exc:
                msg = f"'{self.path}' is not a distributed checkpoint folder."
                suggested_dir = os.path.join(self.path, "model_and_optim")
                if Path(os.path.join(suggested_dir, ".metadata")).exists():
                    msg += f" Did you mean to use '{suggested_dir}'?"
                raise FileNotFoundError(msg) from exc

            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = StorageMeta()
            metadata.storage_meta.load_id = self.load_id

            self._metadata = metadata

        return self._metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: dist_cp.LoadPlan) -> dist_cp.LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: list[dist_cp.LoadPlan]) -> list[dist_cp.LoadPlan]:
        return global_plan

    @property
    def checkpoint_id(self) -> str:
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Path | str) -> bool:
        del checkpoint_id
        return True


class _RestrictedUnpickler(pickle.Unpickler):
    """
    Custom unpickler that handles missing olmo_core module references.
    This allows loading checkpoints saved with olmo_core without having it installed.
    """

    def find_class(self, module, name):
        if module.startswith("torch"):
            return super().find_class(module, name)
        if module in ("collections", "builtins", "_collections_abc"):
            return super().find_class(module, name)
        if module.startswith("olmo_core"):
            return super().find_class("builtins", "dict") if name == "dict" else type(name, (), {})
        return super().find_class(module, name)


def restricted_loads(data):
    """Load pickle data with restricted unpickler."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def restricted_load(file):
    """Load pickle file with restricted unpickler."""
    return _RestrictedUnpickler(file).load()


def load_model(model_path: str):
    """Load model state dict from distributed checkpoint."""
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    def _load_unsharded_keys(
        dir: Path | str,
        keys: list[str],
        *,
        pre_download: bool = False,
        work_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        _load_state_dict(
            state_dict,
            storage_reader=RemoteFileSystemReader(dir, pre_download=pre_download, work_dir=work_dir),
            planner=_EmptyStateDictLoadPlanner(keys=keys),
            no_dist=True,
        )
        return state_dict

    if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
        raise ValueError(
            "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
            "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
            "that could have been tampered with. If you already verified the pickle data and decided to use it, "
            "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
        )
    with (Path(model_path) / ".metadata").open("rb") as metadata_file:
        metadata = restricted_load(metadata_file)
        keys = [key for key in metadata.state_dict_metadata.keys() if key.startswith("model.")]

    return _load_unsharded_keys(model_path, keys)


def get_layer_types_from_config(olmo_config: dict) -> list[str]:
    """
    Determine the layer types (full_attention, linear_attention)
    from the OLMo config.
    """
    model_config = olmo_config["model"]
    block_config = model_config["block"]
    n_layers = model_config["n_layers"]

    fla_hybrid_attention_indices = block_config.get("fla_hybrid_attention_indices", [])

    layer_types = []
    for i in range(n_layers):
        if i in fla_hybrid_attention_indices:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    return layer_types


def convert_attention_layer_weights(
    loaded: dict[str, torch.Tensor],
    layer_i: int,
) -> dict[str, torch.Tensor]:
    """Convert weights for an attention (full or sliding) layer."""
    state_dict = {
        f"model.layers.{layer_i}.self_attn.q_proj.weight": loaded[f"blocks.{layer_i}.attention.w_q.weight"],
        f"model.layers.{layer_i}.self_attn.k_proj.weight": loaded[f"blocks.{layer_i}.attention.w_k.weight"],
        f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"blocks.{layer_i}.attention.w_v.weight"],
        f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"blocks.{layer_i}.attention.w_out.weight"],
        f"model.layers.{layer_i}.self_attn.q_norm.weight": loaded[f"blocks.{layer_i}.attention.q_norm.weight"],
        f"model.layers.{layer_i}.self_attn.k_norm.weight": loaded[f"blocks.{layer_i}.attention.k_norm.weight"],
        f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w1.weight"],
        f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w2.weight"],
        f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w3.weight"],
        f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"blocks.{layer_i}.attention_norm.weight"],
        f"model.layers.{layer_i}.post_feedforward_layernorm.weight": loaded[
            f"blocks.{layer_i}.feed_forward_norm.weight"
        ],
    }

    return state_dict


def convert_fla_layer_weights(
    loaded: dict[str, torch.Tensor],
    layer_i: int,
) -> dict[str, torch.Tensor]:
    """Convert weights for a FLA (GatedDeltaNet / linear attention) layer."""
    state_dict = {
        f"model.layers.{layer_i}.linear_attn.q_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.q_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.k_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.k_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.v_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.v_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.g_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.g_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.a_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.a_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.b_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.b_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.o_proj.weight": loaded[f"blocks.{layer_i}.fla.inner.o_proj.weight"],
        f"model.layers.{layer_i}.linear_attn.q_conv1d.weight": loaded[f"blocks.{layer_i}.fla.inner.q_conv1d.weight"],
        f"model.layers.{layer_i}.linear_attn.k_conv1d.weight": loaded[f"blocks.{layer_i}.fla.inner.k_conv1d.weight"],
        f"model.layers.{layer_i}.linear_attn.v_conv1d.weight": loaded[f"blocks.{layer_i}.fla.inner.v_conv1d.weight"],
        f"model.layers.{layer_i}.linear_attn.o_norm.weight": loaded[f"blocks.{layer_i}.fla.inner.o_norm.weight"],
        f"model.layers.{layer_i}.linear_attn.A_log": loaded[f"blocks.{layer_i}.fla.inner.A_log"],
        f"model.layers.{layer_i}.linear_attn.dt_bias": loaded[f"blocks.{layer_i}.fla.inner.dt_bias"],
        f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w1.weight"],
        f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w2.weight"],
        f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w3.weight"],
        f"model.layers.{layer_i}.attention_layer_norm.weight": loaded[f"blocks.{layer_i}.fla_norm.weight"],
        f"model.layers.{layer_i}.feedforward_layer_norm.weight": loaded[f"blocks.{layer_i}.feed_forward_norm.weight"],
    }
    return state_dict


def write_model(
    model_path: str,
    input_base_path: str,
    include_tokenizer: bool = True,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
):
    """
    Convert OLMo Hybrid checkpoint to HuggingFace format.

    Args:
        model_path: Output directory for the HuggingFace model.
        input_base_path: Path to the OLMo checkpoint directory containing config.json and model_and_optim/.
        include_tokenizer: Whether to save the tokenizer alongside the model.
        tokenizer_id: HuggingFace tokenizer identifier. Defaults to the one in the config.
        max_sequence_length: Override for max sequence length. If None, read from config.
        dtype: Torch dtype for the output model weights.
        device: Device to use for loading/conversion (e.g., "cpu", "cuda"). Defaults to CPU.
    """
    os.makedirs(model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.json"
    olmo_config = json.loads(config_path.read_text())
    model_config = olmo_config["model"]
    block_config = model_config["block"]
    attention_config = block_config.get("attention", {})
    fla_config = block_config.get("fla", {})
    tokenizer_config = olmo_config["dataset"]["tokenizer"]

    n_layers = model_config["n_layers"]
    n_heads = attention_config.get("n_heads", model_config.get("n_heads", 32))
    n_kv_heads = attention_config.get("n_kv_heads", n_heads)
    dim = model_config["d_model"]

    rope_config = attention_config.get("rope")

    if rope_config is not None:
        rope_theta = rope_config.get("theta", 500000.0)

        # Build unified rope_parameters dict
        rope_parameters = {"rope_theta": rope_theta}

        rope_scaling_config = rope_config.get("scaling")
        if rope_scaling_config:
            if hasattr(rope_scaling_config, "to_hf_config"):
                rope_parameters.update(rope_scaling_config.to_hf_config())
            else:
                rope_parameters.update(rope_scaling_config)
        else:
            rope_parameters["rope_type"] = "default"
    else:
        rope_parameters = None

    # Resolve max_position_embeddings with priority:
    # CLI arg > train_module.max_sequence_length > dataset.sequence_length > fallback
    if max_sequence_length is None:
        max_sequence_length = olmo_config.get("train_module", {}).get("max_sequence_length")
    if max_sequence_length is None:
        max_sequence_length = olmo_config.get("dataset", {}).get("sequence_length")
    if max_sequence_length is None:
        max_sequence_length = 65536
        print(f"Warning: max_sequence_length not found in config or CLI, using default: {max_sequence_length}")

    max_position_embeddings = max_sequence_length

    layer_types = get_layer_types_from_config(olmo_config)

    fla_layer_kwargs = fla_config.get("fla_layer_kwargs", {})
    linear_key_head_dim = fla_layer_kwargs.get("head_dim", 96)
    linear_value_head_dim = fla_layer_kwargs.get("head_v_dim", linear_key_head_dim * 2)
    linear_num_heads = fla_layer_kwargs.get("num_heads", n_heads)
    linear_conv_kernel_dim = fla_layer_kwargs.get("conv_kernel_dim", 4)
    linear_allow_neg_eigval = fla_layer_kwargs.get("allow_neg_eigval", True)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    loaded = load_model(os.path.join(input_base_path, "model_and_optim"))["model"]
    print(f"Loaded {len(loaded)} keys from checkpoint")

    param_count = 0
    full_state_dict: dict[str, torch.Tensor] = {}

    for layer_i in range(n_layers):
        layer_type = layer_types[layer_i]

        if layer_type == "linear_attention":
            layer_state = convert_fla_layer_weights(loaded, layer_i)
        else:
            layer_state = convert_attention_layer_weights(loaded, layer_i)

        full_state_dict.update(layer_state)
        param_count += sum(v.numel() for v in layer_state.values())
        print(f"Converted layer {layer_i} ({layer_type})")

    # Add embeddings and lm_head
    full_state_dict["model.embed_tokens.weight"] = loaded["embeddings.weight"]
    full_state_dict["model.norm.weight"] = loaded["lm_head.norm.weight"]
    full_state_dict["lm_head.weight"] = loaded["lm_head.w_out.weight"]
    param_count += sum(
        v.numel() for v in [loaded["embeddings.weight"], loaded["lm_head.norm.weight"], loaded["lm_head.w_out.weight"]]
    )

    # Cast all tensors to target dtype (matches OLMo-core behavior which casts everything,
    # including buffers like A_log and dt_bias)
    full_state_dict = {k: v.to(dtype) if torch.is_tensor(v) else v for k, v in full_state_dict.items()}

    print(f"Total parameters: {param_count}")

    config = OlmoHybridConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=dim,
        intermediate_size=block_config["feed_forward"]["hidden_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer_config.get("pad_token_id"),
        bos_token_id=tokenizer_config.get("bos_token_id"),
        eos_token_id=tokenizer_config.get("eos_token_id"),
        tie_word_embeddings=False,
        rms_norm_eps=block_config.get("layer_norm", {}).get("eps", 1e-6),
        rope_parameters=rope_parameters,
        layer_types=layer_types,
        linear_num_key_heads=linear_num_heads,
        linear_num_value_heads=linear_num_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        linear_allow_neg_eigval=linear_allow_neg_eigval,
    )
    if rope_parameters is None:
        config.rope_parameters = None
        config.rope_theta = None

    # Explicitly set architectures (normally set by model.save_pretrained, but we
    # save directly without the model roundtrip)
    config.architectures = ["OlmoHybridForCausalLM"]

    # Save config and weights directly (no from_pretrained roundtrip, which can
    # corrupt embeddings and fail to cast buffers like A_log)
    config.save_pretrained(model_path)

    from safetensors.torch import save_file

    safetensors_path = os.path.join(model_path, "model.safetensors")
    save_file(full_state_dict, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    del full_state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        tokenizer_id = tokenizer_id or tokenizer_config.get("identifier")
        if tokenizer_id:
            _write_tokenizer(model_path, tokenizer_id, max_sequence_length, tokenizer_config)

    # Update config with tokenizer info
    hf_config_path = Path(model_path) / "config.json"
    with open(hf_config_path, "r") as f:
        config_dict = json.load(f)

    config_dict["max_position_embeddings"] = max_position_embeddings
    config_dict["pad_token_id"] = tokenizer_config.get("pad_token_id")
    config_dict["bos_token_id"] = tokenizer_config.get("bos_token_id")
    config_dict["eos_token_id"] = tokenizer_config.get("eos_token_id")

    with open(hf_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print("Updated config.json with tokenizer settings")


def _write_tokenizer(
    output_path: Path | str,
    tokenizer_id: str,
    max_sequence_length: int | None = None,
    tokenizer_config: dict | None = None,
) -> None:
    """Save tokenizer with proper configuration matching OLMo-core behavior."""
    print(f"Saving tokenizer {tokenizer_id} to {output_path}.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if max_sequence_length is not None:
        tokenizer.model_max_length = max_sequence_length
    if tokenizer_config is not None:
        tokenizer.pad_token_id = tokenizer_config.get("pad_token_id")
        tokenizer.bos_token_id = tokenizer_config.get("bos_token_id")
        tokenizer.eos_token_id = tokenizer_config.get("eos_token_id")
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert OLMo Hybrid weights to HuggingFace format.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo Hybrid weights, which contains config.json and model_and_optim/.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not convert OLMo tokenizer to HF tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer identifier. Defaults to what is set in the config file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Max sequence length. If not set, reads from train_module.max_sequence_length or dataset.sequence_length in the config.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Output dtype for model weights. Defaults to bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for conversion (e.g., 'cpu', 'cuda'). Defaults to CPU.",
    )
    args = parser.parse_args()

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        include_tokenizer=args.include_tokenizer,
        tokenizer_id=args.tokenizer,
        max_sequence_length=args.max_sequence_length,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
    )


if __name__ == "__main__":
    main()
