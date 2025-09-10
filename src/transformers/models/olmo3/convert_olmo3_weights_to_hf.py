# Copyright 2025 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import pickle
import shutil
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
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    ReadItem,
)
from torch.futures import Future

from transformers import AutoTokenizer, Olmo3Config, Olmo3ForCausalLM


"""
Sample usage:

```
python src/transformers/models/olmo3/convert_olmo3_weights_to_hf.py \
    --input_dir /path/to/downloaded/olmo3/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import Olmo3ForCausalLM, AutoTokenizer

model = Olmo3ForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


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
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
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
                    # NOTE: we might get an error here that can't be pickled, which causes a different failure
                    # later when PyTorch tries to reduce that error across ranks. So here we just make
                    # sure we're raising a simple error type that can be pickled.
                    raise RuntimeError(f"Original error:\n{traceback.format_exc()}")

        # Modified from `FileSystemReader.read_data()`
        for read_item, content in read_item_content_results:
            bytes = io.BytesIO(content)
            bytes.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes)
            else:
                # NOTE: 'weights_only=False' needed to load torchao's float8 linear layer checkpoints
                tensor = cast(torch.Tensor, torch.load(bytes, map_location="cpu", weights_only=False))
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
                with (Path(self.path) / ".metadata").open("rb") as metadata_file:
                    metadata = pickle.load(metadata_file)
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


def load_model(model_path: str):
    def _load_unsharded_keys(
        dir: Path | str,
        keys: list[str],
        *,
        pre_download: bool = False,
        work_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

        state_dict: dict[str, Any] = {}
        _load_state_dict(
            state_dict,
            storage_reader=RemoteFileSystemReader(dir, pre_download=pre_download, work_dir=work_dir),
            planner=_EmptyStateDictLoadPlanner(keys=keys),
            no_dist=True,
        )
        return state_dict

    with (Path(model_path) / ".metadata").open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)
        keys = [key for key in metadata.state_dict_metadata.keys() if key.startswith("model.")]

    # keys = ["model.blocks.0.attention.w_q.weight"]

    return _load_unsharded_keys(
        model_path,
        keys,
        # model_path, ["model.blocks.0.attention.w_q.weight", "model.blocks.0.attention.w_k.weight"]
    )


def write_model(
    model_path,
    input_base_path,
    include_tokenizer=True,
    tokenizer_id=None,
    safe_serialization=True,
    tmp_cleanup=True,
):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.json"
    olmo3_config = json.loads(config_path.read_text())
    model_config = olmo3_config["model"]
    block_config = model_config["block"]
    attention_config = block_config["attention"]
    tokenizer_config = olmo3_config["dataset"]["tokenizer"]

    n_layers = model_config["n_layers"]
    n_heads = attention_config["n_heads"]
    dim = model_config["d_model"]
    dims_per_head = dim // n_heads
    base = attention_config["rope"]["theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = olmo3_config["train_module"]["max_sequence_length"]

    if attention_config.get("n_kv_heads", None) is not None:
        num_key_value_heads = model_config["n_kv_heads"]  # for GQA / MQA
    else:
        num_key_value_heads = n_heads

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    # Not sharded
    # (The sharded implementation would also work, but this is simpler.)
    loaded = load_model(os.path.join(input_base_path, "model_and_optim"))["model"]
    print(loaded.keys())
    # loaded = torch.load(os.path.join(input_base_path, "model.pt"), map_location="cpu", weights_only=True)

    param_count = 0
    index_dict: dict[str, Any] = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        # Unsharded
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
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                f"blocks.{layer_i}.attention_norm.weight"
            ],
            f"model.layers.{layer_i}.post_feedforward_layernorm.weight": loaded[
                f"blocks.{layer_i}.feed_forward_norm.weight"
            ],
        }

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"

    # Unsharded
    # TODO: Deal with weight-tying
    state_dict = {
        "model.embed_tokens.weight": loaded["embeddings.weight"],
        "model.norm.weight": loaded["lm_head.norm.weight"],
        "lm_head.weight": loaded["lm_head.w_out.weight"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = Olmo3Config(
        vocab_size=model_config["vocab_size"],
        hidden_size=dim,
        intermediate_size=block_config["feed_forward"]["hidden_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=tokenizer_config["eos_token_id"],
        tie_word_embeddings=False,
        rms_norm_eps=block_config["layer_norm"]["eps"],
        rope_theta=base,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        tokenizer_id = tokenizer_id or tokenizer_config["identifier"]
        _write_tokenizer(model_path, tokenizer_id)

    print("Loading the checkpoint in a Olmo 3 model.")
    model = Olmo3ForCausalLM.from_pretrained(tmp_model_path, dtype=torch.bfloat16)
    print("Resizing token embeddings to match tokenizer config.")
    model.resize_token_embeddings(tokenizer_config["vocab_size"])
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    if tmp_cleanup:
        # Make cleanup optional; attempting to `rmtree` the `tmp_model_path` causes
        # errors if using NFS.
        shutil.rmtree(tmp_model_path)


def _write_tokenizer(
    output_path: Path,
    tokenizer_id: str,
) -> None:
    print(f"Saving a tokenizer to {output_path}.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of Olmo 3 weights, which contains config.yaml and model.pt.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not convert OLMo tokenizer to HF tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Location of Olmo 3 tokenizer json file. Defaults to what is set in the config file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If passed, don't remove temp dir at end of HF conversion.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="Whether or not to save using `safetensors`.",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        include_tokenizer=args.include_tokenizer,
        tokenizer_id=args.tokenizer,
        tmp_cleanup=args.tmp_cleanup,
    )


if __name__ == "__main__":
    main()
