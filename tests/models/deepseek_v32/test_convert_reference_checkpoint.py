import os
from collections import OrderedDict

import torch

from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.convert_deepseek_v32_reference_checkpoint import (
    convert_reference_shards_to_dense,
)
from transformers.models.deepseek_v32.modular_deepseek_v32 import DeepseekV32ForCausalLM
from transformers.testing_utils import require_torch


def _get_tiny_config():
    return DeepseekV32Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=16,
        first_k_dense_replace=1,
    )


def _split_tensor_along_dim(tensor: torch.Tensor, world_size: int, dim: int) -> list[torch.Tensor]:
    chunks = tensor.chunk(world_size, dim=dim)
    return [chunk.contiguous() for chunk in chunks]


def _shard_state_dict(state_dict: OrderedDict[str, torch.Tensor], world_size: int) -> list[OrderedDict[str, torch.Tensor]]:
    """
    Creates fake tensor-parallel shards by splitting parameters along dim-0 (column) or dim-1 (row).

    This mimics the DeepSeek reference sharding strategy enough to exercise the converter logic.
    """

    column_suffixes = (
        "embed_tokens.weight",
        "lm_head.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_b_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
    )
    row_suffixes = (
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
        "shared_experts.down_proj.weight",
    )

    shards = [OrderedDict() for _ in range(world_size)]

    for key, tensor in state_dict.items():
        if any(key.endswith(suffix) for suffix in column_suffixes) and tensor.shape[0] % world_size == 0:
            splits = _split_tensor_along_dim(tensor, world_size, dim=0)
            for rank, chunk in enumerate(splits):
                shards[rank][key] = chunk.clone()
        elif (
            tensor.dim() >= 2
            and any(key.endswith(suffix) for suffix in row_suffixes)
            and tensor.shape[1] % world_size == 0
        ):
            splits = _split_tensor_along_dim(tensor, world_size, dim=1)
            for rank, chunk in enumerate(splits):
                shards[rank][key] = chunk.clone()
        elif tensor.dim() >= 1 and tensor.shape[0] % world_size == 0:
            splits = _split_tensor_along_dim(tensor, world_size, dim=0)
            for rank, chunk in enumerate(splits):
                shards[rank][key] = chunk.clone()
        else:
            for rank in range(world_size):
                shards[rank][key] = tensor.clone()

    return shards


@require_torch
def test_convert_reference_shards_roundtrip(tmp_path):
    config = _get_tiny_config()
    model = DeepseekV32ForCausalLM(config)
    state_dict = model.state_dict()

    world_size = 2
    shards = _shard_state_dict(state_dict, world_size)
    shard_paths = []
    for rank, shard in enumerate(shards):
        shard_path = os.path.join(tmp_path, f"rank{rank}.pt")
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)

    merged_state_dict = convert_reference_shards_to_dense(
        shard_paths=shard_paths,
        config=config,
        output_path=None,
        dtype=None,
    )

    assert set(merged_state_dict.keys()) == set(state_dict.keys())
    for key, original_tensor in state_dict.items():
        merged_tensor = merged_state_dict[key]
        assert merged_tensor.shape == original_tensor.shape
        assert torch.allclose(merged_tensor, original_tensor, atol=0, rtol=0)


@require_torch
def test_convert_reference_shards_dtype_override(tmp_path):
    config = _get_tiny_config()
    model = DeepseekV32ForCausalLM(config)
    state_dict = model.state_dict()

    shards = _shard_state_dict(state_dict, world_size=2)
    shard_paths = []
    for rank, shard in enumerate(shards):
        shard_path = os.path.join(tmp_path, f"rank{rank}.pt")
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)

    merged_state_dict = convert_reference_shards_to_dense(
        shard_paths=shard_paths,
        config=config,
        output_path=None,
        dtype=torch.float32,
    )

    for tensor in merged_state_dict.values():
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float32

