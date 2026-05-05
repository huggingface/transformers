# Copyright 2025 HuggingFace Inc.
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
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

from transformers import PretrainedConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    DtensorShardOperation,
    ErnieFuseAndSplitTextVisionExperts,
    MergeModulelist,
    PermuteForRope,
    WeightConverter,
    WeightRenaming,
    build_glob_alternation,
    convert_and_load_state_dict_in_model,
    rename_source_key,
    revert_weight_conversion,
    spawn_materialize,
)
from transformers.modeling_utils import LoadStateDictConfig
from transformers.utils.import_utils import is_triton_available

from ..test_modeling_common import compare_state_dicts


class TestWeightGlobMatching(unittest.TestCase):
    def setUp(self):
        self.weight_globs_digits = [
            "model.layers.*.mlp.gate_up_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "embed_tokens.weight",
        ]
        self.alt_digits, self.map_digits, _ = build_glob_alternation(self.weight_globs_digits)

        self.weight_globs_any = [
            "model.layers.*.mlp.gate_up_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "embed_tokens.weight",
        ]
        self.alt_any, self.map_any, _ = build_glob_alternation(self.weight_globs_any)

    @staticmethod
    def _match_glob(key, alt, mapping):
        matched = alt.search(key)
        return mapping.get(matched.lastgroup) if matched else None

    def test_exact_match(self):
        self.assertEqual(
            self._match_glob("embed_tokens.weight", self.alt_digits, self.map_digits), "embed_tokens.weight"
        )

    def test_digits_only_star_accepts_digits(self):
        self.assertEqual(
            self._match_glob("model.layers.0.mlp.gate_up_proj.weight", self.alt_digits, self.map_digits),
            "model.layers.*.mlp.gate_up_proj.weight",
        )
        self.assertEqual(
            self._match_glob("model.layers.12.self_attn.q_proj.weight", self.alt_digits, self.map_digits),
            "model.layers.*.self_attn.q_proj.weight",
        )

    def test_anychar_star_accepts_nondigits(self):
        self.assertEqual(
            self._match_glob("model.layers.a.mlp.gate_up_proj.weight", self.alt_any, self.map_any),
            "model.layers.*.mlp.gate_up_proj.weight",
        )
        self.assertEqual(
            self._match_glob("model.layers.00x.mlp.gate_up_proj.weight", self.alt_any, self.map_any),
            "model.layers.*.mlp.gate_up_proj.weight",
        )

    def test_no_match(self):
        self.assertIsNone(self._match_glob("model.layers.0.mlp.up_proj.weight", self.alt_digits, self.map_digits))

    def test_leftmost_alternative_wins_for_overlapping_patterns(self):
        # Overlapping patterns: both could match; ensure leftmost wins
        globs = [
            "model.layers.*.mlp.*.weight",  # broader (first)
            "model.layers.0.mlp.gate_up_proj.weight",  # more specific (second)
        ]
        alt, mapping, _ = build_glob_alternation(globs)

        # Both branches match; Python's regex picks the leftmost alternative → index 0
        self.assertEqual(
            self._match_glob("model.layers.0.mlp.gate_up_proj.weight", alt, mapping), "model.layers.*.mlp.*.weight"
        )

    def test_multiple_patterns_same_prefix(self):
        globs = [
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
        ]
        alt, mapping, _ = build_glob_alternation(
            globs,
        )

        self.assertEqual(
            self._match_glob("model.layers.3.self_attn.q_proj.weight", alt, mapping),
            "model.layers.*.self_attn.q_proj.weight",
        )
        self.assertEqual(
            self._match_glob("model.layers.3.self_attn.k_proj.weight", alt, mapping),
            "model.layers.*.self_attn.k_proj.weight",
        )
        self.assertEqual(
            self._match_glob("model.layers.3.self_attn.v_proj.weight", alt, mapping),
            "model.layers.*.self_attn.v_proj.weight",
        )

    def test_anchor_full_match_only(self):
        self.assertIsNotNone(
            self._match_glob("model.layers.0.mlp.gate_up_proj.weight.bar", self.alt_any, self.map_any)
        )

    def test_large_batch_performance_smoke(self):
        # Not a perf benchmark, but ensures building and matching a larger alternation is OK
        globs = [f"model.layers.*.mlp.block{i}.weight" for i in range(200)]
        alt, mapping, _ = build_glob_alternation(globs)
        key = "model.layers.123.mlp.block57.weight"
        self.assertEqual(self._match_glob(key, alt, mapping), "model.layers.*.mlp.block57.weight")

    def test_sub_key_rewrites_targets(self):
        renamings = [
            WeightRenaming("block_sparse_moe.experts.*.w1.weight", "mlp.experts.gate_up_proj"),
            WeightRenaming("block_sparse_moe.experts.*.w2.weight", "mlp.experts.down_proj"),
            WeightRenaming("model.language_model.*", "language_model"),
        ]

        self.assertEqual(
            rename_source_key("foo.block_sparse_moe.experts.3.w1.weight", renamings, [])[0],
            "foo.mlp.experts.gate_up_proj",
        )
        self.assertEqual(
            rename_source_key("foo.block_sparse_moe.experts.3.w2.weight", renamings, [])[0],
            "foo.mlp.experts.down_proj",
        )
        self.assertEqual(rename_source_key("model.language_model.lm_head.weight", renamings, [])[0], "language_model")

    def test_sub_key_no_match_returns_original(self):
        renamings = [
            WeightRenaming("block_sparse_moe.experts.*.w1.weight", "*.mlp.experts.gate_up_proj"),
        ]

        key = "unrelated.key"
        renamed_key, _ = rename_source_key(key, renamings, [])
        self.assertEqual(renamed_key, key)


class DummyParamModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))


class DummySelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = DummyParamModule((1, 2))
        self.k_proj = DummyParamModule((1, 2))
        self.v_proj = DummyParamModule((1, 2))


class DummyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = DummyParamModule((2, 4, 2))
        self.down_proj = DummyParamModule((2, 2, 2))


class DummyLayer(nn.Module):
    def __init__(self, add_extra_moe=False):
        super().__init__()
        self.self_attn = DummySelfAttn()
        self.experts = DummyExperts()
        if add_extra_moe:
            self.extra_experts = DummyExperts()


class DummyTopModel(nn.Module):
    def __init__(self, add_extra_moe=False):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(add_extra_moe), DummyLayer(add_extra_moe)])


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = DummyParamModule((2, 2))


class DummyRoot(nn.Module):
    base_model_prefix = "model"
    config: PretrainedConfig

    def __init__(self, add_extra_moe=False):
        super().__init__()
        self.model = DummyTopModel(add_extra_moe)
        self.mlp = DummyMLP()


class FakeMesh:
    """Fake multi-dimensional device mesh for testing DtensorShardOperation."""

    def __init__(self, shape, rank, dim_names=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.mesh_dim_names = dim_names or tuple(f"dim{i}" for i in range(self.ndim))
        # Compute nD coordinate (row-major: last dim changes fastest)
        self._coord = []
        r = rank
        for s in reversed(self.shape):
            self._coord.insert(0, r % s)
            r //= s

    def get_local_rank(self):
        return self._coord[0]

    def get_coordinate(self):
        return tuple(self._coord)

    def size(self):
        result = 1
        for s in self.shape:
            result *= s
        return result

    def _is_current_rank_part_of_mesh(self):
        return True

    def _sym_get_coordinate(self, dim):
        return self._coord[dim]

    def __getitem__(self, name):
        idx = self.mesh_dim_names.index(name)
        return FakeMesh(
            shape=(self.shape[idx],),
            rank=self._coord[idx],
            dim_names=(name,),
        )


def _make_dtensor_shard_op(mesh, placements, param_shape, local_shape):
    """Build a DtensorShardOperation without requiring a real DTensor / distributed init.

    The axis-0 ownership cache is computed by mimicking
    ``compute_local_shape_and_global_offset`` for the leading dim only:
    locate the mesh dim that shards param dim 0 (if any) and use its local rank.
    """
    op = object.__new__(DtensorShardOperation)
    op.device_mesh = mesh
    op.placements = tuple(placements)
    op.param_ndim = len(param_shape)
    op._axis0_offset = 0
    op._axis0_local_size = local_shape[0]
    for mesh_dim, p in enumerate(placements):
        if hasattr(p, "dim") and (p.dim % len(param_shape)) == 0:
            sub = mesh[mesh.mesh_dim_names[mesh_dim]] if mesh.ndim > 1 else mesh
            op._axis0_offset = sub.get_local_rank() * local_shape[0]
            break
    return op


class TestConvertAndLoadStateDict(unittest.TestCase):
    def test_dtensor_shard_aware_mixtral_conversion_uses_only_local_experts(self):
        """Integration test: FSDP-sharded expert loading + WeightConverter.

        The problem: Mixtral has 8 experts. The checkpoint stores them separately::

            experts.0.w1.weight  (2x2)
            experts.0.w3.weight  (2x2)
            experts.1.w1.weight  (2x2)
            experts.1.w3.weight  (2x2)

        The model stores them packed into one tensor::

            experts.gate_up_proj.weight  (2, 4, 2)
                                          ^  ^  ^
                                          |  |  +-- features
                                          |  +-- w1 (2) + w3 (2) concatenated
                                          +-- num_experts

        The conversion (without FSDP) is: load all expert w1/w3 tensors,
        MergeModulelist(dim=0) stacks experts, Concatenate(dim=1) joins w1+w3.

        With FSDP, Shard(0) splits the expert dim across ranks. Rank 0 owns
        expert 0, rank 1 owns expert 1. So rank 0 should skip loading expert 1
        entirely -- not load it then discard it.

        What the test checks::

            checkpoint files              shard_tensor              rank 0 gets
            ----------------              ------------              -----------
            experts.0.w1  [[0,1],[2,3]]   idx=0 -> kept            [[0,1],[2,3]]
            experts.1.w1  [[10,11],...]   idx=1 -> None (not owned)
            experts.0.w3  [[4,5],[6,7]]   idx=0 -> kept            [[4,5],[6,7]]
            experts.1.w3  [[14,15],...]   idx=1 -> None (not owned)

        WeightConverter then combines only the kept tensors::

            MergeModulelist(dim=0): stack owned experts  -> shape (1, 2, 2) each
            Concatenate(dim=1):     cat w1 + w3 along dim 1

            gate_up_proj = [[[0,1],[2,3],[4,5],[6,7]]]   shape (1, 4, 2)
                              ~~~~~~~~~~  ~~~~~~~~~~
                                  w1          w3

        The key point: DtensorShardOperation.shard_tensor(tensor_idx=1) returns
        None for rank 0, so the converter never even processes expert 1's data.
        This saves memory during loading.
        """
        shard_op = _make_dtensor_shard_op(
            FakeMesh(shape=(2,), rank=0),
            [Shard(0)],
            param_shape=(2, 4, 2),
            local_shape=(1, 4, 2),
        )
        converter = WeightConverter(
            ["experts.*.w1.weight", "experts.*.w3.weight"],
            "experts.gate_up_proj.weight",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        )

        for idx, tensor in enumerate(
            [
                torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
                torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            ]
        ):
            converter.add_tensor(
                "model.layers.0.experts.gate_up_proj.weight",
                f"model.layers.0.experts.{idx}.w1.weight",
                "experts.*.w1.weight",
                spawn_materialize(None, tensor, device="cpu", dtype=None, sharding_op=shard_op, tensor_idx=idx),
            )

        for idx, tensor in enumerate(
            [
                torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
                torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
            ]
        ):
            converter.add_tensor(
                "model.layers.0.experts.gate_up_proj.weight",
                f"model.layers.0.experts.{idx}.w3.weight",
                "experts.*.w3.weight",
                spawn_materialize(None, tensor, device="cpu", dtype=None, sharding_op=shard_op, tensor_idx=idx),
            )

        converted = converter.convert("model.layers.0.experts.gate_up_proj.weight")

        self.assertEqual(list(converted), ["model.layers.0.experts.gate_up_proj.weight"])
        torch.testing.assert_close(
            converted["model.layers.0.experts.gate_up_proj.weight"],
            torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]]),
        )

    def test_moe_and_qkv_conversion(self):
        model = DummyRoot()
        model.config = PretrainedConfig()

        raw_tensors = {
            "model.layers.0.experts.0.w1.weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "model.layers.0.experts.1.w1.weight": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            "model.layers.0.experts.0.w3.weight": torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            "model.layers.0.experts.1.w3.weight": torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
            "model.layers.0.experts.0.w2.weight": torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
            "model.layers.0.experts.1.w2.weight": torch.tensor([[24.0, 25.0], [26.0, 27.0]]),
            "model.layers.1.experts.0.w1.weight": torch.tensor([[30.0, 31.0], [32.0, 33.0]]),
            "model.layers.1.experts.1.w1.weight": torch.tensor([[34.0, 35.0], [36.0, 37.0]]),
            "model.layers.1.experts.0.w3.weight": torch.tensor([[38.0, 39.0], [40.0, 41.0]]),
            "model.layers.1.experts.1.w3.weight": torch.tensor([[42.0, 43.0], [44.0, 45.0]]),
            "model.layers.1.experts.0.w2.weight": torch.tensor([[46.0, 47.0], [48.0, 49.0]]),
            "model.layers.1.experts.1.w2.weight": torch.tensor([[50.0, 51.0], [52.0, 53.0]]),
            "model.layers.0.self_attn.qkv_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "model.layers.1.self_attn.qkv_proj.weight": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "mlp.w2.weight": torch.tensor([[60.0, 61.0], [62.0, 63.0]]),
        }
        state_dict = {k: v.clone() for k, v in raw_tensors.items()}

        weight_mapping = [
            WeightConverter(
                ["experts.*.w1.weight", "experts.*.w3.weight"],
                "experts.gate_up_proj.weight",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                "experts.*.w2.weight",
                "experts.down_proj.weight",
                operations=[MergeModulelist(dim=0)],
            ),
            WeightConverter(
                "model.layers.0.self_attn.qkv_proj.weight",
                [
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming("mlp.w2.weight", "mlp.down_proj.weight"),
        ]

        load_config = LoadStateDictConfig(
            weight_mapping=weight_mapping,
        )
        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            state_dict,
            load_config,
            tp_plan=None,
        )

        self.assertEqual(
            loading_info.missing_keys,
            {
                "model.layers.1.self_attn.k_proj.weight",
                "model.layers.1.self_attn.v_proj.weight",
                "model.layers.1.self_attn.q_proj.weight",
            },
        )
        self.assertEqual(loading_info.unexpected_keys, {"model.layers.1.self_attn.qkv_proj.weight"})
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        model_state = model.state_dict()

        def cat_gate(layer_prefix: str) -> torch.Tensor:
            w1 = [
                raw_tensors[f"{layer_prefix}.experts.0.w1.weight"],
                raw_tensors[f"{layer_prefix}.experts.1.w1.weight"],
            ]
            w3 = [
                raw_tensors[f"{layer_prefix}.experts.0.w3.weight"],
                raw_tensors[f"{layer_prefix}.experts.1.w3.weight"],
            ]
            return torch.cat([torch.stack(w1, dim=0), torch.stack(w3, dim=0)], dim=1)

        torch.testing.assert_close(
            model_state["model.layers.0.experts.gate_up_proj.weight"], cat_gate("model.layers.0")
        )
        torch.testing.assert_close(
            model_state["model.layers.1.experts.gate_up_proj.weight"], cat_gate("model.layers.1")
        )

        def stack_down(layer_prefix: str) -> torch.Tensor:
            return torch.stack(
                [
                    raw_tensors[f"{layer_prefix}.experts.0.w2.weight"],
                    raw_tensors[f"{layer_prefix}.experts.1.w2.weight"],
                ],
                dim=0,
            )

        torch.testing.assert_close(
            model_state["model.layers.0.experts.down_proj.weight"], stack_down("model.layers.0")
        )
        torch.testing.assert_close(
            model_state["model.layers.1.experts.down_proj.weight"], stack_down("model.layers.1")
        )

        for layer_idx in range(2):
            key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            expected_q, expected_k, expected_v = torch.chunk(raw_tensors[key], chunks=3, dim=0)
            prefix = f"model.layers.{layer_idx}.self_attn"
            if layer_idx == 1:
                # These were missing and thus not loaded
                continue
            torch.testing.assert_close(model_state[f"{prefix}.q_proj.weight"], expected_q)
            torch.testing.assert_close(model_state[f"{prefix}.k_proj.weight"], expected_k)
            torch.testing.assert_close(model_state[f"{prefix}.v_proj.weight"], expected_v)

        torch.testing.assert_close(model_state["mlp.down_proj.weight"], raw_tensors["mlp.w2.weight"])

    def test_moe_and_qkv_conversion_reversed(self):
        model = DummyRoot()
        model.config = PretrainedConfig()

        raw_tensors = {
            "model.layers.0.experts.0.w1.weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "model.layers.0.experts.1.w1.weight": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            "model.layers.0.experts.0.w3.weight": torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            "model.layers.0.experts.1.w3.weight": torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
            "model.layers.0.experts.0.w2.weight": torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
            "model.layers.0.experts.1.w2.weight": torch.tensor([[24.0, 25.0], [26.0, 27.0]]),
            "model.layers.1.experts.0.w1.weight": torch.tensor([[30.0, 31.0], [32.0, 33.0]]),
            "model.layers.1.experts.1.w1.weight": torch.tensor([[34.0, 35.0], [36.0, 37.0]]),
            "model.layers.1.experts.0.w3.weight": torch.tensor([[38.0, 39.0], [40.0, 41.0]]),
            "model.layers.1.experts.1.w3.weight": torch.tensor([[42.0, 43.0], [44.0, 45.0]]),
            "model.layers.1.experts.0.w2.weight": torch.tensor([[46.0, 47.0], [48.0, 49.0]]),
            "model.layers.1.experts.1.w2.weight": torch.tensor([[50.0, 51.0], [52.0, 53.0]]),
            "model.layers.0.self_attn.qkv_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "model.layers.1.self_attn.qkv_proj.weight": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "mlp.w2.weight": torch.tensor([[60.0, 61.0], [62.0, 63.0]]),
        }
        state_dict = {k: v.clone() for k, v in raw_tensors.items()}

        weight_mapping = [
            WeightConverter(
                ["experts.*.w1.weight", "experts.*.w3.weight"],
                "experts.gate_up_proj.weight",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                "experts.*.w2.weight",
                "experts.down_proj.weight",
                operations=[MergeModulelist(dim=0)],
            ),
            WeightConverter(
                "self_attn.qkv_proj.weight",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming("mlp.w2.weight", "mlp.down_proj.weight"),
        ]

        # Use the mapping to load
        load_config = LoadStateDictConfig(
            weight_mapping=weight_mapping,
        )
        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            state_dict,
            load_config,
            tp_plan=None,
        )
        self.assertTrue(len(loading_info.missing_keys) == 0)
        self.assertTrue(len(loading_info.unexpected_keys) == 0)
        self.assertTrue(len(loading_info.mismatched_keys) == 0)
        self.assertTrue(len(loading_info.conversion_errors) == 0)

        # Try to revert the mapping
        reversed_state_dict = revert_weight_conversion(model, model.state_dict())

        # Make sure both saved state_dict are identical
        self.assertTrue(compare_state_dicts(reversed_state_dict, state_dict))

    def test_qkv_chunk_rope_permute_with_fp8_quantization(self):
        if is_triton_available():
            from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        else:
            self.skipTest("Fine-grained FP8 integration tests require Triton to be installed.")
        n_heads = 2
        head_dim = 4
        in_dim = 4
        out_dim = n_heads * head_dim
        block_size = (4, 4)

        class RopeProjector(nn.Module):
            def __init__(self, *, with_scale: bool = False):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
                if with_scale:
                    scale_shape = (out_dim // block_size[0], in_dim // block_size[1])
                    self.weight_scale_inv = nn.Parameter(torch.ones(scale_shape))

        class RopeSelfAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = RopeProjector(with_scale=True)
                self.k_proj = RopeProjector()
                self.v_proj = RopeProjector()

        class RopeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = RopeSelfAttn()

        class RopeModel(nn.Module):
            base_model_prefix = "model"

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([RopeLayer()])

        model = RopeModel()
        model.config = PretrainedConfig()
        model.config.num_attention_heads = n_heads

        raw_q = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0],
                [0.5, -0.5, 0.5, -0.5],
                [-1.0, 1.0, -1.0, 1.0],
                [-0.5, 0.5, -0.5, 0.5],
                [1.0, 1.0, -1.0, -1.0],
                [0.5, 0.5, -0.5, -0.5],
                [-1.0, -1.0, 1.0, 1.0],
                [-0.5, -0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        raw_k = torch.arange(out_dim * in_dim, dtype=torch.float32).reshape(out_dim, in_dim)
        raw_v = torch.arange(out_dim * in_dim, dtype=torch.float32).reshape(out_dim, in_dim) + 100.0
        raw_qkv = torch.cat([raw_q, raw_k, raw_v], dim=0)
        state_dict = {"model.layers.0.self_attn.qkv_proj.weight": raw_qkv.clone()}

        quantizer_cls = type(
            "FineGrainedFP8HfQuantizer",
            (),
            {
                "__init__": lambda self, bs=block_size: setattr(
                    self, "quantization_config", SimpleNamespace(weight_block_size=bs)
                ),
                "param_needs_quantization": lambda self, _model, param_name: param_name.endswith("q_proj.weight"),
                "get_quantize_ops": lambda self: __import__(
                    "transformers.integrations.finegrained_fp8",
                    fromlist=["Fp8Quantize"],
                ).Fp8Quantize(self),
                "pre_quantized": False,
            },
        )
        quantizer = quantizer_cls()

        weight_mapping = [
            WeightConverter(
                "model.layers.*.self_attn.qkv_proj.weight",
                [
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0), PermuteForRope()],
            )
        ]
        load_config = LoadStateDictConfig(weight_mapping=weight_mapping, hf_quantizer=quantizer)
        loading_info, _ = convert_and_load_state_dict_in_model(model, state_dict, load_config, tp_plan=None)

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        permute_op = PermuteForRope()
        permute_op.config = model.config
        expected_q = permute_op._apply(raw_q)
        expected_k = permute_op._apply(raw_k)
        expected_v = permute_op._apply(raw_v)

        model_state = model.state_dict()
        self.assertFalse(torch.allclose(raw_k, expected_k))
        torch.testing.assert_close(model_state["layers.0.self_attn.k_proj.weight"], expected_k)
        torch.testing.assert_close(model_state["layers.0.self_attn.v_proj.weight"], expected_v)

        q_weight_key = "layers.0.self_attn.q_proj.weight"
        scale_key = "layers.0.self_attn.q_proj.weight_scale_inv"
        self.assertIn(scale_key, model_state)
        expected_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.int8
        self.assertEqual(model_state[q_weight_key].dtype, expected_dtype)
        self.assertEqual(model_state[q_weight_key].shape, torch.Size((out_dim, in_dim)))
        self.assertEqual(model_state[scale_key].dtype, torch.float32)
        self.assertEqual(
            model_state[scale_key].shape,
            torch.Size((out_dim // block_size[0], in_dim // block_size[1])),
        )

        dequant = Fp8Dequantize(quantizer)
        dequantized_q = dequant.convert(
            {
                "weight$": [model_state[q_weight_key]],
                "weight_scale_inv": [model_state[scale_key]],
            },
            full_layer_name=q_weight_key,
        )[q_weight_key]
        torch.testing.assert_close(dequantized_q, expected_q, rtol=1e-2, atol=1e-2)

    def test_ernie4_5_vl_moe_conversion(self):
        model = DummyRoot(add_extra_moe=True)
        model.config = PretrainedConfig()

        raw_tensors = {
            "model.layers.0.experts.0.w1.weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "model.layers.0.experts.1.w1.weight": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            "model.layers.0.experts.2.w1.weight": torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
            "model.layers.0.experts.3.w1.weight": torch.tensor([[12.0, 13.0], [14.0, 15.0]]),
            "model.layers.0.experts.0.w3.weight": torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            "model.layers.0.experts.1.w3.weight": torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
            "model.layers.0.experts.2.w3.weight": torch.tensor([[15.0, 16.0], [17.0, 18.0]]),
            "model.layers.0.experts.3.w3.weight": torch.tensor([[16.0, 17.0], [18.0, 19.0]]),
            "model.layers.0.experts.0.w2.weight": torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
            "model.layers.0.experts.1.w2.weight": torch.tensor([[24.0, 25.0], [26.0, 27.0]]),
            "model.layers.0.experts.2.w2.weight": torch.tensor([[25.0, 26.0], [27.0, 28.0]]),
            "model.layers.0.experts.3.w2.weight": torch.tensor([[26.0, 27.0], [28.0, 29.0]]),
            "model.layers.1.experts.0.w1.weight": torch.tensor([[30.0, 31.0], [32.0, 33.0]]),
            "model.layers.1.experts.1.w1.weight": torch.tensor([[34.0, 35.0], [36.0, 37.0]]),
            "model.layers.1.experts.2.w1.weight": torch.tensor([[35.0, 36.0], [37.0, 38.0]]),
            "model.layers.1.experts.3.w1.weight": torch.tensor([[36.0, 37.0], [38.0, 39.0]]),
            "model.layers.1.experts.0.w3.weight": torch.tensor([[38.0, 39.0], [40.0, 41.0]]),
            "model.layers.1.experts.1.w3.weight": torch.tensor([[42.0, 43.0], [44.0, 45.0]]),
            "model.layers.1.experts.2.w3.weight": torch.tensor([[43.0, 44.0], [45.0, 46.0]]),
            "model.layers.1.experts.3.w3.weight": torch.tensor([[44.0, 45.0], [46.0, 47.0]]),
            "model.layers.1.experts.0.w2.weight": torch.tensor([[46.0, 47.0], [48.0, 49.0]]),
            "model.layers.1.experts.1.w2.weight": torch.tensor([[50.0, 51.0], [52.0, 53.0]]),
            "model.layers.1.experts.2.w2.weight": torch.tensor([[51.0, 52.0], [53.0, 54.0]]),
            "model.layers.1.experts.3.w2.weight": torch.tensor([[52.0, 53.0], [54.0, 55.0]]),
            "model.layers.0.self_attn.qkv_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "model.layers.1.self_attn.qkv_proj.weight": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "mlp.w2.weight": torch.tensor([[60.0, 61.0], [62.0, 63.0]]),
        }
        state_dict = {k: v.clone() for k, v in raw_tensors.items()}

        weight_mapping = [
            WeightConverter(
                ["experts.*.w1.weight", "experts.*.w3.weight"],
                ["experts.gate_up_proj.weight", "extra_experts.gate_up_proj.weight"],
                operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
            ),
            WeightConverter(
                "experts.*.w2.weight",
                ["experts.down_proj.weight", "extra_experts.down_proj.weight"],
                operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
            ),
            WeightConverter(
                "self_attn.qkv_proj.weight",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming("mlp.w2.weight", "mlp.down_proj.weight"),
        ]
        loading_info, _ = convert_and_load_state_dict_in_model(
            model, state_dict, LoadStateDictConfig(weight_mapping=weight_mapping), tp_plan=None
        )

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        model_state = model.state_dict()

        def cat_gate(layer_prefix: str) -> torch.Tensor:
            moe_1_w1 = [
                raw_tensors[f"{layer_prefix}.experts.0.w1.weight"],
                raw_tensors[f"{layer_prefix}.experts.1.w1.weight"],
            ]
            moe_2_w1 = [
                raw_tensors[f"{layer_prefix}.experts.2.w1.weight"],
                raw_tensors[f"{layer_prefix}.experts.3.w1.weight"],
            ]
            moe_1_w3 = [
                raw_tensors[f"{layer_prefix}.experts.0.w3.weight"],
                raw_tensors[f"{layer_prefix}.experts.1.w3.weight"],
            ]
            moe_2_w3 = [
                raw_tensors[f"{layer_prefix}.experts.2.w3.weight"],
                raw_tensors[f"{layer_prefix}.experts.3.w3.weight"],
            ]
            moe_1 = torch.cat([torch.stack(moe_1_w1, dim=0), torch.stack(moe_1_w3, dim=0)], dim=1)
            moe_2 = torch.cat([torch.stack(moe_2_w1, dim=0), torch.stack(moe_2_w3, dim=0)], dim=1)
            return moe_1, moe_2

        moe_1, moe_2 = cat_gate("model.layers.0")
        torch.testing.assert_close(model_state["model.layers.0.experts.gate_up_proj.weight"], moe_1)
        torch.testing.assert_close(model_state["model.layers.0.extra_experts.gate_up_proj.weight"], moe_2)

        moe_1, moe_2 = cat_gate("model.layers.1")
        torch.testing.assert_close(model_state["model.layers.1.experts.gate_up_proj.weight"], moe_1)
        torch.testing.assert_close(model_state["model.layers.1.extra_experts.gate_up_proj.weight"], moe_2)

        def stack_down(layer_prefix: str) -> torch.Tensor:
            moe_1 = torch.stack(
                [
                    raw_tensors[f"{layer_prefix}.experts.0.w2.weight"],
                    raw_tensors[f"{layer_prefix}.experts.1.w2.weight"],
                ],
                dim=0,
            )
            moe_2 = torch.stack(
                [
                    raw_tensors[f"{layer_prefix}.experts.2.w2.weight"],
                    raw_tensors[f"{layer_prefix}.experts.3.w2.weight"],
                ],
                dim=0,
            )
            return moe_1, moe_2

        moe_1, moe_2 = stack_down("model.layers.0")
        torch.testing.assert_close(model_state["model.layers.0.experts.down_proj.weight"], moe_1)
        torch.testing.assert_close(model_state["model.layers.0.extra_experts.down_proj.weight"], moe_2)

        moe_1, moe_2 = stack_down("model.layers.1")
        torch.testing.assert_close(model_state["model.layers.1.experts.down_proj.weight"], moe_1)
        torch.testing.assert_close(model_state["model.layers.1.extra_experts.down_proj.weight"], moe_2)

    def test_ernie4_5_vl_moe_conversion_reversed(self):
        model = DummyRoot(add_extra_moe=True)
        model.config = PretrainedConfig()

        raw_tensors = {
            "model.layers.0.experts.0.w1.weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "model.layers.0.experts.1.w1.weight": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
            "model.layers.0.experts.2.w1.weight": torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
            "model.layers.0.experts.3.w1.weight": torch.tensor([[12.0, 13.0], [14.0, 15.0]]),
            "model.layers.0.experts.0.w3.weight": torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            "model.layers.0.experts.1.w3.weight": torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
            "model.layers.0.experts.2.w3.weight": torch.tensor([[15.0, 16.0], [17.0, 18.0]]),
            "model.layers.0.experts.3.w3.weight": torch.tensor([[16.0, 17.0], [18.0, 19.0]]),
            "model.layers.0.experts.0.w2.weight": torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
            "model.layers.0.experts.1.w2.weight": torch.tensor([[24.0, 25.0], [26.0, 27.0]]),
            "model.layers.0.experts.2.w2.weight": torch.tensor([[25.0, 26.0], [27.0, 28.0]]),
            "model.layers.0.experts.3.w2.weight": torch.tensor([[26.0, 27.0], [28.0, 29.0]]),
            "model.layers.1.experts.0.w1.weight": torch.tensor([[30.0, 31.0], [32.0, 33.0]]),
            "model.layers.1.experts.1.w1.weight": torch.tensor([[34.0, 35.0], [36.0, 37.0]]),
            "model.layers.1.experts.2.w1.weight": torch.tensor([[35.0, 36.0], [37.0, 38.0]]),
            "model.layers.1.experts.3.w1.weight": torch.tensor([[36.0, 37.0], [38.0, 39.0]]),
            "model.layers.1.experts.0.w3.weight": torch.tensor([[38.0, 39.0], [40.0, 41.0]]),
            "model.layers.1.experts.1.w3.weight": torch.tensor([[42.0, 43.0], [44.0, 45.0]]),
            "model.layers.1.experts.2.w3.weight": torch.tensor([[43.0, 44.0], [45.0, 46.0]]),
            "model.layers.1.experts.3.w3.weight": torch.tensor([[44.0, 45.0], [46.0, 47.0]]),
            "model.layers.1.experts.0.w2.weight": torch.tensor([[46.0, 47.0], [48.0, 49.0]]),
            "model.layers.1.experts.1.w2.weight": torch.tensor([[50.0, 51.0], [52.0, 53.0]]),
            "model.layers.1.experts.2.w2.weight": torch.tensor([[51.0, 52.0], [53.0, 54.0]]),
            "model.layers.1.experts.3.w2.weight": torch.tensor([[52.0, 53.0], [54.0, 55.0]]),
            "model.layers.0.self_attn.qkv_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "model.layers.1.self_attn.qkv_proj.weight": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "mlp.w2.weight": torch.tensor([[60.0, 61.0], [62.0, 63.0]]),
        }
        state_dict = {k: v.clone() for k, v in raw_tensors.items()}

        weight_mapping = [
            WeightConverter(
                ["experts.*.w1.weight", "experts.*.w3.weight"],
                ["experts.gate_up_proj.weight", "extra_experts.gate_up_proj.weight"],
                operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
            ),
            WeightConverter(
                "experts.*.w2.weight",
                ["experts.down_proj.weight", "extra_experts.down_proj.weight"],
                operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
            ),
            WeightConverter(
                "self_attn.qkv_proj.weight",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming("mlp.w2.weight", "mlp.down_proj.weight"),
        ]

        # Use the mapping to load
        loading_info, _ = convert_and_load_state_dict_in_model(
            model, state_dict, LoadStateDictConfig(weight_mapping=weight_mapping), tp_plan=None
        )
        self.assertTrue(len(loading_info.missing_keys) == 0)
        self.assertTrue(len(loading_info.unexpected_keys) == 0)
        self.assertTrue(len(loading_info.mismatched_keys) == 0)
        self.assertTrue(len(loading_info.conversion_errors) == 0)

        # Try to revert the mapping
        reversed_state_dict = revert_weight_conversion(model, model.state_dict())

        # Make sure both saved state_dict are identical
        self.assertTrue(compare_state_dicts(reversed_state_dict, state_dict))


class TestDtensorShardOperation(unittest.TestCase):
    """Unit tests for DtensorShardOperation.

    The checkpoint can store the parameter in two layouts. Take a stack
    of N MoE experts of shape [in, out] as a running example — the
    param shape is [N, in, out]:

    - Single tensor (tensor_idx is None): the checkpoint holds one
      [N, in, out] tensor, so source.shape == param.shape. Every
      sharded dim is sliced here, including axis 0.

    - One tensor per piece (tensor_idx given): the checkpoint holds N
      separate [in, out] tensors, one per expert. shard_tensor is
      called once per expert; on each call source is the [in, out]
      tensor for expert number tensor_idx (so 0 <= tensor_idx < N).
      Note: source has one fewer dim than the param: the axis-0
      index lives in tensor_idx, not in source.shape.
      If this rank does not own tensor_idx along axis 0, return None
      and the piece is discarded. Otherwise slice only the inner
      dims; the caller (MergeModulelist / Concatenate) collects the
      kept pieces and stacks them back along axis 0 to rebuild the
      full [N, in, out] param.
    """

    def test_no_shard_placements_returns_full_copy(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor,  # rank 0 — no shards, full copy
            1: tensor,  # rank 1 — no shards, full copy
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Replicate()], param_shape=(4, 4), local_shape=(4, 4))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_1D_shard(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor[:2],  # rank 0 — first half
            1: tensor[2:],  # rank 1 — second half
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(4, 4), local_shape=(2, 4))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_1D_strided_shard(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor[[0, 2]],  # first piece of each group — rows {0, 2}
            1: tensor[[1, 3]],  # second piece of each group — rows {1, 3}
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(
                mesh, [_StridedShard(dim=0, split_factor=2)], param_shape=(4, 4), local_shape=(2, 4)
            )
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_shard_different_dims(self):
        tensor = torch.arange(64).reshape(8, 8).float()
        expected = {
            0: tensor[:4, :4],  # top-left
            1: tensor[:4, 4:],  # top-right
            2: tensor[4:, :4],  # bottom-left
            3: tensor[4:, 4:],  # bottom-right
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0), Shard(1)], param_shape=(8, 8), local_shape=(4, 4))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_shard_same_dim(self):
        tensor = torch.arange(64).reshape(8, 8).float()
        expected = {
            0: tensor[:2],  # rows 0-1
            1: tensor[2:4],  # rows 2-3
            2: tensor[4:6],  # rows 4-5
            3: tensor[6:8],  # rows 6-7
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0), Shard(0)], param_shape=(8, 8), local_shape=(2, 8))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_strided_shard_same_dim(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor[[0]],  # row 0
            1: tensor[[2]],  # row 2
            2: tensor[[1]],  # row 1
            3: tensor[[3]],  # row 3
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(
                mesh,
                [_StridedShard(dim=0, split_factor=2), Shard(0)],
                param_shape=(4, 4),
                local_shape=(1, 4),
            )
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_strided_shard_different_dims(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: torch.cat([tensor[:2, 0:1], tensor[:2, 2:3]], dim=1),  # top rows, cols {0, 2}
            1: torch.cat([tensor[:2, 1:2], tensor[:2, 3:4]], dim=1),  # top rows, cols {1, 3}
            2: torch.cat([tensor[2:, 0:1], tensor[2:, 2:3]], dim=1),  # bottom rows, cols {0, 2}
            3: torch.cat([tensor[2:, 1:2], tensor[2:, 3:4]], dim=1),  # bottom rows, cols {1, 3}
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(
                mesh,
                [Shard(0), _StridedShard(dim=1, split_factor=2)],
                param_shape=(4, 4),
                local_shape=(2, 2),
            )
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_moe_1D_shard_filters_by_axis0_ownership(self):
        source = torch.ones(2, 2)
        expected = {
            0: {
                0: source,  # first owned
                1: source,  # last owned
                2: None,  # first not-owned (upper boundary, exclusive)
                3: None,  # not owned
            },
            1: {
                0: None,  # not owned
                1: None,  # last not-owned (just below offset)
                2: source,  # first owned (lower boundary, inclusive)
                3: source,  # last owned
            },
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(4, 2, 2), local_shape=(2, 2, 2))
            for tensor_idx, exp in expected[rank].items():
                with self.subTest(rank=rank, tensor_idx=tensor_idx):
                    shard = op.shard_tensor(source, tensor_idx=tensor_idx)
                    if exp is None:
                        self.assertIsNone(shard)
                    else:
                        torch.testing.assert_close(shard, exp)

    def test_moe_1D_strided_shard_on_inner_dim_degrades_to_contiguous(self):
        source = torch.arange(8).reshape(4, 2).float()
        expected = {
            0: source[:2],  # rank 0 — first half (strided silently degraded to contiguous)
            1: source[2:],  # rank 1 — second half
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(
                mesh,
                [_StridedShard(dim=1, split_factor=2)],
                param_shape=(8, 8, 2),
                local_shape=(8, 4, 2),
            )
            torch.testing.assert_close(op.shard_tensor(source, tensor_idx=0), expected[rank], msg=f"rank {rank}")

    def test_moe_2D_shard_on_axis0_and_inner_dim_slices_inner(self):
        source = torch.arange(8).reshape(4, 2).float()
        expected = {
            0: source[:2],  # owned; inner Shard(1) → source dim 0 first half
            1: source[2:],  # owned; inner Shard(1) → source dim 0 second half
            2: None,  # not owned (axis-0 ownership filter)
            3: None,  # not owned
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0), Shard(1)], param_shape=(4, 4, 2), local_shape=(2, 2, 2))
            shard = op.shard_tensor(source, tensor_idx=1)
            if expected[rank] is None:
                self.assertIsNone(shard, msg=f"rank {rank}")
            else:
                torch.testing.assert_close(shard, expected[rank], msg=f"rank {rank}")

    def test_moe_2D_shard_with_negative_dim_indices(self):
        source = torch.arange(8).reshape(4, 2).float()
        expected = {
            0: source[:, :1],  # owned; inner Shard(-1) → source dim 1, first half
            1: source[:, 1:],  # owned; inner Shard(-1) → source dim 1, second half
            2: None,  # not owned
            3: None,  # not owned
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(-3), Shard(-1)], param_shape=(4, 4, 2), local_shape=(2, 4, 1))
            shard = op.shard_tensor(source, tensor_idx=1)
            if expected[rank] is None:
                self.assertIsNone(shard, msg=f"rank {rank}")
            else:
                torch.testing.assert_close(shard, expected[rank], msg=f"rank {rank}")

    def test_strided_intervals(self):
        # Direct tests for _strided_intervals(intervals, rank, world_size, split_factor).
        # Keys: (input_interval, rank, world_size, split_factor) -> expected output list.
        mesh = FakeMesh(shape=(2,), rank=0)
        op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(8,), local_shape=(4,))
        expected = {
            # Even (0, 8) sf=2 ws=2 -> groups (0,4) (4,8); each rank takes half of each
            ((0, 8), 0, 2, 2): [(0, 2), (4, 6)],
            ((0, 8), 1, 2, 2): [(2, 4), (6, 8)],
            # Uneven (0, 7) sf=2 -> group 0 = (0,4), group 1 = (4,7) -> size 3
            ((0, 7), 0, 2, 2): [(0, 2), (4, 6)],  # rank 0 -> half of each group
            ((0, 7), 1, 2, 2): [(2, 4), (6, 7)],  # rank 1's piece in group 1 is 1 elem wide
            # split_factor=1 collapses to a single group -> contiguous behavior
            ((0, 4), 0, 2, 1): [(0, 2)],
            # split_factor=4 on size-2: groups (0,1), (1,2), (2,2), (3,2) -> last 2 empty -> skipped
            ((0, 2), 0, 2, 4): [(0, 1), (1, 2)],
        }
        for (interval, rank, ws, sf), exp in expected.items():
            with self.subTest(interval=interval, rank=rank, ws=ws, sf=sf):
                self.assertEqual(op._strided_intervals([interval], rank=rank, world_size=ws, split_factor=sf), exp)

    def test_contiguous_intervals(self):
        # Direct tests for _contiguous_intervals(intervals, rank, world_size).
        # Keys: (input_intervals, rank, world_size) -> expected output list.
        mesh = FakeMesh(shape=(2,), rank=0)
        op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(8,), local_shape=(4,))
        expected = {
            # Single interval, even split -> rank 0 -> first 4 elems, rank 1 -> last 4 elems
            (((0, 8),), 0, 2): [(0, 4)],
            (((0, 8),), 1, 2): [(4, 8)],
            # Uneven: size 5 / 2 ranks -> rank 0 -> first 3 elems, rank 1 -> last 2 elems
            (((0, 5),), 0, 2): [(0, 3)],
            (((0, 5),), 1, 2): [(3, 5)],
            # Empty: size 3 / 4 ranks -> rank 3 -> nothing
            (((0, 3),), 3, 4): [],
            # Multi-input intervals (8 elems): rank -> takes its slice from whichever interval(s) cover it
            (((0, 4), (10, 14)), 0, 2): [(0, 4)],  # rank 0 -> first 4 elems = first interval entirely
            (((0, 4), (10, 14)), 1, 2): [(10, 14)],  # rank 1 -> last 4 elems = second interval entirely
            (((0, 4), (10, 14)), 2, 4): [(10, 12)],  # ws=4, rank 2 -> cuts mid-input-interval
        }
        for (intervals, rank, ws), exp in expected.items():
            with self.subTest(intervals=intervals, rank=rank, ws=ws):
                self.assertEqual(op._contiguous_intervals(list(intervals), rank=rank, world_size=ws), exp)

    def test_slice_and_cat(self):
        # Direct tests for _slice_and_cat(source, intervals, device, dtype).
        tensor = torch.arange(64).reshape(8, 8).float()
        mesh = FakeMesh(shape=(2,), rank=0)
        op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(8, 8), local_shape=(4, 8))
        expected = {
            # Fast path: every dim is single-interval -> one slice read, no cat
            "fast_path": ([[(0, 4)], [(0, 8)]], tensor[:4, :]),
            # Cat along dim 1: two disjoint col ranges -> read separately and concat
            "cat_dim1": ([[(0, 4)], [(0, 2), (4, 6)]], torch.cat([tensor[:4, :2], tensor[:4, 4:6]], dim=1)),
            # Cat along dim 0: two disjoint row ranges -> read separately and concat
            "cat_dim0": ([[(0, 2), (4, 6)], [(0, 8)]], torch.cat([tensor[:2, :], tensor[4:6, :]], dim=0)),
        }
        for case, (intervals, exp) in expected.items():
            with self.subTest(case=case):
                torch.testing.assert_close(op._slice_and_cat(tensor, intervals, None, None), exp)

        # Reject: two dims with disjoint ranges -> would require an outer-product of reads.
        with self.assertRaises(ValueError):
            op._slice_and_cat(tensor, [[(0, 2), (4, 6)], [(0, 2), (4, 6)]], None, None)

        result = op._slice_and_cat(tensor, [[(0, 4)], [(0, 8)]], None, torch.float16)
        self.assertEqual(result.dtype, torch.float16)


class TestConversionMapping(unittest.TestCase):
    def test_register_checkpoint_conversion_mapping(self):
        register_checkpoint_conversion_mapping(
            "foobar",
            [
                WeightRenaming(".block_sparse_moe.gate", ".mlp.gate"),
            ],
        )
        self.assertEqual(len(get_checkpoint_conversion_mapping("foobar")), 1)

    def test_register_checkpoint_conversion_mapping_overwrites(self):
        register_checkpoint_conversion_mapping(
            "foobarbaz",
            [
                WeightRenaming(".block_sparse_moe.gate", ".mlp.gate"),
            ],
        )
        with self.assertRaises(ValueError):
            register_checkpoint_conversion_mapping(
                "foobarbaz",
                [
                    WeightRenaming(".block_sparse_moe.foo", ".mlp.foo"),
                    WeightRenaming(".block_sparse_moe.bar", ".mlp.bar"),
                ],
            )

        register_checkpoint_conversion_mapping(
            "foobarbaz",
            [
                WeightRenaming(".block_sparse_moe.foo", ".mlp.foo"),
                WeightRenaming(".block_sparse_moe.bar", ".mlp.bar"),
            ],
            overwrite=True,
        )

        self.assertEqual(len(get_checkpoint_conversion_mapping("foobarbaz")), 2)


if __name__ == "__main__":
    unittest.main()
