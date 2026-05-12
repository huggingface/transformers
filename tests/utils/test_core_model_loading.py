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
import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from transformers import PretrainedConfig, PreTrainedModel
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    get_model_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    Conv3dToLinear,
    ErnieFuseAndSplitTextVisionExperts,
    LinearToConv3d,
    MergeModulelist,
    PermuteForRope,
    PrefixChange,
    WeightConverter,
    WeightRenaming,
    build_glob_alternation,
    convert_and_load_state_dict_in_model,
    rename_source_key,
    revert_weight_conversion,
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


class DummyRoot(PreTrainedModel):
    base_model_prefix = "model"
    config: PretrainedConfig

    def __init__(self, config, add_extra_moe=False, with_mlp=True):
        super().__init__(config)
        self.model = DummyTopModel(add_extra_moe)
        if with_mlp:
            self.mlp = DummyMLP()
        self.post_init()


class TestConvertAndLoadStateDict(unittest.TestCase):
    def test_moe_and_qkv_conversion(self):
        model = DummyRoot(PretrainedConfig())

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
        model = DummyRoot(PretrainedConfig())

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
            from transformers.integrations.finegrained_fp8 import Fp8Dequantize, Fp8Quantize
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

        class RopeModel(PreTrainedModel):
            base_model_prefix = "model"

            def __init__(self, config):
                super().__init__(config)
                self.layers = nn.ModuleList([RopeLayer()])
                self.post_init()

        config = PretrainedConfig()
        config.num_attention_heads = n_heads
        model = RopeModel(config)

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
                "get_quantize_ops": lambda self: Fp8Quantize(self),
                "pre_quantized": False,
            },
        )
        quantizer = quantizer_cls()

        weight_mapping = [
            WeightConverter(
                "self_attn.qkv_proj.weight",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
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
        torch.testing.assert_close(model_state["model.layers.0.self_attn.k_proj.weight"], expected_k)
        torch.testing.assert_close(model_state["model.layers.0.self_attn.v_proj.weight"], expected_v)

        q_weight_key = "model.layers.0.self_attn.q_proj.weight"
        scale_key = "model.layers.0.self_attn.q_proj.weight_scale_inv"
        self.assertIn(scale_key, model_state)
        expected_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.int8
        self.assertEqual(model_state[q_weight_key].dtype, expected_dtype)
        self.assertEqual(model_state[q_weight_key].shape, torch.Size((out_dim, in_dim)))
        self.assertEqual(model_state[scale_key].dtype, torch.float32)
        self.assertEqual(
            model_state[scale_key].shape,
            torch.Size((out_dim // block_size[0], in_dim // block_size[1])),
        )

        dequant = Fp8Dequantize(block_size=block_size)
        dequantized_q = dequant.convert(
            [model_state[q_weight_key], model_state[scale_key]],
            context={"quantization_config": quantizer.quantization_config},
        )
        torch.testing.assert_close(dequantized_q, expected_q, rtol=1e-2, atol=1e-2)

    def test_scoped_renaming_does_not_leak_to_sibling_or_parent(self):
        """scope_prefix gates a WeightRenaming to keys under one submodel only —
        neither the sibling submodel nor the parent's own keys must be affected.

        This mirrors the real use-case: a composite model (e.g. vision-language) has
        two top-level submodels and a root-level weight, each with an ``old_q`` key.
        The rename is registered only for ``vision_model``; the sibling
        ``language_model.old_q`` and the root-level ``old_q`` must remain unmatched.
        """

        class _Submodel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = DummyParamModule((1, 2))

        class _CompositeModel(PreTrainedModel):
            base_model_prefix = ""

            def __init__(self, config):
                super().__init__(config)
                self.vision_model = _Submodel()
                self.language_model = _Submodel()
                self.q = DummyParamModule((1, 2))  # root-level weight with the same name
                self.post_init()

        model = _CompositeModel(PretrainedConfig())

        vision_val = torch.tensor([[1.0, 2.0]])
        checkpoint = {
            "vision_model.old_q.weight": vision_val.clone(),
            "language_model.old_q.weight": torch.tensor([[9.0, 9.0]]),
            "old_q.weight": torch.tensor([[7.0, 7.0]]),  # root-level, must not be renamed
        }

        scoped_rename = WeightRenaming("^old_q", "q")
        scoped_rename.scope_prefix = "vision_model"

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            checkpoint,
            LoadStateDictConfig(weight_mapping=[scoped_rename]),
            tp_plan=None,
        )

        # Sibling and parent keys must be unmatched.
        self.assertEqual(loading_info.unexpected_keys, {"language_model.old_q.weight", "old_q.weight"})
        self.assertEqual(loading_info.missing_keys, {"language_model.q.weight", "q.weight"})
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        torch.testing.assert_close(model.vision_model.q.weight, vision_val)
        torch.testing.assert_close(model.language_model.q.weight, torch.zeros(1, 2))
        torch.testing.assert_close(model.q.weight, torch.zeros(1, 2))

    def test_interleaved_renaming_and_converter_round_trip(self):
        """A WeightRenaming preceding a WeightConverter must also fire on the save path
        even after the converter has already set source_pattern.

        Loading (checkpoint key → model key):
          1. WeightRenaming  (source "^decoder", target "encoder"):
             "decoder.attn.qkv_proj.weight"          → "encoder.attn.qkv_proj.weight"
          2. WeightConverter (source "^attn.qkv_proj.weight", targets "attn.{q,k,v}_proj.weight", scope "encoder"):
             strips "encoder.", matches source, unpacks QKV, re-attaches "encoder."
             "encoder.attn.qkv_proj.weight"          → "encoder.attn.{q,k,v}_proj.weight"

        Saving (model key → checkpoint key, transforms applied in reverse):
          1. rev(WeightConverter) (source "attn.{q,k,v}_proj.weight", target "^attn.qkv_proj.weight", scope "encoder"):
             strips "encoder.", matches source, repacks QKV, re-attaches "encoder."
             "encoder.attn.{q,k,v}_proj.weight"      → "encoder.attn.qkv_proj.weight"
          2. rev(WeightRenaming) (source "^encoder", target "decoder"):
             "encoder.attn.qkv_proj.weight"          → "decoder.attn.qkv_proj.weight"
        """

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = DummyParamModule((2, 4))
                self.k_proj = DummyParamModule((2, 4))
                self.v_proj = DummyParamModule((2, 4))

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = _Attn()

        class _InterleavedModel(PreTrainedModel):
            base_model_prefix = ""

            def __init__(self, config):
                super().__init__(config)
                self.encoder = _Encoder()
                self.post_init()

        qkv = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        model = _InterleavedModel(PretrainedConfig())

        # Checkpoint uses a "decoder" prefix and stores QKV packed together.
        checkpoint = {"decoder.attn.qkv_proj.weight": qkv.clone()}

        qkv_converter = WeightConverter(
            "^attn.qkv_proj.weight",
            ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"],
            operations=[Chunk(dim=0)],
        )
        # scope_prefix mirrors what get_model_conversion_mapping sets for a submodel
        qkv_converter.scope_prefix = "encoder"

        weight_mapping = [
            WeightRenaming("^decoder", "encoder"),  # step 1: fix prefix
            qkv_converter,  # step 2: unpack QKV (fires after rename, scoped to encoder)
        ]

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            checkpoint,
            LoadStateDictConfig(weight_mapping=weight_mapping),
            tp_plan=None,
        )

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        q, k, v = torch.chunk(qkv, 3, dim=0)
        torch.testing.assert_close(model.encoder.attn.q_proj.weight, q)
        torch.testing.assert_close(model.encoder.attn.k_proj.weight, k)
        torch.testing.assert_close(model.encoder.attn.v_proj.weight, v)

        # Round-trip: saving must reconstruct the original "decoder.*" checkpoint.
        # This relies on rev(WeightRenaming) firing after rev(WeightConverter) has set
        # source_pattern — if it were skipped the prefix would remain "encoder".
        saved = revert_weight_conversion(model, model.state_dict())
        self.assertTrue(compare_state_dicts(saved, checkpoint))

    def test_ernie4_5_vl_moe_conversion(self):
        model = DummyRoot(PretrainedConfig(), add_extra_moe=True)

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
        model = DummyRoot(PretrainedConfig(), add_extra_moe=True)

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


class TestConversionMapping(unittest.TestCase):
    def test_conv3d_linear_conversion_ops(self):
        weight_name = "patch_embed.proj.weight"
        kernel_size = (2, 2, 2)

        def convert(converter, weight):
            return converter.convert({weight_name: weight}, [weight_name], [weight_name])[weight_name]

        conv_to_linear = Conv3dToLinear(in_channels=3, kernel_size=kernel_size)
        linear_to_conv = LinearToConv3d(in_channels=3, kernel_size=kernel_size)

        conv_weight = torch.randn(2 * 3 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 3, *kernel_size)
        linear_weight = torch.randn(2 * 24, dtype=torch.float32).reshape(2, 24)

        # test conv3d -> linear
        torch.testing.assert_close(convert(conv_to_linear, conv_weight), conv_weight.reshape(2, 24))
        # test linear -> conv3d
        torch.testing.assert_close(convert(linear_to_conv, linear_weight), linear_weight.reshape(2, 3, *kernel_size))
        # test conv3d -> linear -> conv3d round trip
        torch.testing.assert_close(
            convert(conv_to_linear.reverse_op, convert(conv_to_linear, conv_weight)), conv_weight
        )

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

    def test_can_remove_prefix(self):
        model = DummyRoot(PretrainedConfig())

        bad_serialized_checkpoints = {f"bad_name.{k}": v.clone() for k, v in model.state_dict().items()}
        weight_mapping = [PrefixChange(prefix_to_remove="bad_name")]

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            bad_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, re-adding the bad prefix
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(bad_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == bad_serialized_checkpoints[k]).all())

        # Now, check that using the same conversion with already good keys works when loading and resaving
        good_serialized_checkpoints = {k: v.clone() for k, v in model.state_dict().items()}

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            good_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, i.e. it will not re-add the bad prefix since it was
        # not present at loading time
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(good_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == good_serialized_checkpoints[k]).all())

        # Now, use a fresh model, without going trough loading first, so the model won't have `_weight_conversions` attached
        # and the prefix should not be added when saving directly (i.e. the conversion should be dropped)
        model = DummyRoot(PretrainedConfig())
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        model_state_dict = model.state_dict()
        self.assertEqual(set(model_state_dict.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == model_state_dict[k]).all())

    def test_can_add_prefix(self):
        # we cannot have another param next to the model, otherwise the prefix adding will already be added even with correct
        # checkpoints starting with the prefix
        model = DummyRoot(PretrainedConfig(), with_mlp=False)

        bad_serialized_checkpoints = {k.removeprefix("model."): v.clone() for k, v in model.state_dict().items()}
        weight_mapping = [PrefixChange(prefix_to_add="model")]

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            bad_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, re-adding the bad prefix
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(bad_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == bad_serialized_checkpoints[k]).all())

        # Now, check that using the same conversion with already good keys works when loading and resaving
        good_serialized_checkpoints = {k: v.clone() for k, v in model.state_dict().items()}

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            good_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, i.e. it will not remove the prefix since it was
        # already present at loading time
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(good_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == good_serialized_checkpoints[k]).all())

        # Now, use a fresh model, without going trough loading first, so the model won't have `_weight_conversions` attached
        # and the prefix should not be removed when saving directly (i.e. the conversion should be dropped)
        model = DummyRoot(PretrainedConfig())
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        model_state_dict = model.state_dict()
        self.assertEqual(set(model_state_dict.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == model_state_dict[k]).all())

    def test_can_remove_prefix_submodule(self):
        model = DummyRoot(PretrainedConfig())

        bad_serialized_checkpoints = {
            f"model.layers.bad_name.{k.replace('model.layers.', '')}" if "model.layers." in k else k: v.clone()
            for k, v in model.state_dict().items()
        }
        weight_mapping = [PrefixChange(prefix_to_remove="bad_name", model_prefix="model.layers")]

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            bad_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, re-adding the bad prefix
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(bad_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == bad_serialized_checkpoints[k]).all())

        # Now, check that using the same conversion with already good keys works when loading and resaving
        good_serialized_checkpoints = {k: v.clone() for k, v in model.state_dict().items()}

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            good_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, i.e. it will not re-add the bad prefix since it was
        # not present at loading time
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(good_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == good_serialized_checkpoints[k]).all())

        # Now, use a fresh model, without going trough loading first, so the model won't have `_weight_conversions` attached
        # and the prefix should not be added when saving directly (i.e. the conversion should be dropped)
        model = DummyRoot(PretrainedConfig())
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        model_state_dict = model.state_dict()
        self.assertEqual(set(model_state_dict.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == model_state_dict[k]).all())

    def test_can_add_prefix_submodule(self):
        # we cannot have another param next to the model, otherwise the prefix adding will already be added even with correct
        # checkpoints starting with the prefix
        model = DummyRoot(PretrainedConfig(), with_mlp=False)

        bad_serialized_checkpoints = {k.replace(".layers.", "."): v.clone() for k, v in model.state_dict().items()}
        weight_mapping = [PrefixChange(prefix_to_add="layers", model_prefix="model")]

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            bad_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, re-adding the bad prefix
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(bad_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == bad_serialized_checkpoints[k]).all())

        # Now, check that using the same conversion with already good keys works when loading and resaving
        good_serialized_checkpoints = {k: v.clone() for k, v in model.state_dict().items()}

        loading_info, _ = convert_and_load_state_dict_in_model(
            model,
            good_serialized_checkpoints,
            LoadStateDictConfig(weight_mapping=copy.deepcopy(weight_mapping)),
            tp_plan=None,
        )

        # Assert we can load without issues
        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        # Assert that re-saving will lead to the exact same state_dict, i.e. it will not remove the prefix since it was
        # already present at loading time
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        self.assertEqual(set(good_serialized_checkpoints.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == good_serialized_checkpoints[k]).all())

        # Now, use a fresh model, without going trough loading first, so the model won't have `_weight_conversions` attached
        # and the prefix should not be removed when saving directly (i.e. the conversion should be dropped)
        model = DummyRoot(PretrainedConfig())
        saved_state_dict = revert_weight_conversion(model, model.state_dict())
        model_state_dict = model.state_dict()
        self.assertEqual(set(model_state_dict.keys()), set(saved_state_dict.keys()))
        for k, v in saved_state_dict.items():
            self.assertTrue((v == model_state_dict[k]).all())

    def test_class_name_wins_over_model_type(self):
        """Class-name registry entry takes priority over model_type for the same model."""
        register_checkpoint_conversion_mapping("_TstCls", [WeightRenaming(r"^cls_key", "cls_renamed")], overwrite=True)
        register_checkpoint_conversion_mapping(
            "_tst_mtype", [WeightRenaming(r"^type_key", "type_renamed")], overwrite=True
        )

        class _TstCls(PreTrainedModel): ...

        class _TstOther(PreTrainedModel): ...

        # A module whose class name has a registry entry → class entry wins.
        transforms = get_model_conversion_mapping(_TstCls(PretrainedConfig(model_type="_tst_mtype")), add_legacy=False)
        patterns = [t.source_patterns for t in transforms]
        self.assertIn(["^cls_key"], patterns)
        self.assertNotIn(["^type_key"], patterns)

        # A module with no class entry falls through to the model_type entry.
        transforms = get_model_conversion_mapping(
            _TstOther(PretrainedConfig(model_type="_tst_mtype")), add_legacy=False
        )
        patterns = [t.source_patterns for t in transforms]
        self.assertIn(["^type_key"], patterns)
        self.assertNotIn(["^cls_key"], patterns)

    def test_sibling_submodels_same_model_type_both_get_transforms(self):
        """Two sibling sub-models sharing a model_type must each get their own scoped transforms.

        Old flat-set deduplication would mark the model_type seen after the first
        sibling and silently skip the second one.  The ancestor-based check must
        allow both because neither sibling is an ancestor of the other.
        """
        register_checkpoint_conversion_mapping(
            "_tst_shared_type", [WeightRenaming(r"^w", "renamed_w")], overwrite=True
        )

        class _TstEncCls(PreTrainedModel): ...

        class _TstDecCls(PreTrainedModel): ...

        class _TstRoot(PreTrainedModel): ...

        child_a = _TstEncCls(PretrainedConfig(model_type="_tst_shared_type"))
        child_b = _TstDecCls(PretrainedConfig(model_type="_tst_shared_type"))
        root = _TstRoot(PretrainedConfig(model_type="_tst_root_only"))
        root.encoder = child_a
        root.decoder = child_b

        transforms = get_model_conversion_mapping(root, add_legacy=False)
        scope_prefixes = [t.scope_prefix for t in transforms]

        # Both siblings must be represented with their own scoped transforms.
        self.assertIn("encoder", scope_prefixes)
        self.assertIn("decoder", scope_prefixes)

    def test_sibling_submodels_same_class_both_get_transforms(self):
        """Two sibling sub-models of the *same* class must each get their own scoped transforms."""
        register_checkpoint_conversion_mapping("_TstSharedCls", [WeightRenaming(r"^w", "renamed_w")], overwrite=True)

        class _TstSharedCls(PreTrainedModel): ...

        class _TstRootSharedCls(PreTrainedModel): ...

        child_a = _TstSharedCls(PretrainedConfig(model_type="_tst_shared_cls_mtype"))
        child_b = _TstSharedCls(PretrainedConfig(model_type="_tst_shared_cls_mtype"))
        root = _TstRootSharedCls(PretrainedConfig(model_type="_tst_root_only2"))
        root.encoder = child_a
        root.decoder = child_b

        transforms = get_model_conversion_mapping(root, add_legacy=False)
        scope_prefixes = [t.scope_prefix for t in transforms]

        self.assertIn("encoder", scope_prefixes)
        self.assertIn("decoder", scope_prefixes)

    def test_child_with_same_model_type_as_root_is_skipped(self):
        """When the root model claims a model_type unscoped, a nested child with the
        same model_type must NOT produce a second (incorrectly scoped) copy of those
        transforms — the root's unscoped transforms already cover all keys."""

        class _TstChildSame(PreTrainedModel): ...

        class _TstRootSame(PreTrainedModel): ...

        register_checkpoint_conversion_mapping(
            "_tst_root_child_shared", [WeightRenaming(r"^w", "renamed_w")], overwrite=True
        )

        child = _TstChildSame(PretrainedConfig(model_type="_tst_root_child_shared"))
        root = _TstRootSame(PretrainedConfig(model_type="_tst_root_child_shared"))
        root.submodel = child

        transforms = get_model_conversion_mapping(root, add_legacy=False)
        # Only one unscoped transform (from the root); child must be suppressed.
        self.assertEqual(len(transforms), 1)
        self.assertIsNone(transforms[0].scope_prefix)


if __name__ == "__main__":
    unittest.main()
