# Copyright 2024 HuggingFace Inc.
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

from transformers import PretrainedConfig
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    MergeModulelist,
    PermuteForRope,
    WeightConverter,
    WeightRenaming,
    build_glob_alternation,
    convert_and_load_state_dict_in_model,
    repl,
)
from transformers.utils.import_utils import is_triton_available


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
        rename_alt, _, rename_by_group = build_glob_alternation(renamings)

        def rename(original_key: str) -> str:
            return rename_alt.sub(lambda m: repl(m, rename_by_group), original_key).replace("\\", "")

        self.assertEqual(rename("foo.block_sparse_moe.experts.3.w1.weight"), "foo.mlp.experts.gate_up_proj")
        self.assertEqual(rename("foo.block_sparse_moe.experts.3.w2.weight"), "foo.mlp.experts.down_proj")
        self.assertEqual(rename("model.language_model.lm_head.weight"), "language_model")

    def test_sub_key_no_match_returns_original(self):
        renamings = [
            WeightRenaming("block_sparse_moe.experts.*.w1.weight", "*.mlp.experts.gate_up_proj"),
        ]
        rename_alt, _, rename_by_group = build_glob_alternation(renamings)

        key = "unrelated.key"
        renamed_key = rename_alt.sub(lambda m: repl(m, rename_by_group), key).replace("\\", "")
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
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttn()
        self.experts = DummyExperts()


class DummyTopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(), DummyLayer()])


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = DummyParamModule((2, 2))


class DummyRoot(nn.Module):
    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.model = DummyTopModel()
        self.mlp = DummyMLP()


class TestConvertAndLoadStateDict(unittest.TestCase):
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
                operations=[Chunk(dim=0, chunks=3)],
            ),
            WeightRenaming("mlp.w2.weight", "mlp.down_proj.weight"),
        ]
        missing, unexpected, mismatch, _, misc = convert_and_load_state_dict_in_model(
            model, state_dict, weight_mapping, tp_plan=None, hf_quantizer=None
        )

        self.assertEqual(
            missing,
            {
                "model.layers.1.self_attn.k_proj.weight",
                "model.layers.1.self_attn.v_proj.weight",
                "model.layers.1.self_attn.q_proj.weight",
            },
        )
        self.assertEqual(unexpected, {"model.layers.1.self_attn.qkv_proj.weight"})
        self.assertEqual(mismatch, set())
        self.assertEqual(misc, {})

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
                operations=[Chunk(dim=0, chunks=3), PermuteForRope()],
            )
        ]

        missing, unexpected, mismatch, _, misc = convert_and_load_state_dict_in_model(
            model, state_dict, weight_mapping, tp_plan=None, hf_quantizer=quantizer
        )

        self.assertEqual(missing, set())
        self.assertEqual(unexpected, set())
        self.assertEqual(mismatch, set())
        self.assertEqual(misc, {})

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


if __name__ == "__main__":
    unittest.main()
