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
import re
import unittest

import torch
import torch.nn as nn

from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    MergeModulelist,
    WeightConverter,
    _apply_star_subst,
    _glob_to_regex_src,
    build_glob_alt,
    convert_and_load_state_dict_in_model,
    glob_to_re,
    match_glob,
)


class TestGlobRegexHelpers(unittest.TestCase):
    def test_glob_to_regex_src_digits_only(self):
        pattern = _glob_to_regex_src("model.layers.*.mlp.weight", digits_only=True)
        self.assertEqual(pattern, r"model\.layers\.(\d+)\.mlp\.weight")

    def test_glob_to_regex_src_any_chars(self):
        pattern = _glob_to_regex_src("model.layers.*.mlp.weight", digits_only=False)
        self.assertEqual(pattern, r"model\.layers\.(.+)\.mlp\.weight")

    def test_glob_to_re_fullmatch(self):
        regex_src = glob_to_re("model.layers.*.mlp.weight", digits_only=True)
        regex = re.compile(f"^{regex_src}$")
        self.assertIsNotNone(regex.fullmatch("model.layers.12.mlp.weight"))
        self.assertIsNone(regex.fullmatch("model.layers.foo.mlp.weight"))

    def test_apply_star_subst(self):
        pattern = "model.layers.*.block.*.weight"
        replaced = _apply_star_subst(pattern, ["03", "attn"])
        self.assertEqual(replaced, "model.layers.03.block.attn.weight")

    def test_build_glob_alt_without_prefix(self):
        globs = ["model.layers.*.weight"]
        alt, mapping = build_glob_alt(globs, allow_prefix=False)
        self.assertIsNone(match_glob("foo.model.layers.0.weight", alt, mapping))
        self.assertEqual(match_glob("model.layers.0.weight", alt, mapping), "model.layers.*.weight")

    def test_build_glob_alt_with_prefix(self):
        globs = ["layers.*.weight"]
        alt, mapping = build_glob_alt(globs, allow_prefix=True)
        self.assertEqual(match_glob("model.layers.0.weight", alt, mapping), "layers.*.weight")


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
    def __init__(self):
        super().__init__()
        self.model = DummyTopModel()
        self.mlp = DummyMLP()


class TestConvertAndLoadStateDict(unittest.TestCase):
    def test_moe_and_qkv_conversion(self):
        model = DummyRoot()

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
        state_dict = {k: (k, v.clone()) for k, v in raw_tensors.items()}

        weight_mapping = [
            WeightConverter(
                ["model.layers.*.experts.*.w1.weight", "model.layers.*.experts.*.w3.weight"],
                "model.layers.*.experts.gate_up_proj.weight",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                "model.layers.*.experts.*.w2.weight",
                "model.layers.*.experts.down_proj.weight",
                operations=[MergeModulelist(dim=0)],
            ),
            WeightConverter(
                "model.layers.*.self_attn.qkv_proj.weight",
                [
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ],
                operations=[Concatenate(dim=0), Chunk(dim=0, chunks=3)],
            ),
            WeightConverter("mlp.w2.weight", "mlp.down_proj.weight"),
        ]

        missing, unexpected, mismatch, misc = convert_and_load_state_dict_in_model(
            model, state_dict, weight_mapping, tp_plan=None, hf_quantizer=None
        )

        self.assertEqual(missing, set())
        self.assertEqual(unexpected, set())
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
            torch.testing.assert_close(model_state[f"{prefix}.q_proj.weight"], expected_q)
            torch.testing.assert_close(model_state[f"{prefix}.k_proj.weight"], expected_k)
            torch.testing.assert_close(model_state[f"{prefix}.v_proj.weight"], expected_v)

        torch.testing.assert_close(model_state["mlp.down_proj.weight"], raw_tensors["mlp.w2.weight"])


if __name__ == "__main__":
    unittest.main()
