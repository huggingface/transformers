# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""The `finegrained` conversion ops: what each one emits from a checkpoint layout, and how the
loader chains them. Pure torch, so no kernel is loaded and no GPU is needed."""

import re
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import transformers.integrations.finegrained.core as fg
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.core_model_loading import (
    Chunk,
    PermuteForRope,
    WeightConverter,
    convert_and_load_state_dict_in_model,
)
from transformers.integrations.finegrained import FineGrainedExperts
from transformers.modeling_utils import LoadStateDictConfig
from transformers.testing_utils import require_torch

from .test_core import _Cfg, _quantizer_for, _targets


@require_torch
class FineGrainedMxfp4ConverterTest(unittest.TestCase):
    """The gpt-oss checkpoint layout through the real ConversionOps: blocks (E, N, K/32, 16)
    uint8 low-nibble-first E2M1 + biased-127 exponent scales, gate|up rows INTERLEAVED —
    deserialized output must dequantize identically to the reference LUT unpack."""

    FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def _reference_dequant(self, blocks, scales):
        # the mxfp4.py reference semantics: lut[lo], lut[hi] interleaved along K, x 2^(scale-127)
        lut = torch.tensor(self.FP4_VALUES)
        lo = lut[(blocks & 0xF).long()]
        hi = lut[(blocks >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(*blocks.shape[:-1], -1)  # (E, N, K/32, 32)
        exp = (scales.long() - 127).unsqueeze(-1)
        return (vals * torch.pow(torch.tensor(2.0), exp)).reshape(*blocks.shape[:2], -1)

    def test_blocks_and_scales_convert_to_the_kernel_pair_and_back(self):
        from transformers.integrations.finegrained.conversions import (
            FineGrainedPackedBlocks,
            FineGrainedScaleContainer,
        )

        torch.manual_seed(0)
        E, N, K = 2, 8, 64  # N = 2I interleaved gate|up rows
        blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype=torch.uint8)
        scales = torch.randint(110, 140, (E, N, K // 32), dtype=torch.uint8)

        weight = FineGrainedPackedBlocks().convert({"gate_up_proj_blocks$": blocks}, target_patterns=["gate_up_proj"])[
            "gate_up_proj"
        ]
        self.assertEqual((weight.dtype, weight.shape), (torch.int8, (E, N, K // 2)))
        back = FineGrainedPackedBlocks().reverse_op.convert({"gate_up_proj": weight})["gate_up_proj"]
        self.assertTrue(torch.equal(back, blocks))

        # the scales are the container op's job: a module holding e8m0 receives the exponent bytes as is
        module = fg.FineGrainedExperts.__new__(fg.FineGrainedExperts)
        torch.nn.Module.__init__(module)
        module.gate_up_proj_scale_inv = torch.nn.Parameter(
            torch.zeros(E, N, K // 32, dtype=torch.float8_e8m0fnu), requires_grad=False
        )
        model = torch.nn.Module()
        model.experts = module
        scale_inv = FineGrainedScaleContainer().convert(
            {"gate_up_proj_scales$": scales},
            model=model,
            full_layer_name="experts.gate_up_proj_scale_inv",
            target_patterns=["gate_up_proj_scale_inv"],
        )["gate_up_proj_scale_inv"]
        self.assertEqual(scale_inv.dtype, torch.float8_e8m0fnu)

        ref = self._reference_dequant(blocks, scales)  # rows pass through interleaved
        # dequantize the converted pair: packed E2M1 low-nibble-first x 2^(e8m0)
        lut = torch.tensor(self.FP4_VALUES)
        w_u8 = weight.view(torch.uint8)
        lo = lut[(w_u8 & 0xF).long()]
        hi = lut[(w_u8 >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(E, N, -1)
        exp = (scale_inv.view(torch.uint8).long() - 127).repeat_interleave(32, dim=-1)
        got = vals * torch.pow(torch.tensor(2.0), exp)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@require_torch
class FineGrainedOnTheFlyQuantizeTest(unittest.TestCase):
    """`FineGrainedQuantize` reads the module's format: block-FP8 in torch (the group formats run
    the kernels' own quantizers, which need a GPU), emitted in the module's layout."""

    def test_mxfp4_blocks_dequantize_like_the_integration_they_replace(self):
        """`dequantize=True` on a GPT-OSS MXFP4 checkpoint used to be the mxfp4 integration's job.
        The finegrained chain has to land the same bf16 tensor, in the (E, hidden, 2I) orientation
        the unquantized experts hold — so it is compared against that integration directly."""
        from transformers.core_model_loading import Transpose
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedPackedBlocks
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        torch.manual_seed(0)
        num_experts, rows, hidden = 2, 8, 64
        blocks = torch.randint(0, 256, (num_experts, rows, hidden // 32, 16), dtype=torch.uint8)
        scales = torch.randint(120, 134, (num_experts, rows, hidden // 32), dtype=torch.uint8)
        reference = convert_moe_packed_tensors(blocks.clone(), scales.clone(), dtype=torch.bfloat16)

        sources = ["gate_up_proj_blocks$", "gate_up_proj_scales$"]
        tensors = {sources[0]: blocks.clone(), sources[1]: scales.clone()}
        for op in (FineGrainedPackedBlocks(None), FineGrainedDequantize(None), Transpose(1, 2)):
            tensors = op.convert(tensors, source_patterns=sources, target_patterns=["gate_up_proj"])
        self.assertEqual(len(tensors), 1)  # the scale is consumed, not passed down the chain
        torch.testing.assert_close(next(iter(tensors.values())).float(), reference.float(), rtol=0, atol=0)

    def test_a_static_scheme_gets_its_activation_scales_written(self):
        """A calibration-fed slot the checkpoint does not supply is still WRITTEN, as the identity.

        The loader materializes a missing key with `torch.empty_like` and `_init_weights` has no
        branch for a scale, so a slot this op leaves out reaches the kernels as uninitialized
        memory — a zero divides the activations by zero, a negative flips their sign.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        model = torch.nn.Module()
        model.experts = fg.FineGrainedExperts(
            cfg, block_size=(128, 128), weight_format="fp8", activation_scheme="static"
        )
        model.proj = fg.FineGrainedLinear(
            in_features=256,
            out_features=256,
            block_size=(128, 128),
            weight_format="fp8",
            activation_scheme="static",
        )
        op = FineGrainedQuantize(hf_quantizer=None)
        for key, tensor, slot in (
            ("experts.gate_up_proj", torch.randn(4, 256, 256), "experts.gate_up_proj_activation_scale"),
            ("proj.weight", torch.randn(256, 256), "proj.activation_scale"),
        ):
            with self.subTest(key=key):
                out = op.convert({key: tensor}, model=model)
                self.assertIn(slot, out, f"{key}: the static activation scale was not written")
                held = model.get_parameter(slot)
                self.assertEqual((out[slot].shape, out[slot].dtype), (held.shape, held.dtype))
                torch.testing.assert_close(out[slot], torch.ones_like(out[slot]), rtol=0, atol=0)

    def test_an_expert_bias_is_passed_through_not_quantized(self):
        """A GPT-OSS expert bias is `(E, rows)` — 2-D, like a dense weight — so the rank guard
        alone lets it through. It has no scale slot, and emitting one gives the loader a
        `<proj>_bias_scale_inv` no module holds."""
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        model = torch.nn.Module()
        model.experts = fg.FineGrainedExperts(cfg, block_size=(128, 128), weight_format="fp8", has_bias=True)
        bias = torch.randn(4, 256)
        out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.down_proj_bias": bias}, model=model)
        self.assertEqual(list(out), ["experts.down_proj_bias"])
        self.assertIs(out["experts.down_proj_bias"], bias)

    def test_a_partial_trailing_block_round_trips_against_the_configured_block(self):
        """DeepSeek-V3 ships `kv_a_proj_with_mqa` as `(576, 7168)` against a 128x128 block, so the
        grid rounds UP to `(5, 56)` and the short block quantizes on its own values.

        The grid cannot be inverted to recover the block: 576 // 5 is 116, and `(192, 256)` divides
        evenly at 96 while the real block is still 128. A dequantize that guesses then reads the
        wrong rows per scale, which only lifts the error to 4.4e-2 against a 2.6e-2 floor — hence
        the tight bound, since a loose one passes the broken arm too."""
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        quantizer = SimpleNamespace(
            quantization_config=SimpleNamespace(weight_block_size=(128, 128), scale_fmt="float")
        )
        for rows, cols in ((576, 256), (192, 256), (256, 256)):
            with self.subTest(shape=(rows, cols)):
                original = torch.randn(rows, cols)
                weight, scale = FineGrainedQuantize._quantize_block_fp8(original, (128, 128), ue8m0=False)
                self.assertEqual(weight.shape, (rows, cols))
                self.assertEqual(weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(scale.shape, (-(-rows // 128), -(-cols // 128)))

                recovered = FineGrainedDequantize(quantizer)._dequantize_one(
                    weight, scale.float(), output_dtype=torch.float32
                )
                rel = ((recovered - original).norm() / original.norm()).item()
                self.assertLess(rel, 3e-2, f"{(rows, cols)}: {rel:.4f}")

    def test_block_fp8_round_trips_within_its_floor(self):
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        for scale_fmt in ("float", "ue8m0"):
            with self.subTest(scale_fmt=scale_fmt), mock.patch.object(fg, "is_sm100", return_value=False):
                experts = fg.FineGrainedExperts(cfg, block_size=(128, 128), weight_format="fp8", scale_fmt=scale_fmt)
                model = torch.nn.Module()
                model.experts = experts
                original = torch.randn(4, 256, 256) * 0.02
                out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.gate_up_proj": original}, model=model)
                weight, scale = out["experts.gate_up_proj"], out["experts.gate_up_proj_scale_inv"]
                self.assertEqual(
                    (weight.dtype, weight.shape), (experts.gate_up_proj.dtype, experts.gate_up_proj.shape)
                )
                self.assertEqual(
                    (scale.dtype, scale.shape),
                    (experts.gate_up_proj_scale_inv.dtype, experts.gate_up_proj_scale_inv.shape),
                )
                deq = FineGrainedDequantize(None)._dequantize_one(weight, scale.float(), output_dtype=torch.float32)
                rel = ((deq - original).norm() / original.norm()).item()
                self.assertLess(rel, 4e-2, f"{scale_fmt}: {rel:.3f}")

    def test_ue8m0_quantizes_against_the_rounded_scale(self):
        # ``scale_fmt="ue8m0"`` rounds weight_scale_inv to a power of two, and the weight has to be
        # divided by that rounded value or the dequantized block is off by up to an octave
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        quantizer = SimpleNamespace(
            quantization_config=SimpleNamespace(weight_block_size=(128, 128), scale_fmt="ue8m0")
        )
        torch.manual_seed(0)
        weight = torch.randn(128, 128, dtype=torch.float32)

        # no model to ask, so the op takes the block and the scale format from the config
        quantized = FineGrainedQuantize(quantizer)._quantize_one("layer.weight", weight, None)
        recovered = FineGrainedDequantize(quantizer)._dequantize_one(
            quantized["layer.weight"], quantized["layer.weight_scale_inv"].float(), output_dtype=torch.float32
        )
        rel_err = ((recovered - weight).abs().sum() / weight.abs().sum()).item()
        self.assertLess(rel_err, 5e-2)  # ~2e-2 for a round trip; a scale mismatch inflates it past 0.2

    def test_float_scale_fmt_stays_bit_identical_to_its_formula(self):
        """The inverse scale is `amax / MAX` and the weight is DIVIDED by it. The algebraically
        equal `weight * (MAX / amax)` is not bit-equal — it moves 29 of these 100 blocks' scales —
        so the divergence only shows across many blocks, on particular fp32 maxima."""
        from transformers.integrations.finegrained.conversions import (
            _FP8_DTYPE,
            _FP8_MAX,
            _FP8_MIN,
            FineGrainedQuantize,
        )

        quantizer = FineGrainedQuantize(
            SimpleNamespace(quantization_config=SimpleNamespace(weight_block_size=(128, 128), scale_fmt="float"))
        )
        for seed in range(100):
            torch.manual_seed(seed)
            weight = torch.randn(128, 128, dtype=torch.float32)
            out = quantizer._quantize_one("layer.weight", weight, None)
            ref_inv = (weight.abs().amax() / _FP8_MAX).to(torch.float32).reshape(1, 1)
            ref_q = torch.clamp(weight / ref_inv, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
            self.assertTrue(torch.equal(out["layer.weight"], ref_q), f"float weight diverged at seed {seed}")
            self.assertTrue(
                torch.equal(out["layer.weight_scale_inv"], ref_inv), f"float scale diverged at seed {seed}"
            )


@require_torch
class FineGrainedModeloptConverterTest(unittest.TestCase):
    """The ModelOpt (NVFP4) weight conversions. `MergeModulelist` is load-bearing beyond the
    merge itself: `core_model_loading` stamps a source with its expert index ONLY when one is
    present in the chain, and expert parallelism selects experts by that index. Without it every
    rank collects all E globals and the forward asserts on the per-expert count."""

    def _modelopt_conversions(self, **cfg_kwargs):
        from transformers.quantizers.finegrained.nvfp4 import FineGrainedNvfp4HfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        cfg = FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4", **cfg_kwargs)
        return FineGrainedNvfp4HfQuantizer(cfg).get_weight_conversions()

    def test_every_declared_target_is_a_slot_the_module_holds(self):
        """A converter target that no module holds is a load failure, and the activation global is
        exactly such a target under a weight-only run: the experts allocate one only when they
        quantize activations. Both runs are checked against a real module's parameter names, since
        the declarations are the only thing standing between the two."""
        cfg = _Cfg()
        for activation_format, holds_global in ((None, True), ("bf16", False)):
            experts = FineGrainedExperts(
                cfg, block_size=(4, 4), weight_format="nvfp4", activation_format=activation_format
            )
            slots = {name for name, _ in experts.named_parameters()} | set(experts._buffers)
            self.assertEqual(
                any("input_global_scale" in slot for slot in slots),
                holds_global,
                f"module with activation_format={activation_format!r} disagrees about the slot",
            )
            targets = {
                t.split("experts.")[-1].rstrip("$")
                for c in self._modelopt_conversions(activation_format=activation_format)
                for t in _targets(c)
            }
            for target in targets:
                self.assertIn(target, slots, f"activation_format={activation_format!r} converts onto a missing slot")

    def test_global_scale_converters_carry_an_expert_index(self):
        from transformers.core_model_loading import MergeModulelist

        # the per-expert layout only (`experts.*.`): a checkpoint that ships one already-stacked
        # tensor per layer has no per-expert sources to stamp — it is sharded like the fused
        # expert weight next to it
        globals_converters = [
            c
            for c in self._modelopt_conversions()
            if any("global_scale" in t for t in _targets(c))
            and any(re.search(p, "model.layers.0.mlp.experts.7.gate_proj.weight_scale_2") for p in c.source_patterns)
        ]
        self.assertTrue(globals_converters, "no global-scale converter found")
        for conv in globals_converters:
            self.assertTrue(
                any(isinstance(op, MergeModulelist) for op in conv.operations),
                f"{_targets(conv)} has no MergeModulelist, so expert parallelism cannot select "
                "experts and every rank would collect all of them",
            )

    def test_both_modelopt_scale_layouts_have_converters(self):
        """modelopt ships the scales either as one tensor per expert per projection (GLM-5.2) or
        as one already-stacked tensor per layer (the fused vLLM layout). Both are
        converted; the block scale takes the module's layout ops either way."""
        sources = {p for c in self._modelopt_conversions() for p in c.source_patterns}
        for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
            per_expert = f"model.layers.0.mlp.experts.7.gate_proj.{suffix}"
            fused = f"model.layers.0.mlp.experts.gate_up_proj_{suffix}"
            self.assertTrue(any(re.search(p, per_expert) for p in sources), f"no per-expert converter for {suffix}")
            self.assertTrue(any(re.search(p, fused) for p in sources), f"no fused converter for {suffix}")
            # the per-expert patterns are REGEXES: an unescaped dot also matches `_`, so
            # `mlp.experts.*.up_proj.weight_scale_2` would claim the fused key as well
            claimed = [p for p in sources if re.search(p, fused) and re.search(p, per_expert)]
            self.assertEqual(claimed, [], f"one pattern claims BOTH layouts for {suffix}: {claimed}")

    def test_a_checkpoint_key_is_claimed_by_exactly_one_converter(self):
        """Two converters for one key is not additive — the later one WINS and silently drops
        whatever ops the first carried. That is how a fused modelopt checkpoint lost its packed
        uint8 -> int8 view (a catch-all duplicated the fused converter) and how the per-expert
        patterns, which are REGEXES whose unescaped dots also match `_`, claimed the fused
        globals and flattened the gate|up pair instead of folding it."""
        import re

        from transformers.quantizers.finegrained.base import FineGrainedHfQuantizer
        from transformers.utils.quantization_config import FineGrainedConfig

        for quant_kwargs in (
            {"quant_method": "fp8"},
            {"quant_method": "mxfp4"},
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        ):
            quantizer = FineGrainedHfQuantizer(FineGrainedConfig(**quant_kwargs))
            quantizer.pre_quantized = True
            converters = quantizer.update_weight_conversions([])
            for key in (
                "model.layers.3.mlp.experts.gate_up_proj",
                "model.layers.3.mlp.experts.down_proj",
                "model.layers.3.mlp.experts.gate_up_proj_weight_scale_2",
                "model.layers.3.mlp.experts.7.up_proj.weight_scale_2",
            ):
                claimed = [
                    c
                    for c in converters
                    for pattern in (c.source_patterns if isinstance(c.source_patterns, list) else [c.source_patterns])
                    if re.search(str(pattern), key)
                ]
                with self.subTest(quant=quant_kwargs["quant_method"], key=key.rsplit(".", 1)[1]):
                    self.assertLessEqual(
                        len(claimed),
                        1,
                        f"{len(claimed)} converters claim this key; the last one wins and drops "
                        f"the others' ops: {[[type(o).__name__ for o in c.operations] for c in claimed]}",
                    )

    def test_a_calibrated_checkpoint_s_input_scale_reaches_the_module_s_slot(self):
        """A static checkpoint calibrates one `input_scale` per quantized module, and each key
        must reach the slot its module holds, in that module's shape: one value on a dense linear,
        one per expert on the stacked experts, and the gate|up pair reduced to one per expert
        rather than flattened to 2E. Unrouted, the module keeps its registered default of 1.0 and
        every activation quantizes against the wrong scale, silently."""
        import re

        from transformers.utils.quantization_config import FineGrainedConfig

        quantizer = _quantizer_for(FineGrainedConfig(quant_method="fp8", activation_scheme="static"))
        quantizer.pre_quantized = True
        converters = quantizer.update_weight_conversions([])

        def claim(key):
            return [
                c
                for c in converters
                for pattern in (c.source_patterns if isinstance(c.source_patterns, list) else [c.source_patterns])
                if re.search(str(pattern), key)
            ]

        for key, expected in (
            ("model.layers.3.self_attn.q_proj.input_scale", "model.layers.3.self_attn.q_proj.activation_scale"),
            ("model.layers.3.mlp.experts.7.gate_proj.input_scale", "mlp.experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.7.up_proj.input_scale", "mlp.experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.7.down_proj.input_scale", "mlp.experts.down_proj_activation_scale"),
            ("model.layers.3.mlp.experts.gate_up_proj_input_scale", "experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.down_proj_input_scale", "experts.down_proj_activation_scale"),
        ):
            with self.subTest(key=key.rsplit(".", 2)[-2]):
                claimed = claim(key)
                self.assertEqual(len(claimed), 1, f"{len(claimed)} converters claim {key}")
                # a renaming's target carries a backreference; resolve it against the key
                targets = [
                    re.sub(str(claimed[0].source_patterns[0]), str(t), key) if "\\1" in str(t) else str(t)
                    for t in claimed[0]._original_target_patterns
                ]
                self.assertTrue(any(expected in t for t in targets), f"{key} -> {targets}")
        # a dynamic checkpoint has no such key to route, and a weight-only one ignores it
        dynamic = _quantizer_for(FineGrainedConfig(quant_method="fp8"))
        dynamic.pre_quantized = True
        self.assertEqual(dynamic.get_weight_conversions(), [])

    def test_the_calibrated_expert_scales_reduce_per_expert_not_per_tensor(self):
        """`FineGrainedInputScales` collapses the gate_up to ONE value for the NVFP4 global — the
        global is a split of the block scale, so an inflated one is exact. A static scale IS the
        quantization scale, with no block level to absorb it, so collapsing it would quantize
        every expert against the largest one's range. It stays per expert."""
        import torch

        from transformers.integrations.finegrained.conversions import FineGrainedInputScales

        sources = {
            "mlp.experts.*.gate_proj.input_scale": torch.tensor([1.0, 3.0]),
            "mlp.experts.*.up_proj.input_scale": torch.tensor([2.0, 1.0]),
        }
        target = "mlp.experts.gate_up_proj_activation_scale"
        scale = FineGrainedInputScales().convert(dict(sources), full_layer_name=target)[target]
        torch.testing.assert_close(scale, torch.tensor([2.0, 3.0]))  # per expert, over the pair

    def test_the_gate_up_fold_survives_the_loader_s_list_wrapping(self):
        """The loader hands a converter its tensors in a list. Unwrapped, a fused `(E, 2)` pair
        reads as `(1, 2E)`, the per-half fold does not fire, and the module gets `2E` globals
        where the kernels assert one per expert."""
        import torch

        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
            "experts.down_proj_input_global_scale",
        ]
        experts = 8
        for wrap in (False, True):
            pair = torch.rand(experts, 2) + 1
            src = {
                "experts.gate_up_proj_weight_scale_2": [pair] if wrap else pair,
                "experts.down_proj_weight_scale_2": torch.rand(experts) + 1,
                "experts.down_proj_input_scale": torch.rand(experts) + 1,
            }
            out = FineGrainedWeightGlobals(None).convert(src, target_patterns=targets, model=None)
            with self.subTest(list_wrapped=wrap):
                self.assertEqual(
                    out["experts.gate_up_proj_weight_global_scale"].shape,
                    torch.Size([experts]),
                    "the gate|up halves were not folded to one global per expert",
                )

    def test_one_converter_merges_a_layer_s_globals(self):
        """modelopt calibrates the gate|up halves separately. One converter owns every global of
        a layer, because merging them to the one-per-expert the kernels take moves the up half's
        onto the down projection: SwiGLU is linear in the up half, so the stack keeps the gate's
        global and the down's weight global scales the expert output back. The down's calibrated
        input scale moves the other way, keeping the requantized intermediate on the range the
        checkpoint calibrated."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        gate, up = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        down, down_input = torch.tensor([5.0, 6.0]), torch.tensor([0.5, 2.0])
        targets = [
            "mlp.experts.gate_up_proj_weight_global_scale",
            "mlp.experts.down_proj_weight_global_scale",
            "mlp.experts.down_proj_input_global_scale",
        ]
        out = FineGrainedWeightGlobals().convert(
            {
                "mlp.experts.*.up_proj.weight_scale_2": up,
                "mlp.experts.*.gate_proj.weight_scale_2": gate,
                "mlp.experts.*.down_proj.weight_scale_2": down,
                "mlp.experts.*.down_proj.input_scale": down_input,
            },
            target_patterns=targets,
        )
        ratio = up / gate
        torch.testing.assert_close(out[targets[0]], gate)
        torch.testing.assert_close(out[targets[1]], down * ratio)
        torch.testing.assert_close(out[targets[2]], down_input / ratio)

    def test_globals_with_one_calibrated_projection_pass_through(self):
        """A checkpoint that calibrates the stack as one matrix (its halves agree, or it ships a
        single global) has nothing to merge, so every global reaches its module unchanged."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
            "experts.down_proj_input_global_scale",
        ]
        out = FineGrainedWeightGlobals().convert(
            {
                "experts.gate_up_proj_weight_scale_2": torch.tensor([[1.0], [2.0]]),
                "experts.down_proj_weight_scale_2": torch.tensor([3.0, 4.0]),
                "experts.down_proj_input_scale": torch.tensor([0.5, 0.25]),
            },
            target_patterns=targets,
        )
        torch.testing.assert_close(out[targets[0]], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out[targets[1]], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(out[targets[2]], torch.tensor([0.5, 0.25]))

    def test_the_fused_layout_ships_both_halves_in_one_tensor(self):
        """A stacked `(E, 2)` `weight_scale_2` is the same pair as the per-expert layout's two
        keys, and merges the same way."""
        from transformers.integrations.finegrained.conversions import FineGrainedWeightGlobals

        targets = [
            "experts.gate_up_proj_weight_global_scale",
            "experts.down_proj_weight_global_scale",
        ]
        out = FineGrainedWeightGlobals().convert(
            {
                "experts.gate_up_proj_weight_scale_2": torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
                "experts.down_proj_weight_scale_2": torch.tensor([5.0, 6.0]),
            },
            target_patterns=targets,
        )
        torch.testing.assert_close(out[targets[0]], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out[targets[1]], torch.tensor([5.0 * 3.0, 6.0 * 2.0]))

    def test_activation_global_is_one_value_for_the_gate_up_and_per_expert_for_the_down(self):
        """The gate_up quantizes the hidden states once, BEFORE routing, so its calibrated
        `input_scale` reduces to one value; the down's rows are per expert, so its stays per
        expert — the requant epilogue normalizes each row by its own expert's value."""
        from transformers.integrations.finegrained.conversions import FineGrainedInputScales

        up = FineGrainedInputScales().convert(
            {
                "mlp.experts.*.gate_proj.input_scale": torch.tensor([1.0, 3.0]),
                "mlp.experts.*.up_proj.input_scale": torch.tensor([2.0, 1.0]),
            },
            full_layer_name="mlp.experts.gate_up_proj_input_global_scale",
        )["mlp.experts.gate_up_proj_input_global_scale"]
        torch.testing.assert_close(up, torch.tensor([3.0]))
        down = FineGrainedInputScales().convert(
            {"mlp.experts.*.down_proj.input_scale": torch.tensor([1.0, 4.0])},
            full_layer_name="mlp.experts.down_proj_input_global_scale",
        )["mlp.experts.down_proj_input_global_scale"]
        torch.testing.assert_close(down, torch.tensor([1.0, 4.0]))


@require_torch
class FineGrainedLoaderChainTest(unittest.TestCase):
    """A core conversion chain and an on-the-fly quantization in the same load: the permute has to
    run on the full-precision tensor, and only the finegrained module of the three gets a scale."""

    def test_a_chunked_rope_permute_quantizes_only_the_finegrained_target(self):
        from transformers.integrations.finegrained import FineGrainedLinear
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        n_heads = 2
        head_dim = 4
        in_dim = 4
        out_dim = n_heads * head_dim
        block_size = (4, 4)

        class RopeProjector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim))

        class RopeSelfAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = FineGrainedLinear(in_dim, out_dim, block_size=block_size, weight_format="fp8")
                self.k_proj = RopeProjector()
                self.v_proj = RopeProjector()

        class RopeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = RopeSelfAttn()

        class RopeModel(PreTrainedModel):
            base_model_prefix = "model"

            def __init__(self, config):
                super().__init__(config)
                self.layers = torch.nn.ModuleList([RopeLayer()])
                self.post_init()

            def _init_weights(self, module):
                pass  # every weight here comes from the state dict, and `normal_` has no fp8 kernel

        config = PreTrainedConfig()
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
        state_dict = {"layers.0.self_attn.qkv_proj.weight": raw_qkv.clone()}

        quantizer_cls = type(
            "FineGrainedHfQuantizer",
            (),
            {
                "__init__": lambda self, bs=block_size: setattr(
                    self, "quantization_config", SimpleNamespace(weight_block_size=bs, scale_fmt="float")
                ),
                "param_needs_quantization": lambda self, _model, param_name: param_name.endswith("q_proj.weight"),
                "get_quantize_ops": lambda self: FineGrainedQuantize(self),
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
                operations=[Chunk(dim=0), PermuteForRope(permute_layer_names=["q_proj", "k_proj"])],
            )
        ]
        load_config = LoadStateDictConfig(weight_mapping=weight_mapping, hf_quantizer=quantizer)
        loading_info, _ = convert_and_load_state_dict_in_model(model, state_dict, load_config)

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.unexpected_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})

        permute_op = PermuteForRope(permute_layer_names=["q_proj", "k_proj"])
        permute_op.config = model.config
        expected_q = permute_op._apply(raw_q)
        expected_k = permute_op._apply(raw_k)
        expected_v = raw_v  # V carries no RoPE, so the permute names only q and k

        model_state = model.state_dict()
        self.assertFalse(torch.allclose(raw_k, expected_k))
        torch.testing.assert_close(model_state["layers.0.self_attn.k_proj.weight"], expected_k)
        torch.testing.assert_close(model_state["layers.0.self_attn.v_proj.weight"], expected_v)

        q_weight_key = "layers.0.self_attn.q_proj.weight"
        scale_key = "layers.0.self_attn.q_proj.weight_scale_inv"
        self.assertIn(scale_key, model_state)
        self.assertEqual(model_state[q_weight_key].dtype, torch.float8_e4m3fn)
        self.assertEqual(model_state[q_weight_key].shape, torch.Size((out_dim, in_dim)))
        self.assertEqual(model_state[scale_key].dtype, torch.float32)
        self.assertEqual(
            model_state[scale_key].shape,
            torch.Size((out_dim // block_size[0], in_dim // block_size[1])),
        )

        dequantized_q = FineGrainedDequantize(None)._dequantize_one(
            model_state[q_weight_key], model_state[scale_key], output_dtype=torch.float32
        )
        torch.testing.assert_close(dequantized_q, expected_q, rtol=1e-2, atol=1e-2)
