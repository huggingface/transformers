"""The `finegrained` conversions. The ops first, which are scheme-agnostic (quantize, the NVFP4
global scales); then one class per scheme that loads a checkpoint in that scheme's real layout
through the real quantizer onto a dummy model, checks what the modules hold, and saves it back —
with each scheme's quirks. Pure torch: no kernel is loaded and no GPU is needed."""

import re
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

import transformers.integrations.finegrained.core as fg
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    MergeModulelist,
    PermuteForRope,
    WeightConverter,
    convert_and_load_state_dict_in_model,
    revert_weight_conversion,
)
from transformers.integrations.finegrained import FineGrainedExperts
from transformers.integrations.moe import use_experts_implementation
from transformers.modeling_utils import LoadStateDictConfig
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import FineGrainedConfig

from ...kernels.test_finegrained import fake_finegrained_kernel, patch_finegrained_kernel
from .test_core import _Cfg, _quantizer_for, _targets


E, H, I = 2, 64, 32  # experts, hidden, intermediate: one 32-group along each input


@use_experts_implementation
class _Experts(nn.Module):
    """A model's unquantized experts, which the quantizer replaces with `FineGrainedExperts`."""

    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.empty(E, H, I))


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(H, H, bias=False)
        self.mlp = nn.Module()
        self.mlp.experts = _Experts(config)


class _MoeModel(PreTrainedModel):
    """One decoder layer: a dense `q_proj` and a MoE, the two module kinds a checkpoint quantizes."""

    base_model_prefix = "model"
    config: PreTrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(config)])
        self.post_init()

    def _init_weights(self, module):
        pass  # every weight here comes from the state dict


def _model_mapping():
    """The per-expert merge a Qwen3-MoE / DeepSeek model declares for its experts."""
    return [
        WeightConverter(
            ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
            "mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            "mlp.experts.*.down_proj.weight", "mlp.experts.down_proj", operations=[MergeModulelist(dim=0)]
        ),
    ]


def _load_and_save(checkpoint, quantization_config):
    """`checkpoint` through the real quantizer's chain, as `from_pretrained` runs it (off SM100, so the
    scales stay affine and no kernel is needed), then saved back. Returns `(model, loading_info, saved)`."""
    config = PreTrainedConfig()
    config.hidden_size, config.intermediate_size, config.num_local_experts, config.hidden_act = H, I, E, "silu"
    quantizer = _quantizer_for(quantization_config)
    quantizer.pre_quantized = True
    model = _MoeModel(config)
    with mock.patch.object(fg, "is_sm100", return_value=False):
        quantizer._process_model_before_weight_loading(model)
        load_config = LoadStateDictConfig(
            weight_mapping=quantizer.update_weight_conversions(_model_mapping()), hf_quantizer=quantizer
        )
        info, _ = convert_and_load_state_dict_in_model(
            model, {k: v.clone() for k, v in checkpoint.items()}, load_config
        )
        model.hf_quantizer = quantizer
        saved = revert_weight_conversion(model, model.state_dict())
    return model, info, saved


def _raw(tensor):
    return tensor.reshape(-1).view(torch.uint8) if tensor.element_size() == 1 else tensor


def _per_expert(make):
    """A per-expert checkpoint layout: `make(n, k)` gives one projection's `{suffix: tensor}`."""
    checkpoint = {}
    for expert in range(E):
        for proj, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            for suffix, tensor in make(*shape).items():
                checkpoint[f"model.layers.0.mlp.experts.{expert}.{proj}.{suffix}"] = tensor
    for suffix, tensor in make(H, H).items():
        checkpoint[f"model.layers.0.self_attn.q_proj.{suffix}"] = tensor
    return checkpoint


class _ChainAssertions:
    def assertLoadsCleanly(self, info):
        self.assertEqual(
            (info.missing_keys, info.unexpected_keys, info.mismatched_keys, info.conversion_errors),
            (set(), set(), set(), {}),
        )

    def assertSavesBack(self, saved, checkpoint):
        """The save writes the checkpoint's own keys, dtypes and bytes."""
        self.assertEqual(sorted(saved), sorted(checkpoint))
        for key, tensor in checkpoint.items():
            with self.subTest(key=key):
                self.assertEqual((saved[key].dtype, saved[key].shape), (tensor.dtype, tensor.shape))
                self.assertTrue(torch.equal(_raw(saved[key]), _raw(tensor)))

    def assertHoldsInterleaved(self, experts, checkpoint, suffix, held):
        """The experts hold gate|up stacked and row-interleaved: `[g0, u0, g1, u1, ...]`."""
        self.assertTrue(experts.holds_interleaved_gate_up)
        for expert in range(E):
            gate = checkpoint[f"model.layers.0.mlp.experts.{expert}.gate_proj.{suffix}"]
            up = checkpoint[f"model.layers.0.mlp.experts.{expert}.up_proj.{suffix}"]
            expected = torch.stack([_raw(gate).view(gate.shape), _raw(up).view(up.shape)], dim=1)
            got = _raw(held[expert]).view(held[expert].shape)
            self.assertTrue(torch.equal(got, expected.reshape(-1, gate.shape[-1])), f"expert {expert} {suffix}")


@require_torch
class FineGrainedQuantizeOpTest(unittest.TestCase):
    """`FineGrainedQuantize` reads the module's format: block-FP8 in torch (the group formats run
    the kernels' own quantizers, which need a GPU), emitted in the module's layout."""

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
class FineGrainedLayoutOpsTest(unittest.TestCase):
    """The module HOLDS its block scales swizzled (a 5-D Parameter) when its chain runs the tcgen05
    scaled-MMA under a triton dispatch on SM100 — keyed off the declared weight format, not the
    scale dtype (V4-style block-FP8 ships UE8M0 scales) — and affine otherwise; the loader's
    `FineGrainedSwizzleScales` op fills that layout and its reverse restores the checkpoint grid."""

    E, HIDDEN, INTER = 4, 256, 128  # gate_up rows 2*INTER = 256 and down rows HIDDEN = 256: whole 128-row blocks

    def _experts(self, weight_format, activation_format=None, impl="grouped_mm", sm100=True):
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = self.HIDDEN, self.INTER, self.E
        cfg._experts_implementation = impl
        with (
            mock.patch.object(fg, "is_sm100", return_value=sm100),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            experts = fg.FineGrainedExperts(
                cfg, block_size=(128, 128), weight_format=weight_format, activation_format=activation_format
            )
        model = torch.nn.Module()
        model.config = cfg
        model.experts = experts
        return model, experts

    def test_module_holds_swizzled_scales_for_the_scaled_mma_chain(self):
        _, experts = self._experts("mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 2, 2, 2, 256))  # (E, 256/128, (256/32)/4, 2, 256)
        self.assertEqual(experts.down_proj_scale_inv.shape, (4, 2, 1, 2, 256))
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)
        self.assertTrue(experts.holds_interleaved_gate_up)
        # every MX group-32 format holds UE8M0 scales whatever `scale_fmt` says (the kernels reject a
        # float32 grid there; GPT-OSS's config has no scale_fmt at all)
        _, experts = self._experts("mxfp4", activation_format="bf16")
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)

    def test_affine_off_sm100_weight_only_deepgemm_and_block_fp8(self):
        for kw in (
            {"weight_format": "mxfp8", "sm100": False},
            {"weight_format": "mxfp4", "activation_format": "bf16"},  # W4A16 reads scales per group affinely
            {"weight_format": "mxfp8", "impl": "deepgemm"},  # the DeepGEMM backends read affine scales
            {"weight_format": "nvfp4", "impl": "deepgemm_megamoe"},
        ):
            with self.subTest(**kw):
                _, experts = self._experts(**kw)
                self.assertEqual(
                    experts.gate_up_proj_scale_inv.shape,
                    (4, 256, 256 // (16 if kw["weight_format"] == "nvfp4" else 32)),
                )
        # block-FP8 has no per-group scale grid to swizzle: one scalar per (N/128, K/128) block
        _, experts = self._experts("fp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 2, 2))
        with mock.patch.object(fg, "_holds_swizzled_scales", return_value=False):
            _, experts = self._experts("mxfp8")
        self.assertEqual(experts.gate_up_proj_scale_inv.shape, (4, 256, 8))

    def test_megamoe_holds_gate_up_stacked(self):
        _, experts = self._experts("fp8", impl="deepgemm_megamoe")
        self.assertFalse(experts.holds_interleaved_gate_up)

    def _op_kernel(self):
        kernel, _ = fake_finegrained_kernel()
        kernel.swizzle_mx_scales = mock.Mock(
            side_effect=lambda s: s.reshape(s.shape[0], s.shape[1] // 128, -1, 2, 256)
        )
        kernel.unswizzle_mx_scales = mock.Mock(
            side_effect=lambda s, rows, cols, num_experts=None: s.reshape(num_experts, rows, cols)
        )
        return kernel

    def test_swizzle_op_fills_only_the_scales_the_module_holds_swizzled(self):
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

        model, experts = self._experts("mxfp8")
        kernel = self._op_kernel()
        op = FineGrainedSwizzleScales(hf_quantizer=None)
        grid = torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu)
        weight = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            # the loader hands a single target under its pattern, the full name (with suffix) alongside
            out = op.convert(
                {"mlp.experts.gate_up_proj": grid}, model=model, full_layer_name="experts.gate_up_proj_scale_inv"
            )
            self.assertEqual(out["mlp.experts.gate_up_proj"].shape, (4, 2, 2, 2, 256))
            # the same converter's weight passes through
            out = op.convert({"mlp.experts.gate_up_proj": weight}, model=model, full_layer_name="experts.gate_up_proj")
            self.assertIs(out["mlp.experts.gate_up_proj"], weight)
            # a multi-tensor converter (GPT-OSS deserialize) names its outputs fully
            out = op.convert(
                {
                    "experts.down_proj": weight,
                    "experts.down_proj_scale_inv": torch.zeros(4, 256, 4, dtype=torch.float8_e8m0fnu),
                },
                model=model,
                full_layer_name="experts.down_proj",
            )
            self.assertEqual(out["experts.down_proj_scale_inv"].shape, (4, 2, 1, 2, 256))
            self.assertIs(out["experts.down_proj"], weight)
            # a module that holds affine scales keeps them
            affine_model, _ = self._experts("mxfp4", activation_format="bf16")
            out = op.convert(
                {"mlp.experts.gate_up_proj": grid},
                model=affine_model,
                full_layer_name="experts.gate_up_proj_scale_inv",
            )
            self.assertIs(out["mlp.experts.gate_up_proj"], grid)
        self.assertEqual(kernel.swizzle_mx_scales.call_count, 2)

    def test_scale_container_op_round_trips_every_container(self):
        """dsv4-flash-base ships UE8M0 scales as float32 values, MiniMax as uint8 exponent bytes:
        the op brings both into the held e8m0 (exact cast / same bytes), records the container on
        the module, and its reverse restores that container on save; weights and native scales
        pass both ways."""
        from transformers.integrations.finegrained.conversions import FineGrainedScaleContainer

        op = FineGrainedScaleContainer(hf_quantizer=None)
        native = torch.pow(2.0, torch.randint(-8, 8, (4, 256, 8)).float()).to(torch.float8_e8m0fnu)
        for container, shipped in ((torch.float32, native.float()), (torch.uint8, native.view(torch.uint8))):
            with self.subTest(container=container):
                model, experts = self._experts("mxfp8", sm100=False)  # affine e8m0 scale Parameters
                out = op.convert(
                    {"mlp.experts.gate_up_proj": shipped},
                    model=model,
                    full_layer_name="experts.gate_up_proj_scale_inv",
                )
                held = out["mlp.experts.gate_up_proj"]
                self.assertEqual(held.dtype, torch.float8_e8m0fnu)
                self.assertTrue(torch.equal(held.view(torch.uint8), native.view(torch.uint8)))
                self.assertIs(experts.scale_container_dtype, container)
                back = op.reverse_op.convert({"x": held}, model=model)["x"]
                self.assertEqual(back.dtype, container)
                self.assertTrue(torch.equal(back, shipped))
                weight = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
                self.assertIs(op.reverse_op.convert({"w": weight}, model=model)["w"], weight)
        model, experts = self._experts("mxfp8", sm100=False)
        self.assertIs(
            op.convert({"mlp.experts.down_proj": native}, model=model, full_layer_name="experts.down_proj_scale_inv")[
                "mlp.experts.down_proj"
            ],
            native,
        )
        self.assertIsNone(experts.scale_container_dtype)
        self.assertIs(op.reverse_op.convert({"x": native}, model=model)["x"], native)  # nothing recorded: native stays

    def test_reverse_op_restores_the_affine_grid_from_the_artifact_alone(self):
        from transformers.integrations.finegrained.conversions import FineGrainedSwizzleScales

        kernel = self._op_kernel()
        reverse = FineGrainedSwizzleScales(hf_quantizer=None).reverse_op
        artifact = torch.zeros(4, 2, 2, 2, 256, dtype=torch.float8_e8m0fnu)
        affine = torch.zeros(4, 256, 8, dtype=torch.float8_e8m0fnu)
        p1, p2, p3 = patch_finegrained_kernel(kernel)
        with p1, p2, p3:
            out = reverse.convert({"experts.gate_up_proj_scale_inv": artifact}, model=None, full_layer_name="x")
            self.assertEqual(out["experts.gate_up_proj_scale_inv"].shape, (4, 256, 8))
            kernel.unswizzle_mx_scales.assert_called_once_with(artifact, 256, 8, num_experts=4)
            out = reverse.convert({"experts.gate_up_proj_scale_inv": affine}, model=None, full_layer_name="x")
            self.assertIs(out["experts.gate_up_proj_scale_inv"], affine)

    def _quantizer(self, quant_method):
        from transformers.utils.quantization_config import FineGrainedConfig

        return _quantizer_for(FineGrainedConfig(quant_method=quant_method)).update_weight_conversions

    def test_blocks_scales_converters_carry_the_layout_ops(self):
        from transformers.core_model_loading import WeightConverter
        from transformers.integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedSwizzleScales,
        )

        converters = self._quantizer("mxfp4")([])
        by_target = {_targets(c)[0]: c for c in converters if isinstance(c, WeightConverter)}
        ops = by_target["gate_up_proj"].operations
        self.assertIsInstance(ops[-2], FineGrainedInterleaveGateUp)
        self.assertIsInstance(ops[-1], FineGrainedSwizzleScales)
        self.assertIsInstance(by_target["gate_up_proj_scale_inv"].operations[0], FineGrainedInterleaveGateUp)
        self.assertIsInstance(by_target["down_proj"].operations[-1], FineGrainedSwizzleScales)
        self.assertIsInstance(by_target["down_proj_scale_inv"].operations[-1], FineGrainedSwizzleScales)

    def test_catch_all_converters_deliver_the_full_parameter_name(self):
        """A key already under the fused name (GPT-OSS's gate_up_proj_bias, a renamed MiniMax
        down_proj_scale_inv) goes through the real converter plumbing: the ops receive the tensor
        under the source PATTERN and must hand back the target, which the loader expands to the
        full name."""
        from transformers.core_model_loading import WeightConverter, rename_source_key

        model, _ = self._experts("mxfp8")
        converters = [c for c in self._quantizer("mxfp8")([]) if isinstance(c, WeightConverter)]
        for key, shape in (("experts.gate_up_proj_bias", (4, 256)), ("experts.down_proj_scale_inv", (4, 256, 4))):
            renamed, source_pattern = rename_source_key(key, [], converters)
            self.assertEqual(renamed, key)
            converter = next(c for c in converters if source_pattern in c.source_patterns)
            tensor = torch.zeros(shape, dtype=torch.float8_e8m0fnu if "scale" in key else torch.float32)
            converter.add_tensor(renamed, key, source_pattern, lambda t=tensor: t)
            kernel = self._op_kernel()
            p1, p2, p3 = patch_finegrained_kernel(kernel)
            with p1, p2, p3:
                out = converter.convert(renamed, model=model)
            self.assertEqual(list(out), [key])

    def test_replacement_keeps_the_models_gate_up_convention(self):
        """`replace_with_finegrained_layer` goes through `use_experts_implementation`, which stamps
        the layout flags on the instance after __init__ — GPT-OSS's `is_concatenated=False` must
        survive it (it decides whether the loader interleaves)."""
        from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
        from transformers.utils.quantization_config import FineGrainedConfig

        gpt_oss = GptOssConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=32,
            num_local_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            sliding_window=16,
        )
        qwen = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
        for cls, cfg, quant_method, expected in (
            (GptOssForCausalLM, gpt_oss, "mxfp4", False),
            (Qwen3MoeForCausalLM, qwen, "mxfp8", True),
        ):
            with self.subTest(model=cls.__name__), torch.device("meta"):
                model = cls(cfg)
                fg.replace_with_finegrained_layer(
                    model, modules_to_not_convert=[], quantization_config=FineGrainedConfig(quant_method=quant_method)
                )
                experts = model.model.layers[0].mlp.experts
                self.assertIsInstance(experts, fg.FineGrainedExperts)
                self.assertIs(experts.is_concatenated, expected)

    def test_interleave_op_follows_the_model_and_the_experts_backend(self):
        """The op reads the model's experts modules: a model whose own rows already come
        interleaved (`is_concatenated=False`, GPT-OSS) keeps them, Mega MoE packs gate|up itself
        and keeps the stack; every other case gets the interleaved order, and the reverse restores
        the stack."""
        from transformers.integrations.finegrained.conversions import FineGrainedInterleaveGateUp

        stacked = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)  # rows [g0,g1,g2,u0,u1,u2]
        op = FineGrainedInterleaveGateUp(hf_quantizer=None)
        model, experts = self._experts("mxfp8")
        out = op.convert({"mlp.experts.gate_up_proj": stacked}, model=model)["mlp.experts.gate_up_proj"]
        torch.testing.assert_close(out, stacked[:, [0, 3, 1, 4, 2, 5]])
        back = op.reverse_op.convert({"mlp.experts.gate_up_proj": out}, model=model)["mlp.experts.gate_up_proj"]
        torch.testing.assert_close(back, stacked)
        experts.is_concatenated = False
        self.assertIs(op.convert({"x": stacked}, model=model)["x"], stacked)
        experts.is_concatenated = True
        experts.holds_interleaved_gate_up = False
        self.assertIs(op.convert({"x": stacked}, model=model)["x"], stacked)


@require_torch
class FineGrainedGlobalScaleOpsTest(unittest.TestCase):
    """The NVFP4 second-level globals and calibrated input scales, in the layout the kernels index:
    one per expert per projection, the gate|up pair folded, the gate_up activation global one value."""

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
class FineGrainedConversionPlanTest(unittest.TestCase):
    """What every scheme's conversion plan must hold whatever the layout."""

    def test_a_checkpoint_key_is_claimed_by_exactly_one_converter(self):
        """Two converters for one key is not additive — the later one WINS and silently drops
        whatever ops the first carried. That is how a fused modelopt checkpoint lost its packed
        uint8 -> int8 view (a catch-all duplicated the fused converter) and how the per-expert
        patterns, which are REGEXES whose unescaped dots also match `_`, claimed the fused
        globals and flattened the gate|up pair instead of folding it."""

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
                # the loader routes a key through each converter's compiled sources
                claimed = [c for c in converters if isinstance(c, WeightConverter) and c.compiled_sources.search(key)]
                with self.subTest(quant=quant_kwargs["quant_method"], key=key.rsplit(".", 1)[1]):
                    self.assertLessEqual(
                        len(claimed),
                        1,
                        f"{len(claimed)} converters claim this key; the last one wins and drops "
                        f"the others' ops: {[[type(o).__name__ for o in c.operations] for c in claimed]}",
                    )


@require_torch
class FineGrainedBlockFp8ChainTest(_ChainAssertions, unittest.TestCase):
    """Block-FP8 and per-tensor FP8, as DeepSeek-V3, DeepSeek-V4 and a static (Mistral) checkpoint ship them."""

    def test_deepseek_v3_block_fp8(self):
        """Per-expert e4m3 weights with fp32 `weight_scale_inv` per block, merged onto the stacked experts."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "weight_scale_inv": torch.rand(-(-n // 16), -(-k // 16)) + 0.5,
            }
        )
        model, info, saved = _load_and_save(
            checkpoint, FineGrainedConfig(quant_method="fp8", weight_block_size=(16, 16))
        )
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)

    def test_deepseek_v4_ue8m0_scale_keys(self):
        """DeepSeek-V4 ships its block scales as E8M0 under `.scale`; they load as `weight_scale_inv`
        and save back under their own name."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "scale": torch.randint(120, 130, (-(-n // 16), -(-k // 16)), dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            }
        )
        config = FineGrainedConfig(quant_method="fp8", weight_block_size=(16, 16), scale_fmt="ue8m0")
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        self.assertEqual(model.model.layers[0].self_attn.q_proj.weight_scale_inv.dtype, torch.float8_e8m0fnu)
        self.assertSavesBack(saved, checkpoint)

    def test_mistral4_static_per_tensor(self):
        """Mistral-Small-4 ships its experts already stacked under the module's own names: e4m3, one
        BF16 weight scale per expert `(E, 1, 1)` and a calibrated BF16 activation scale per expert;
        its dense linears hold one 0-dim BF16 scale of each."""
        torch.manual_seed(0)
        checkpoint = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": (torch.rand(()) + 0.5).to(torch.bfloat16),
            "model.layers.0.self_attn.q_proj.activation_scale": (torch.rand(()) + 0.5).to(torch.bfloat16),
        }
        for proj, (rows, cols) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
            prefix = f"model.layers.0.mlp.experts.{proj}"
            checkpoint[prefix] = torch.randn(E, rows, cols).to(torch.float8_e4m3fn)
            checkpoint[f"{prefix}_scale_inv"] = (torch.rand(E, 1, 1) + 0.5).to(torch.bfloat16)
            checkpoint[f"{prefix}_activation_scale"] = (torch.rand(E) + 0.5).to(torch.bfloat16)
        config = FineGrainedConfig(quant_method="fp8", weight_block_size=None, activation_scheme="static")
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        torch.testing.assert_close(
            experts.gate_up_proj_activation_scale.float(),
            checkpoint["model.layers.0.mlp.experts.gate_up_proj_activation_scale"].float(),
        )
        self.assertSavesBack(saved, checkpoint)

    def test_a_calibrated_checkpoint_s_input_scale_reaches_the_module_s_slot(self):
        """A static checkpoint calibrates one `input_scale` per quantized module, and each key
        must reach the slot its module holds, in that module's shape: one value on a dense linear,
        one per expert on the stacked experts, and the gate|up pair reduced to one per expert
        rather than flattened to 2E. Unrouted, the module keeps its registered default of 1.0 and
        every activation quantizes against the wrong scale, silently."""

        from transformers.utils.quantization_config import FineGrainedConfig

        quantizer = _quantizer_for(FineGrainedConfig(quant_method="fp8", activation_scheme="static"))
        quantizer.pre_quantized = True
        converters = quantizer.update_weight_conversions([])

        def claim(key):
            # the loader routes a key through each converter's compiled sources, not the raw patterns
            return [c for c in converters if c.compiled_sources.search(key)]

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


@require_torch
class FineGrainedMxfp8ChainTest(_ChainAssertions, unittest.TestCase):
    """MXFP8, as MiniMax-M3 ships it."""

    def test_minimax_m3_e8m0_scale_bytes(self):
        """MiniMax-M3-MXFP8 ships its E8M0 scales as raw `uint8` bytes: they load as the module's
        `float8_e8m0fnu`, and a save restores the container the checkpoint used."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "weight_scale_inv": torch.randint(120, 130, (n, k // 32), dtype=torch.uint8),
            }
        )
        model, info, saved = _load_and_save(checkpoint, FineGrainedConfig(quant_method="mxfp8"))
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(experts.gate_up_proj_scale_inv.dtype, torch.float8_e8m0fnu)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)


@require_torch
class FineGrainedMxfp4ChainTest(_ChainAssertions, unittest.TestCase):
    """MXFP4 as GPT-OSS ships it: blocks (E, N, K/32, 16) uint8 low-nibble-first E2M1 + biased-127
    exponent scales, gate|up rows INTERLEAVED."""

    FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def _reference_dequant(self, blocks, scales):
        # the mxfp4.py reference semantics: lut[lo], lut[hi] interleaved along K, x 2^(scale-127)
        lut = torch.tensor(self.FP4_VALUES)
        lo = lut[(blocks & 0xF).long()]
        hi = lut[(blocks >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(*blocks.shape[:-1], -1)  # (E, N, K/32, 32)
        exp = (scales.long() - 127).unsqueeze(-1)
        return (vals * torch.pow(torch.tensor(2.0), exp)).reshape(*blocks.shape[:2], -1)

    def test_packed_experts(self):
        """Packed E2M1 per expert (two values per `int8` byte along K) with an E8M0 scale per 32, as
        DeepSeek-V4 ships its experts, merged onto the stacked experts."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randint(-128, 128, (n, k // 2), dtype=torch.int8),
                "weight_scale_inv": torch.randint(120, 130, (n, k // 32), dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            }
        )
        model, info, saved = _load_and_save(checkpoint, FineGrainedConfig(quant_method="mxfp4"))
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)

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


@require_torch
class FineGrainedNvfp4ChainTest(_ChainAssertions, unittest.TestCase):
    """NVFP4 as modelopt ships it, per expert (GLM-5.2-NVFP4), stacked per layer (vLLM) and on dense
    linears (modelopt's default export). `MergeModulelist` is load-bearing beyond the merge itself:
    `core_model_loading` stamps a source with its expert index ONLY when one is present in the chain,
    and expert parallelism selects experts by that index."""

    def _modelopt_checkpoint(self):
        """modelopt's per-expert layout, dense linear included: packed E2M1 bytes, E4M3 block scales
        per 16, fp32 second-level `weight_scale_2` and calibrated `input_scale`. Gate and up share
        their globals, as modelopt's linear fusion exports them, so a save writes each back exactly."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8),
                "weight_scale": (torch.rand(n, k // 16) + 0.5).to(torch.float8_e4m3fn),
                "weight_scale_2": torch.rand(()) * 1e-3 + 1e-4,
                "input_scale": torch.rand(()) * 1e-2 + 1e-3,
            }
        )
        for expert in range(E):
            prefix = f"model.layers.0.mlp.experts.{expert}"
            for suffix in ("weight_scale_2", "input_scale"):
                checkpoint[f"{prefix}.up_proj.{suffix}"] = checkpoint[f"{prefix}.gate_proj.{suffix}"].clone()
        # the gate|up activation global is one value for the layer: its rows are quantized before routing
        for expert in range(E):
            for proj in ("gate_proj", "up_proj"):
                checkpoint[f"model.layers.0.mlp.experts.{expert}.{proj}.input_scale"] = torch.tensor(5e-3)
        return checkpoint

    def test_modelopt_per_expert_and_dense(self):
        checkpoint = self._modelopt_checkpoint()
        model, info, saved = _load_and_save(checkpoint, FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4"))
        self.assertLoadsCleanly(info)
        experts, q_proj = model.model.layers[0].mlp.experts, model.model.layers[0].self_attn.q_proj
        self.assertEqual((experts.gate_up_proj.dtype, q_proj.weight.dtype), (torch.int8, torch.int8))
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        torch.testing.assert_close(
            experts.gate_up_proj_weight_global_scale,
            torch.stack([checkpoint[f"model.layers.0.mlp.experts.{e}.gate_proj.weight_scale_2"] for e in range(E)]),
        )
        for slot, key in (
            ("weight_global_scale", "weight_scale_2"),
            ("input_global_scale", "input_scale"),
        ):
            torch.testing.assert_close(getattr(q_proj, slot), checkpoint[f"model.layers.0.self_attn.q_proj.{key}"])
        self.assertSavesBack(saved, checkpoint)

    def test_modelopt_fused_layout(self):
        """The vLLM layout: the per-expert modelopt tensors already stacked per layer."""
        per_expert = self._modelopt_checkpoint()
        checkpoint = {k: v for k, v in per_expert.items() if ".experts." not in k}
        for proj, halves in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
            for suffix, stacked in (("", "weight"), ("_weight_scale", "weight_scale")):
                checkpoint[f"model.layers.0.mlp.experts.{proj}{suffix}"] = torch.stack(
                    [
                        torch.cat([per_expert[f"model.layers.0.mlp.experts.{e}.{h}.{stacked}"] for h in halves])
                        for e in range(E)
                    ]
                )
            # the globals stack one per expert; the gate|up pair agrees here, so its one value covers both
            # halves (a pair calibrated apart folds, which the global-scale op tests cover)
            for suffix in ("weight_scale_2", "input_scale"):
                checkpoint[f"model.layers.0.mlp.experts.{proj}_{suffix}"] = torch.stack(
                    [per_expert[f"model.layers.0.mlp.experts.{e}.{halves[0]}.{suffix}"] for e in range(E)]
                )
        model, info, saved = _load_and_save(checkpoint, FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4"))
        self.assertLoadsCleanly(info)
        self.assertEqual(model.model.layers[0].mlp.experts.gate_up_proj.dtype, torch.int8)
        self.assertSavesBack(saved, checkpoint)

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
        from transformers.integrations.finegrained import FineGrainedLinear

        cfg = _Cfg()
        for activation_format, holds_global in ((None, True), ("bf16", False)):
            experts = FineGrainedExperts(
                cfg, block_size=(4, 4), weight_format="nvfp4", activation_format=activation_format
            )
            linear = FineGrainedLinear(32, 32, weight_format="nvfp4", activation_format=activation_format)
            for module in (experts, linear):
                slots = {name for name, _ in module.named_parameters()}
                self.assertEqual(
                    any("input_global_scale" in slot for slot in slots),
                    holds_global,
                    f"{type(module).__name__} with activation_format={activation_format!r} disagrees about the slot",
                )
            for c in self._modelopt_conversions(activation_format=activation_format):
                for target in _targets(c):
                    # an expert target names the experts' slot; a dense rename's, the linear's
                    module, slot = (
                        (experts, target.split("experts.")[-1])
                        if "experts." in target
                        else (linear, target.removeprefix("\\1."))
                    )
                    self.assertIn(
                        slot.rstrip("$"),
                        {name for name, _ in module.named_parameters()},
                        f"activation_format={activation_format!r} converts onto a missing slot",
                    )

    def test_global_scale_converters_carry_an_expert_index(self):
        from transformers.core_model_loading import MergeModulelist

        # the per-expert layout only (`experts.*.`): a checkpoint that ships one already-stacked
        # tensor per layer has no per-expert sources to stamp — it is sharded like the fused
        # expert weight next to it
        globals_converters = [
            c
            for c in self._modelopt_conversions()
            if any("global_scale" in t for t in _targets(c))
            and c.compiled_sources.search("model.layers.0.mlp.experts.7.gate_proj.weight_scale_2")
        ]
        self.assertTrue(globals_converters, "no global-scale converter found")
        for conv in globals_converters:
            self.assertTrue(
                any(isinstance(op, MergeModulelist) for op in conv.operations),
                f"{_targets(conv)} has no MergeModulelist, so expert parallelism cannot select "
                "experts and every rank would collect all of them",
            )


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
