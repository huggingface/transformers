# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Shared format specs and fast quantizer contract tests."""

import re
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from unittest import mock

import torch

from transformers import AutoModelForCausalLM
from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
from transformers.integrations.finegrained import FineGrainedExperts, FineGrainedLinear
from transformers.quantizers.auto import AutoHfQuantizer
from transformers.testing_utils import (
    require_accelerate,
    require_kernels,
    require_torch_accelerator,
    torch_device,
)

from ..integrations.finegrained.utils import _Cfg, _conversion_model, _model_mapping, _quantizer_for


@dataclass(frozen=True)
class FormatSpec:
    """What the shared tests need to know about one format.

    - `quantize_kwargs` configure the on-the-fly quantization test. `None` means the quantize op is tested on its own.
    - `checkpoint_keys` are keys of the format's shipped layouts. At most one converter may claim each.
    - `dequantize_keys` are a quantized linear's keys. A dequantizing load must route each to a converter.
    - `tiny_checkpoints` maps each tiny checkpoint to its logits budget against its `reference/` bf16 copy.
    - `distributed_models` names the `LoadPathEquivalenceTesterMixin` fixtures built in this format.
    """

    name: str
    config_cls: type
    quantizer_cls: type
    checkpoint_config: Callable
    activation_formats: tuple = ()
    weight_bits: int = 16
    two_level: bool = False
    quantize_kwargs: dict | None = None
    checkpoint_keys: tuple = ()
    dequantize_keys: tuple = ()
    tiny_checkpoints: dict = field(default_factory=dict)
    distributed_models: tuple = ()


FORMATS = ()


@contextmanager
def no_accelerator():
    with ExitStack() as stack:
        stack.enter_context(mock.patch("torch.cuda.is_available", return_value=False))
        if hasattr(torch, "xpu"):
            stack.enter_context(mock.patch("torch.xpu.is_available", return_value=False))
        stack.enter_context(
            mock.patch("transformers.quantizers.block_scaled_utils.is_torch_xpu_available", return_value=False)
        )
        yield


def _require_kernel_capability(test):
    # Below 8.9 the quantizer dequantizes, which is not what these tests check
    if torch_device == "cuda" and torch.cuda.get_device_capability() < (8, 9):
        test.skipTest("fine-grained quantized compute needs compute capability >= 8.9")


class BlockScaledFormatTesterMixin:
    """Contract tests every format passes. A format's test file sets `spec` and inherits them."""

    spec: FormatSpec

    def test_config_round_trips_through_a_dict(self):
        config = self.spec.config_cls(modules_to_not_convert=["lm_head"])
        read = self.spec.config_cls.from_dict(config.to_dict())
        self.assertEqual(read.to_dict(), config.to_dict())
        self.assertEqual(read.quant_method.value, self.spec.name)

    def test_from_pretrained_picks_the_format_quantizer(self):
        quantizer = AutoHfQuantizer.from_config(self.spec.config_cls())
        self.assertIsInstance(quantizer, self.spec.quantizer_cls)

    def test_the_config_takes_only_the_activation_formats_its_kernels_serve(self):
        """Check that the config refuses activation formats the weight format's kernels do not serve."""
        for activation_format in {None} | {f for spec in FORMATS for f in spec.activation_formats}:
            with self.subTest(activation_format=activation_format):
                if activation_format in self.spec.activation_formats:
                    self.spec.config_cls(activation_format=activation_format)
                else:
                    with self.assertRaises(ValueError):
                        self.spec.config_cls(activation_format=activation_format)

    def test_a_checkpoint_key_is_claimed_by_at_most_one_converter(self):
        """Check that at most one converter claims each checkpoint key.

        When two claim a key, the last one wins and drops the other's ops. Keys are renamed first, as in the loader.
        """
        quantizer = _quantizer_for(self.spec.checkpoint_config())
        quantizer.pre_quantized = True
        conversions = quantizer.update_weight_conversions(_model_mapping(), _conversion_model(quantizer))
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]
        for key in self.spec.checkpoint_keys:
            renamed, _ = rename_source_key(key, renamings, [])
            claimed = [c for c in converters if c.compiled_sources.search(renamed)]
            with self.subTest(key=key):
                self.assertLessEqual(
                    len(claimed),
                    1,
                    f"{len(claimed)} converters claim this key; the last one wins and drops "
                    f"the others' ops: {[[type(o).__name__ for o in c.operations] for c in claimed]}",
                )

    def test_dequantize_claims_every_key_of_a_quantized_linear(self):
        """Check that a dequantizing load routes every key of a quantized linear to a converter.

        No fine-grained module is swapped in, so no scale slot exists to receive them.
        """
        quantizer = _quantizer_for(self.spec.config_cls(dequantize=True))
        quantizer.pre_quantized = True
        conversions = [*quantizer.get_checkpoint_renamings(), *quantizer.get_weight_conversions()]
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]
        for key in self.spec.dequantize_keys:
            renamed, _ = rename_source_key(key, renamings, [])
            with self.subTest(key=key):
                self.assertTrue(
                    any(c.compiled_sources.search(renamed) for c in converters), f"nothing dequantizes {key}"
                )

    def test_module_sizes_count_the_quantized_weights(self):
        """Check that module sizes count the quantized weights.

        The device map is sized from the replaced modules' dtypes and shapes, before any weight loads.
        """
        from transformers import Qwen3MoeConfig
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.modeling_utils import expand_device_map, get_total_byte_count

        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=256,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=4,
            num_experts_per_tok=2,
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
            sizes, _ = compute_module_sizes(model, only_modules=False)
            quantizer = AutoHfQuantizer.from_config(self.spec.config_cls(), pre_quantized=False)
            quantizer.preprocess_model(model=model, config=model.config)
            quantized_sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
            keys = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
            (total,) = get_total_byte_count(model, expand_device_map({"": torch_device}, keys), quantizer).values()

        weights = [
            f"{name}.{param}"
            for name, module in model.named_modules()
            if isinstance(module, (FineGrainedLinear, FineGrainedExperts))
            for param in (("weight",) if isinstance(module, FineGrainedLinear) else ("gate_up_proj", "down_proj"))
        ]
        self.assertTrue(weights)
        for weight in weights:
            self.assertEqual(quantized_sizes[weight], sizes[weight] * self.spec.weight_bits // 16, weight)
        self.assertEqual(total, quantized_sizes[""])

    @require_kernels
    @require_torch_accelerator
    def test_on_the_fly_quantization_lands_in_the_module_layout(self):
        """Check that on-the-fly quantization fills the module's held layout.

        Only a two-level format adds the second-level global scale.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        if self.spec.quantize_kwargs is None:
            self.skipTest(f"{self.spec.name}'s quantize op is tested on its own")
        _require_kernel_capability(self)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 512, 256, 4
        cfg._experts_implementation = "grouped_mm"
        experts = FineGrainedExperts(cfg, quantization_config=self.spec.config_cls(**self.spec.quantize_kwargs))
        experts = experts.to(torch_device)
        model = torch.nn.Module()
        model.experts = experts
        original = torch.randn(4, 512, 512, device=torch_device, dtype=torch.bfloat16)
        out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.gate_up_proj": original}, model=model)
        for name in ("gate_up_proj", "gate_up_proj_scale_inv"):
            for op in _quantizer_for(self.spec.config_cls()).get_param_ops(model, f"experts.{name}"):
                key = f"experts.{name}"
                out.update(op.convert({key: out[key]}, model=model, full_layer_name=key))
        for name in ("gate_up_proj", "gate_up_proj_scale_inv"):
            held = getattr(experts, name)
            self.assertEqual((out[f"experts.{name}"].dtype, out[f"experts.{name}"].shape), (held.dtype, held.shape))
        global_key = "experts.gate_up_proj_weight_global_scale"
        if self.spec.two_level:
            self.assertEqual(out[global_key].shape, experts.gate_up_proj_weight_global_scale.shape)
        else:
            self.assertNotIn(global_key, out)

    @require_accelerate
    def test_a_missing_requirement_dequantizes_a_checkpoint_and_refuses_to_quantize(self):
        from transformers.utils import logging as hf_logging

        def cuda(capability=(9, 0), kernels=True):
            stack = ExitStack()
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(mock.patch("torch.cuda.get_device_capability", return_value=capability))
            stack.enter_context(
                mock.patch("transformers.quantizers.block_scaled_utils.is_kernels_available", return_value=kernels)
            )
            return stack

        causes = {
            "a GPU or XPU": no_accelerator,
            "compute capability": lambda: cuda(capability=(6, 0)),
            "the `kernels` package": lambda: cuda(kernels=False),
        }
        for cause, environment in causes.items():
            with self.subTest(cause=cause):
                hf_logging.warning_once.cache_clear()
                quantizer = _quantizer_for(self.spec.checkpoint_config())
                quantizer.pre_quantized = True
                with environment(), self.assertLogs("transformers", level="WARNING") as logs:
                    quantizer.validate_environment(device_map="cuda")
                self.assertTrue(quantizer.quantization_config.dequantize)
                self.assertEqual(len(logs.records), 1)
                self.assertIn(cause, logs.output[0])

                quantizer = _quantizer_for(self.spec.config_cls())
                quantizer.pre_quantized = False
                with environment(), self.assertRaisesRegex(RuntimeError, re.escape(cause)):
                    quantizer.validate_environment(device_map="cuda")
