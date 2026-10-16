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
"""The tests every block-scaled format runs, driven by its `FormatSpec`: a format's test file instantiates the
testers with its spec and adds only what is its own (its checkpoint layouts, its producers' quirks).

- `BlockScaledFormatTesterMixin`: the config, the quantizer a load picks, the activation formats it takes, the
  no-accelerator fallbacks, one converter per checkpoint key (CPU); every forward against the format's
  dequantized weights and the on-the-fly quantization round trip (GPU, the real kernels).
- `TinyCheckpointTesterMixin` (slow): the format's pushed tiny checkpoints load with every key accounted for,
  compute what their dequantized weights compute, match the recorded logits, and save back to their layout.
  `FINEGRAINED` runs it on the checkpoints that mix formats.
- `LoadPathEquivalenceTesterMixin` (slow, 2 GPUs): expert and tensor parallelism reproduce `device_map`."""

import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from unittest import mock

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
from transformers.integrations.finegrained import FineGrainedExperts, FineGrainedLinear
from transformers.quantizers.auto import AutoHfQuantizer, AutoQuantizationConfig
from transformers.quantizers.quantizer_blockfp8 import BlockFp8HfQuantizer
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
from transformers.quantizers.quantizer_mxfp8 import Mxfp8HfQuantizer
from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer
from transformers.testing_utils import (
    Expectations,
    cleanup,
    get_device_properties,
    get_torch_dist_unique_port,
    require_accelerate,
    require_kernels,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils.quantization_config import (
    BlockFp8Config,
    FineGrainedConfig,
    Mxfp4Config,
    Mxfp8Config,
    NVFP4Config,
)

from ..integrations.finegrained.utils import _Cfg, _conversion_model, _model_mapping, _quantizer_for


def _cheap_tuning():
    """One autotune trial per shape: a cold tune dwarfs everything a test runs."""
    return mock.patch.dict(os.environ, {"FINEGRAINED_AUTOTUNE_TRIALS": "1"})


@dataclass(frozen=True)
class FormatSpec:
    """What the shared tests need to know about one format; no test hard-codes it.

    `quantize_kwargs` configure the on-the-fly quantization whose layout is checked (`None`: the format's
    quantize op is tested on its own). `checkpoint_keys` are keys of the format's shipped layouts, each of which
    at most one converter may claim; `dequantize_keys` are a quantized linear's keys, each of which a
    dequantizing load must route to a converter. `tiny_checkpoints` maps each pushed tiny
    checkpoint to its logits budget against its `reference/` bf16 copy; `distributed_models` names the
    `LoadPathEquivalenceTesterMixin` fixtures built in this format."""

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


BLOCK_FP8 = FormatSpec(
    name="fp8",
    config_cls=BlockFp8Config,
    quantizer_cls=BlockFp8HfQuantizer,
    activation_formats=(None, "fp8"),
    checkpoint_config=BlockFp8Config,
    weight_bits=8,
    checkpoint_keys=(
        "model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv",
        "model.layers.3.mlp.experts.gate_up_proj_scale_inv",
        "model.layers.3.self_attn.q_proj.weight_scale_inv",
    ),
    dequantize_keys=("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight_scale_inv"),
    tiny_checkpoints={
        "IlyasMoutawwakil/tiny-random-Qwen3-FP8": 0.1,
        "IlyasMoutawwakil/tiny-random-Mistral3-FP8-static": 0.1,
        "IlyasMoutawwakil/tiny-random-Mistral4-FP8-static": 0.15,
        "IlyasMoutawwakil/tiny-random-DeepseekV3-FP8": 0.25,
    },
    distributed_models=("deepseek_v3-fp8", "mistral4-fp8_tensor_static"),
)
MXFP8 = FormatSpec(
    name="mxfp8",
    config_cls=Mxfp8Config,
    quantizer_cls=Mxfp8HfQuantizer,
    activation_formats=(None, "mxfp8", "mxfp4", "bf16"),
    checkpoint_config=Mxfp8Config,
    weight_bits=8,
    quantize_kwargs={},
    checkpoint_keys=(
        "model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv",
        "model.layers.3.mlp.experts.down_proj_scale_inv",
    ),
    dequantize_keys=("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight_scale_inv"),
    tiny_checkpoints={"IlyasMoutawwakil/tiny-random-MiniMaxM3-MXFP8": 0.3},
    distributed_models=("minimax_m3_vl-mxfp8",),
)
MXFP4 = FormatSpec(
    name="mxfp4",
    config_cls=Mxfp4Config,
    quantizer_cls=Mxfp4HfQuantizer,
    activation_formats=(None, "mxfp8", "mxfp4", "bf16"),
    checkpoint_config=Mxfp4Config,
    weight_bits=4,
    quantize_kwargs={"activation_format": None},
    checkpoint_keys=(
        "model.layers.3.mlp.experts.gate_up_proj_blocks",
        "model.layers.3.mlp.experts.gate_up_proj_scales",
        "model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv",
    ),
    # GPT-OSS's blocks + scales pair
    dequantize_keys=(
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
    ),
    tiny_checkpoints={"IlyasMoutawwakil/tiny-random-GptOss-MXFP4": 0.05},
    distributed_models=("gpt_oss-mxfp4",),
)
NVFP4 = FormatSpec(
    name="nvfp4",
    config_cls=NVFP4Config,
    quantizer_cls=NVFP4HfQuantizer,
    activation_formats=(None, "nvfp4", "bf16"),
    # what a modelopt export reads as
    checkpoint_config=lambda: AutoQuantizationConfig.from_dict({"quant_method": "modelopt", "quant_algo": "NVFP4"}),
    weight_bits=4,
    two_level=True,
    quantize_kwargs={},
    checkpoint_keys=(
        "model.layers.3.mlp.experts.gate_up_proj",
        "model.layers.3.mlp.experts.down_proj",
        "model.layers.3.mlp.experts.gate_up_proj_weight_scale_2",
        "model.layers.3.mlp.experts.7.up_proj.weight_scale_2",
    ),
    dequantize_keys=(
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale",
        "model.layers.0.mlp.down_proj.weight_scale_2",
    ),
    tiny_checkpoints={
        "IlyasMoutawwakil/tiny-random-Llama-NVFP4": 0.25,
        "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4": 0.25,
        "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4-all-linears": 0.6,
        "IlyasMoutawwakil/tiny-random-Muse-Spark-NVFP4": 0.3,
    },
    distributed_models=("glm4v_moe-nvfp4",),
)
FORMATS = (BLOCK_FP8, MXFP8, MXFP4, NVFP4)

# Plans mixing the formats, as DeepSeek-V4 ships one (block-FP8 over MXFP4 experts).
FINEGRAINED = FormatSpec(
    name="finegrained",
    config_cls=FineGrainedConfig,
    quantizer_cls=FineGrainedHfQuantizer,
    checkpoint_config=FineGrainedConfig,
    tiny_checkpoints={"IlyasMoutawwakil/tiny-random-DeepseekV4-Flash": 0.25},
    distributed_models=("deepseek_v4-fp4_experts+fp8_dense",),
)


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
    # below it the quantizer dequantizes instead, which is not what is under test
    if torch_device == "cuda" and torch.cuda.get_device_capability() < (8, 9):
        test.skipTest("fine-grained quantized compute needs compute capability >= 8.9")


class BlockScaledFormatTesterMixin:
    """The contract every format keeps. A format's test file sets `spec` and inherits these."""

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
        """The kernels quantize activations only to what the weight format's MMA takes, and refuse the rest at
        the first forward; the config refuses them up front."""
        for activation_format in {None} | {f for spec in FORMATS for f in spec.activation_formats}:
            with self.subTest(activation_format=activation_format):
                if activation_format in self.spec.activation_formats:
                    self.spec.config_cls(activation_format=activation_format)
                else:
                    with self.assertRaises(ValueError):
                        self.spec.config_cls(activation_format=activation_format)

    @require_accelerate
    def test_no_accelerator_refuses_to_quantize(self):
        # quantizing a bf16 checkpoint needs the hardware; there is nothing to fall back to
        quantizer = _quantizer_for(self.spec.config_cls())
        quantizer.pre_quantized = False
        with no_accelerator(), self.assertRaises(RuntimeError):
            quantizer.validate_environment()

    @require_accelerate
    def test_no_accelerator_dequantizes_a_quantized_checkpoint(self):
        quantizer = _quantizer_for(self.spec.checkpoint_config())
        quantizer.pre_quantized = True
        with no_accelerator():
            quantizer.validate_environment()
        self.assertTrue(quantizer.quantization_config.dequantize)

    @require_accelerate
    def test_a_gpu_below_compute_capability_8_9_dequantizes_a_quantized_checkpoint(self):
        quantizer = _quantizer_for(self.spec.checkpoint_config())
        quantizer.pre_quantized = True
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        ):
            quantizer.validate_environment()
        self.assertTrue(quantizer.quantization_config.dequantize)

    def test_a_checkpoint_key_is_claimed_by_at_most_one_converter(self):
        """Two converters claiming one key is not additive: the later one wins and drops the ops the first carried.
        Keys are routed as the loader routes them, renamed first."""
        quantizer = _quantizer_for(self.spec.checkpoint_config())
        quantizer.pre_quantized = True
        conversions = quantizer.update_weight_conversions(_model_mapping(), _conversion_model(quantizer))
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]
        for key in self.spec.checkpoint_keys:
            renamed, _ = rename_source_key(key, renamings, converters)
            claimed = [c for c in converters if c.compiled_sources.search(renamed)]
            with self.subTest(key=key):
                self.assertLessEqual(
                    len(claimed),
                    1,
                    f"{len(claimed)} converters claim this key; the last one wins and drops "
                    f"the others' ops: {[[type(o).__name__ for o in c.operations] for c in claimed]}",
                )

    def test_dequantize_claims_every_key_of_a_quantized_linear(self):
        """A dequantizing load swaps in no fine-grained module, so no scale slot exists: every key of the pair
        (and the global, for a two-level format) must reach a converter."""
        quantizer = _quantizer_for(self.spec.config_cls(dequantize=True))
        quantizer.pre_quantized = True
        conversions = quantizer.get_weight_conversions()
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]
        for key in self.spec.dequantize_keys:
            renamed, _ = rename_source_key(key, renamings, converters)
            with self.subTest(key=key):
                self.assertTrue(
                    any(c.compiled_sources.search(renamed) for c in converters), f"nothing dequantizes {key}"
                )

    def test_module_sizes_count_the_quantized_weights(self):
        """The device map is sized before any weight loads, from the replaced modules' dtypes and shapes."""
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
        """The kernels' quantizers then the module's layout ops fill the held layout (swizzled where it is held
        so), with the second-level global only for a two-level format."""
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


def _checkpoint_bytes(path):
    """`{key: (dtype, shape, raw bytes)}` of a saved checkpoint, for comparing one save against another."""
    import glob

    from safetensors import safe_open

    tensors = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                tensors[key] = (tensor.dtype, tuple(tensor.shape), raw)
    return tensors


# The logits `[0, :3, :8]` each checkpoint gives on `torch.arange(3, 35)`, per device: bit-stable across tuner
# caches, so a tight tolerance holds.
EXPECTED_SLICES = {
    "IlyasMoutawwakil/tiny-random-Qwen3-FP8": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.6094, -0.6055, -0.3262, 0.1396, -0.1270, -0.0420, 0.3750, -0.1592],
                [0.1689, -0.4961, 0.1621, 0.0752, -0.1895, 0.0688, 0.3320, -0.0234],
                [0.1069, -0.3027, 0.1050, 0.1553, 0.1699, -0.1572, 0.1699, 0.2598],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Mistral3-FP8-static": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.1992, -0.4141, -0.1650, -0.1826, 0.1309, 0.4375, -0.2832, 0.2480],
                [-0.1367, -0.2422, -0.1138, -0.3418, 0.2910, 0.3965, -0.0293, -0.0330],
                [-0.1436, -0.1055, -0.3320, -0.3203, 0.3633, 0.2480, 0.1523, -0.1621],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Mistral4-FP8-static": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.0850, 0.1816, -0.2344, 0.5547, -0.0747, 0.6875, -0.2949, 0.1982],
                [-0.0056, -0.0065, 0.0530, 0.0996, 0.1436, 0.2217, -0.3281, 0.1846],
                [0.0055, 0.2559, -0.1045, -0.0031, 0.0608, 0.0449, 0.0962, 0.1416],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Llama-NVFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.7617, -0.5938, -0.2832, 0.2559, -0.1021, 0.0082, 0.3418, -0.1699],
                [0.6523, -0.7148, 0.2930, 0.3750, -0.5039, 0.0835, 0.2793, -0.0153],
                [0.3457, -0.4785, 0.1924, 0.4102, -0.2715, -0.0835, 0.2070, 0.1602],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-DeepseekV3-FP8": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.1260, -0.2051, -0.1836, -0.6523, 0.4395, -0.0771, -0.7461, 0.3613],
                [-0.1025, -0.0640, -0.0132, -0.6602, 0.3672, -0.1270, -0.9023, -0.0786],
                [-0.0874, 0.0552, 0.0442, -0.4824, 0.4375, -0.4121, -0.4980, -0.1836],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-DeepseekV4-Flash": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.3984, 0.3633, 0.1348, -0.0476, -0.1118, -0.0518, -0.0952, -0.4590],
                [-0.4336, 0.1836, 0.0049, -0.1514, -0.0449, -0.2383, -0.1113, -0.4766],
                [-0.2559, 0.0693, -0.0664, -0.2207, -0.2500, -0.1177, -0.1562, -0.2949],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-GptOss-MXFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.1309, 0.2061, 0.2559, 0.5156, -0.2715, 0.5469, 0.4004, -0.0854],
                [-0.2490, 0.6406, 0.1289, 0.4277, -0.3809, 0.3125, 0.3516, -0.0811],
                [-0.2334, 0.1367, 0.2930, 0.6875, -0.1836, 0.4844, 0.1953, -0.2363],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.1709, -0.4199, -0.4199, -0.2402, -0.3125, 0.6016, 0.3008, 0.1914],
                [-0.2051, -0.6328, -0.5391, 0.3184, 0.1299, 0.2773, 0.4824, 0.0938],
                [0.0447, -0.5078, -0.3672, -0.1992, 0.1553, 0.6523, 0.4141, -0.0038],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4-all-linears": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.3086, -0.2773, -0.4883, -0.1797, -0.0073, 0.4531, 0.2158, 0.2754],
                [0.0295, -0.2598, -0.5586, 0.2432, -0.0106, 0.3496, 0.4805, 0.1279],
                [0.1533, -0.4551, -0.5742, -0.1973, 0.2139, 0.4141, 0.3574, 0.1338],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Muse-Spark-NVFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.0869, 0.0566, -0.0145, 0.0112, 0.0103, 0.0012, -0.0679, -0.0064],
                [0.0283, -0.0557, 0.0192, 0.0322, -0.0295, -0.0117, 0.0173, 0.0267],
                [-0.0304, -0.0079, 0.0510, 0.0081, 0.0280, 0.0067, 0.0635, 0.0201],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-MiniMaxM3-MXFP8": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.2637, -0.0815, -0.3418, 0.1553, -0.2715, 0.1562, 0.4590, 0.0635],
                [0.4355, -0.1016, -0.2002, -0.2412, -0.2930, 0.1187, 0.5664, 0.1104],
                [0.1865, -0.0996, -0.6875, 0.2988, 0.0635, 0.1035, 0.5586, -0.0703],
            ],
        }
    ),  # fmt: skip
}


class TinyCheckpointTesterMixin:
    """A format's pushed tiny checkpoints, random copies of the shipped ones written in their producers'
    layouts (modelopt, OpenAI's and DeepSeek's quantizers): each loads with every key accounted for, computes
    what its dequantized weights compute within the format's rounding, matches its recorded logits, and saves
    back to its own layout. So a converter is checked against what the format really is, not against what this
    code believes it to be. A format's test file sets `spec`."""

    spec: FormatSpec

    def setUp(self):
        super().setUp()
        self.enterContext(_cheap_tuning())
        _require_kernel_capability(self)

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

    @staticmethod
    def _load(repo, **kwargs):
        config = AutoConfig.from_pretrained(repo, **kwargs)
        model_cls = AutoModelForImageTextToText if "vision_config" in config.to_dict() else AutoModelForCausalLM
        return model_cls.from_pretrained(repo, device_map=torch_device, output_loading_info=True, **kwargs)

    @staticmethod
    def _unbuilt_mtp_keys(model, keys):
        """The multi-token-prediction layer the checkpoint ships and the model does not build: it
        sits at index `num_hidden_layers`, which the models' own ignore rule only names at the
        full-size index."""
        mtp = model.config.get_text_config().num_hidden_layers
        return {key for key in keys if re.search(rf"(^|\.)layers\.{mtp}\.", key)}

    def _not_loaded(self, model, keys):
        """Checkpoint keys the model does not load, so a save does not write them back: the unbuilt MTP
        layer, and what the load declares it ignores (an FP8 KV cache's scales)."""
        ignored = model._keys_to_ignore_on_load_unexpected or []
        return self._unbuilt_mtp_keys(model, keys) | {k for k in keys if any(re.search(p, k) for p in ignored)}

    def _reference_error(self, model, repo):
        """The logits' error against the dequantized reference, and the logits."""
        reference, _ = self._load(repo, subfolder="reference", dtype=torch.bfloat16)
        ids = torch.arange(3, 35, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            got, want = model(ids).logits.float(), reference(ids).logits.float()
        return ((got - want).norm() / want.norm()).item(), got

    @staticmethod
    def _recorded_slice(repo):
        """The slice recorded on this exact device, if any: another device's kernels round differently."""
        return dict(EXPECTED_SLICES[repo].unpacked()).get(get_device_properties())

    @slow
    @require_kernels
    @require_torch_accelerator
    def test_logits_match_the_dequantized_reference_and_the_recorded_slice(self):
        for repo, budget in self.spec.tiny_checkpoints.items():
            with self.subTest(repo=repo):
                model, info = self._load(repo)
                unexpected = set(info["unexpected_keys"]) - self._unbuilt_mtp_keys(model, info["unexpected_keys"])
                self.assertEqual(
                    (set(info["missing_keys"]), unexpected, set(info["mismatched_keys"])), (set(), set(), set())
                )
                error, logits = self._reference_error(model, repo)
                self.assertLessEqual(error, budget, f"{repo}: logits {error:.4f} away from the dequantized reference")
                expected = self._recorded_slice(repo)
                if expected is not None:
                    torch.testing.assert_close(logits[0, :3, :8].cpu(), torch.tensor(expected), rtol=1e-2, atol=5e-3)

    @slow
    @require_kernels
    @require_torch_accelerator
    def test_dequantized_load_matches_the_reference(self):
        """`dequantize=True` folds each weight against its scales (and its global, for a two-level format)."""
        for repo, budget in self.spec.tiny_checkpoints.items():
            with self.subTest(repo=repo):
                model, info = self._load(repo, quantization_config=self.spec.config_cls(dequantize=True))
                unexpected = set(info["unexpected_keys"]) - self._unbuilt_mtp_keys(model, info["unexpected_keys"])
                self.assertEqual((set(info["missing_keys"]), unexpected), (set(), set()))
                # nothing stays quantized; a model keeps its own fp32 parameters (DeepSeek-V4's hyper-connections)
                quantized = {torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8, torch.uint8}
                self.assertEqual([name for name, param in model.named_parameters() if param.dtype in quantized], [])
                error, _ = self._reference_error(model, repo)
                self.assertLessEqual(error, budget, f"{repo}: logits {error:.4f} away from the dequantized reference")

    @slow
    @require_kernels
    @require_torch_accelerator
    def test_save_writes_the_checkpoint_layout_back(self):
        """A pre-quantized model saved again lands on its checkpoint's keys, dtypes and bytes."""
        from huggingface_hub import snapshot_download

        for repo in self.spec.tiny_checkpoints:
            with self.subTest(repo=repo):
                model, _ = self._load(repo)
                saved = os.path.join(self.get_auto_remove_tmp_dir(), "saved")
                model.save_pretrained(saved)
                source = _checkpoint_bytes(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"]))
                source = {key: value for key, value in source.items() if key not in self._not_loaded(model, source)}
                written = _checkpoint_bytes(saved)
                self.assertEqual(sorted(written), sorted(source))
                for key, (dtype, shape, raw) in written.items():
                    with self.subTest(repo=repo, key=key):
                        source_dtype, source_shape, source_raw = source[key]
                        self.assertEqual(shape, source_shape)
                        if dtype != source_dtype:
                            # Loading honors initialized dtypes, including FP32 norms and BF16
                            # MiniMax routers/patch embeddings. Float32 UE8M0 containers save natively.
                            self.assertTrue(
                                (
                                    dtype == torch.float32
                                    and key.endswith("norm.weight")
                                    and source_dtype == torch.bfloat16
                                )
                                or (
                                    dtype == torch.bfloat16
                                    and source_dtype == torch.float32
                                    and key.endswith(
                                        ("gate.weight", "e_score_correction_bias", "patch_embedding.weight")
                                    )
                                )
                                or (
                                    dtype == torch.float8_e8m0fnu
                                    and source_dtype == torch.float32
                                    and key.endswith((".scale", "_scale_inv"))
                                )
                            )
                            expected = torch.frombuffer(bytearray(source_raw), dtype=source_dtype).to(dtype)
                            self.assertEqual(raw, expected.view(torch.uint8).numpy().tobytes())
                        else:
                            self.assertEqual(raw, source_raw)


_SHARDING_WORKER = """
import importlib, os, sys, torch
from transformers.distributed import DistributedConfig

model_dir, out_dir, modes, cls_path = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
module_name, cls_name = cls_path.split(":")
model_cls = getattr(importlib.import_module(module_name), cls_name)
world = int(os.environ["WORLD_SIZE"])
ids = torch.arange(16, dtype=torch.long).unsqueeze(0)

# both modes in one launch: they share the mesh and differ only in the plan
for mode in modes:
    model = model_cls.from_pretrained(
        model_dir,
        dtype="auto",
        attn_implementation="eager",
        distributed_config=DistributedConfig(tp_size=world, ep_size=world if mode == "ep" else None),
    ).eval()
    experts = next(m for n, m in model.named_modules() if n.endswith("mlp.experts"))
    weight = getattr(experts, experts.gate_up_name)
    local = weight.to_local() if hasattr(weight, "to_local") else weight
    with torch.no_grad():
        logits = model(ids.to(model.device)).logits.float().cpu()
    if int(os.environ["RANK"]) == 0:
        torch.save({"logits": logits, "expert_local": tuple(local.shape)},
                   os.path.join(out_dir, mode + ".pt"))
    del model
    torch.accelerator.empty_cache()
"""


def _checkpoint_expert_shape(model_dir):
    """The unsharded `(experts, rows)` of the stacked gate|up projection the module holds, read from the
    checkpoint, which ships the experts either fused (`experts.gate_up_proj`) or one per expert
    (`experts.0.gate_proj.weight`, or DeepSeek's `w1`/`w3`)."""
    import glob
    import json

    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = sorted({os.path.join(model_dir, f) for f in weight_map.values()})

    experts: set[int] = set()
    rows = 0
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                shape = None
                if re.search(r"\.experts\.gate_up_proj$", key):
                    shape = handle.get_slice(key).get_shape()
                    if len(shape) == 3:
                        return (shape[0], shape[1])
                if re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|w1|w3)\.weight$", key):
                    experts.add(int(key.rsplit(".experts.", 1)[1].split(".", 1)[0]))
                    rows = max(rows, handle.get_slice(key).get_shape()[0])
    if not experts or not rows:
        raise AssertionError(f"no expert projection in {model_dir} to size the shard against")
    return (len(experts), 2 * rows)  # the module stacks gate and up into one row extent


class LoadPathEquivalenceTesterMixin:
    """Expert parallelism and intra-expert tensor parallelism must both reproduce what
    `device_map` gives, which is the baseline because it places modules across devices without
    splitting a single tensor — no process group, no DTensor, no collectives. So it computes the
    unsharded answer, on the multi-GPU path people actually deploy.

    Each model is built in the format it actually ships, because the conversion path differs by
    format and by checkpoint layout: a per-expert FP8 checkpoint and a fused NVFP4 one reach the
    experts through different converters.

    Three properties make this able to fail:
      * the fixture is PRE-QUANTIZED. Quantizing on the fly derives each rank's scales from the
        shard it already holds, so they come out correctly sized whatever the plan says and a
        plan that shards no scale at all still produces the right answer.
      * the model's plan must carry expert entries. A model whose `base_model_tp_plan` is empty
        (GPT-OSS, DeepSeek-V4) shards nothing under TP, so that leg is not run.
      * every leg reports the local expert shard, so a leg that placed nothing fails loudly.

    A format's test file sets `spec`; the fixtures are its `distributed_models`."""

    spec: FormatSpec

    @staticmethod
    def _model_table():
        """`{label: (config_cls, model_cls, config_kwargs, quantization_config)}`: each model in the format and
        under the quantization config it ships with. Dims are multiples of the 128 block so a 2-way split stays
        block-aligned, and the expert count divides the mesh."""
        from transformers import (
            DeepseekV3Config,
            DeepseekV3ForCausalLM,
            DeepseekV4Config,
            DeepseekV4ForCausalLM,
            Glm4vMoeConfig,
            Glm4vMoeForConditionalGeneration,
            Glm4vMoeTextConfig,
            Glm4vMoeVisionConfig,
            GptOssConfig,
            GptOssForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
            MiniMaxM3VLConfig,
            Mistral4Config,
            Mistral4ForCausalLM,
        )

        return {
            "deepseek_v3-fp8": (
                DeepseekV3Config,
                DeepseekV3ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "n_group": 1,
                    "topk_group": 1,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                    "q_lora_rank": None,
                    "kv_lora_rank": 32,
                    "qk_nope_head_dim": 32,
                    "qk_rope_head_dim": 16,
                    "v_head_dim": 32,
                },
                BlockFp8Config(weight_block_size=(128, 128)),
            ),
            # the only shipped STATIC scheme: per-TENSOR weights (no `weight_block_size`) and a
            # calibrated activation scale per quantized module, which for a MoE is one per expert.
            # Its conversion script asserts `qscheme_act == "TENSOR"`; Ministral-3 is the dense
            # counterpart of the same export.
            "mistral4-fp8_tensor_static": (
                Mistral4Config,
                Mistral4ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                },
                BlockFp8Config(weight_block_size=None, activation_scheme="static"),
            ),
            # multimodal: the experts' plans live on `text_config`, not on the config the quantizer is handed
            "glm4v_moe-nvfp4": (
                Glm4vMoeConfig,
                Glm4vMoeForConditionalGeneration,
                {
                    "text_config": Glm4vMoeTextConfig(
                        vocab_size=64,
                        hidden_size=256,
                        intermediate_size=256,
                        moe_intermediate_size=256,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        max_position_embeddings=32,
                        rope_parameters={"type": "default", "mrope_section": [16, 8, 8], "partial_rotary_factor": 1.0},
                        rope_theta=10000,
                        tie_word_embeddings=True,
                        bos_token_id=0,
                        eos_token_id=0,
                        pad_token_id=0,
                        n_routed_experts=4,
                        n_shared_experts=1,
                        n_group=1,
                        topk_group=1,
                        num_experts_per_tok=2,
                        first_k_dense_replace=0,
                    ),
                    "vision_config": Glm4vMoeVisionConfig(
                        depth=2,
                        num_heads=4,
                        hidden_size=64,
                        out_hidden_size=256,
                        intermediate_size=64,
                        patch_size=14,
                        spatial_merge_size=1,
                        temporal_patch_size=2,
                    ),
                },
                NVFP4.checkpoint_config(),
            ),
            # interleaved rows (`is_concatenated=False`), transposed, with expert biases its EP plan already names
            "gpt_oss-mxfp4": (
                GptOssConfig,
                GptOssForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "num_local_experts": 4,
                    "num_experts_per_tok": 2,
                    "max_position_embeddings": 32,
                },
                Mxfp4Config(activation_format="bf16"),
            ),
            # mixed: `expert_dtype` makes the experts MXFP4 over block-FP8 with UE8M0 scales
            "deepseek_v4-fp4_experts+fp8_dense": (
                DeepseekV4Config,
                DeepseekV4ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                    "expert_dtype": "fp4",
                },
                BlockFp8Config(weight_block_size=(128, 128), scale_fmt="ue8m0"),
            ),
            # a second multimodal nesting; the MoE dims are raised so a 2-way split stays 128-aligned
            "minimax_m3_vl-mxfp8": (
                MiniMaxM3VLConfig,
                MiniMaxM3SparseForConditionalGeneration,
                {
                    "text_config": {
                        "hidden_size": 256,
                        # the 2-way split leaves 256: a swizzled tile is 128 wide along the down's K
                        "intermediate_size": 512,
                        "dense_intermediate_size": 256,
                        "shared_intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "head_dim": 64,
                        "rotary_dim": 32,
                        "vocab_size": 64,
                        "max_position_embeddings": 32,
                        "bos_token_id": 0,
                        "eos_token_id": 1,
                        "pad_token_id": 2,
                        "num_local_experts": 4,
                        "num_experts_per_tok": 2,
                        "n_shared_experts": 1,
                        "moe_layer_freq": [0, 1],
                        "layer_types": ["full_attention", "minimax_m3_sparse"],
                        "tie_word_embeddings": False,
                        "index_n_heads": 2,
                        "index_head_dim": 16,
                        "index_block_size": 8,
                        "index_topk_blocks": 4,
                        "index_local_blocks": 1,
                    },
                    "vision_config": {
                        # the 2-way split leaves 128, one swizzled K tile
                        "hidden_size": 256,
                        "intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_channels": 3,
                        "image_size": 14,
                        "patch_size": 14,
                        "temporal_patch_size": 2,
                        "spatial_merge_size": 1,
                    },
                    "image_token_index": 4,
                    "video_token_index": 5,
                    "projector_hidden_size": 256,
                    "pad_token_id": 2,
                },
                Mxfp8Config(),
            ),
        }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls._env = {"FINEGRAINED_AUTOTUNE_TRIALS": "1", "TRITON_CACHE_DIR": os.path.join(cls._tmp, "triton")}
        cls.enterClassContext(mock.patch.dict(os.environ, cls._env))

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

    def _fixture(self, label):
        """A pre-quantized checkpoint of one tiny model, in the format that model ships."""
        config_cls, model_cls, kwargs, quantization_config = self._model_table()[label]
        config = config_cls(**kwargs)

        bf16_dir = os.path.join(self._tmp, label, "bf16")
        quant_dir = os.path.join(self._tmp, label, "quantized")
        torch.manual_seed(0)
        # saved in bf16, the dtype every model here ships, so `dtype="auto"` reloads bf16
        model_cls(config).to(torch.bfloat16).save_pretrained(bf16_dir, safe_serialization=True)
        quantized = model_cls.from_pretrained(
            bf16_dir,
            dtype="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
            device_map=f"{torch_device}:0",
        )
        quantized.save_pretrained(quant_dir, safe_serialization=True)

        # a reload saved again lands on the same bytes: every layout op's reverse restores the checkpoint
        reloaded = model_cls.from_pretrained(quant_dir, dtype="auto", device_map=f"{torch_device}:0")
        round_trip = os.path.join(self._tmp, label, "round_trip")
        reloaded.save_pretrained(round_trip, safe_serialization=True)
        saved, resaved = _checkpoint_bytes(quant_dir), _checkpoint_bytes(round_trip)
        mismatched = sorted(key for key in saved.keys() | resaved.keys() if saved.get(key) != resaved.get(key))
        self.assertEqual(mismatched, [], f"{label}: save did not round-trip")
        return quant_dir, config, model_cls

    def _logits(self, model_dir, model_cls, **load_kwargs):
        model = model_cls.from_pretrained(model_dir, dtype="auto", attn_implementation="eager", **load_kwargs).eval()
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits.float().cpu()
        return logits

    def _rounding_floor(self, model_dir, model_cls, reference):
        """How far this model's logits move when every sharded block is perturbed by BF16 rounding.

        Sharding cannot be bit-exact: a rowwise all-reduce sums the same terms in a different order. How far
        that travels is a property of the model (a chain of MoE layers can amplify it a hundredfold), so the
        same magnitude injected on one device gives each model the budget its own conditioning earns.
        """
        model = model_cls.from_pretrained(
            model_dir, dtype="auto", attn_implementation="eager", device_map=f"{torch_device}:0"
        ).eval()
        # every block a sharded reduction passes through: the rounding accumulates down the stack
        blocks = [
            m
            for n, m in model.named_modules()
            if n.endswith((".self_attn", ".mlp", ".block_sparse_moe")) and n.count(".layers.") == 1
        ]
        generator = torch.Generator(device=model.device).manual_seed(0)

        def perturb(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return output
            ulp = torch.finfo(tensor.dtype).eps * tensor.abs().max()
            noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * ulp
            tensor = tensor + noise
            return (tensor,) + output[1:] if isinstance(output, tuple) else tensor

        handles = [b.register_forward_hook(perturb) for b in blocks]
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            perturbed = model(ids).logits.float().cpu()
        for handle in handles:
            handle.remove()
        return (perturbed - reference).abs().max().item()

    def _sharded(self, model_dir, model_cls, modes):
        """`{mode: payload}` from one 2-rank `torchrun`, through the real EP / TP load path."""
        script = os.path.join(self._tmp, "sharding_worker.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_SHARDING_WORKER)
        out = os.path.join(self._tmp, "out")
        os.makedirs(out, exist_ok=True)
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=2",
                f"--master_port={get_torch_dist_unique_port()}",
                script,
                model_dir,
                out,
                ",".join(modes),
                f"{model_cls.__module__}:{model_cls.__name__}",
            ],
            check=True,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
        return {m: torch.load(os.path.join(out, m + ".pt")) for m in modes}

    @slow
    @require_kernels
    @require_torch_multi_accelerator
    def test_every_load_path_agrees(self):
        for label in self.spec.distributed_models:
            with self.subTest(model=label):
                model_dir, config, model_cls = self._fixture(label)
                # `max_memory` forces the baseline across both devices; `auto` alone fits it on one
                reference = self._logits(model_dir, model_cls, device_map="auto", max_memory={0: "120MiB", 1: "40GiB"})
                noise_floor = self._rounding_floor(model_dir, model_cls, reference)

                # only the modes whose plans shard experts (a multimodal model keeps them on a sub-config)
                owners = [config] + [
                    c for name in getattr(type(config), "sub_configs", {}) if (c := getattr(config, name, None))
                ]
                modes = [
                    mode
                    for mode, attr in (("ep", "base_model_ep_plan"), ("tp", "base_model_tp_plan"))
                    if any(".experts." in key for owner in owners for key in (getattr(owner, attr, None) or {}))
                ]
                self.assertTrue(modes, f"{label}: no plan shards experts, so nothing is under test")
                whole = _checkpoint_expert_shape(model_dir)
                for mode, payload in self._sharded(model_dir, model_cls, modes).items():
                    with self.subTest(model=label, mode=mode):
                        local = payload["expert_local"]
                        # against the checkpoint's shape, never another mode's shard
                        self.assertLess(
                            local[0] * local[1],
                            whole[0] * whole[1],
                            f"{label}/{mode}: experts were not sharded ({local} of {whole}) — "
                            f"this leg cannot catch a bad axis",
                        )
                        # a wrong shard axis lands orders of magnitude above the model's rounding floor
                        budget = max(2e-2, noise_floor)
                        gap = (payload["logits"] - reference).abs().max().item()
                        self.assertLessEqual(
                            gap,
                            budget,
                            f"{label}/{mode}: sharded logits differ by {gap:.4g}, beyond this "
                            f"model's own rounding budget {budget:.4g} — that is a sharding bug, "
                            f"not reduction order",
                        )
                        if noise_floor <= 2e-2:  # rounding alone does not move the argmax
                            self.assertTrue(
                                torch.equal(payload["logits"].argmax(-1), reference.argmax(-1)),
                                f"{label}/{mode}: argmax diverged from the unsharded model",
                            )
