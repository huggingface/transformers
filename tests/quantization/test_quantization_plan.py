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
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub import hf_hub_download

from transformers import (
    AutoConfig,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    FineGrainedFP8Config,
    LlamaConfig,
    LlamaForCausalLM,
    Mistral4ForCausalLM,
    NVFP4Config,
)
from transformers.quantizers.auto import AutoHfQuantizer, QuantizationPlanHfQuantizer, get_hf_quantizer
from transformers.testing_utils import require_kernels, require_torch, require_torch_gpu, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

# A tiny Mistral4 saved pre-quantized like Mistral-Large-4's NVFP4 release: fp8 attention and experts, nvfp4 shared
# experts. `dense_reference.pt` holds the bf16 weights it was quantized from.
REPO = "hf-internal-testing/tiny-random-Mistral4-fp8-nvfp4"

# Trimmed `quantization_config`s of released ModelOpt checkpoints, quantized per layer
NVFP4 = {"quant_algo": "NVFP4", "group_size": 16}
QWEN = {  # nvidia/Qwen3.6-35B-A3B-NVFP4
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
        "model.language_model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
        "model.language_model.layers.3.self_attn.q_proj": {"quant_algo": "FP8"},
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
    },
    "ignore": ["mtp.layers.0*", "mtp*"],
}
DEEPSEEK = {  # nvidia/DeepSeek-V4-Flash-NVFP4: block fp8, but ModelOpt's NVFP4 experts
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
    "quant_algo": "MIXED_PRECISION",
    "moe_quant_algo": "NVFP4",
    "quantized_layers": {"layers.0.ffn.experts": NVFP4, "layers.1.ffn.experts": NVFP4},
    "ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
}
MISTRAL = {  # mistralai/Mistral-Large-4-1T-A52B-NVFP4 (params.json)
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "group_size": 16,
    "ignore": [],
    "quantized_layers": {
        "layers.0.attention.wq": {"quant_algo": "FP8_PB_WO"},
        "layers.0.feed_forward.w1": {"quant_algo": "FP8_PB_WO"},
        "layers.3.experts": NVFP4,
        "layers.4.experts": NVFP4,
    },
}


@require_torch
class QuantizationPlanTest(unittest.TestCase):
    def dense_reference(self):
        return torch.load(hf_hub_download(REPO, "dense_reference.pt"))

    def check_logits(self, model):
        config = AutoConfig.from_pretrained(REPO)
        del config.quantization_config
        dense = Mistral4ForCausalLM(config).to("cuda", torch.bfloat16).eval()
        dense.load_state_dict(self.dense_reference())
        # One token: the sm120 NVFP4 GEMM fails for more rows, its decode GEMV works
        input_ids = torch.randint(0, config.vocab_size, (1, 1), generator=torch.Generator().manual_seed(0)).cuda()
        with torch.no_grad():
            expected, logits = dense(input_ids).logits.float(), model(input_ids).logits.float()
        self.assertLess(((logits - expected).norm() / expected.norm()).item(), 0.2)

    def test_dequantized(self):
        # Every pattern of the checkpoint's plan, without restating its recipe
        model = Mistral4ForCausalLM.from_pretrained(
            REPO, dtype=torch.bfloat16, quantization_config={"dequantize": True}
        )
        # Each weight dequantized from its own format: fp8 ~2.6%, nvfp4 ~9.5% away from the bf16 it came from
        for name, weight in self.dense_reference().items():
            loaded = model.state_dict()[name]
            self.assertEqual(loaded.dtype, torch.bfloat16, name)
            self.assertLess(((loaded.float() - weight.float()).norm() / weight.float().norm()).item(), 0.12, name)
        # Every pattern dequantized: the plan leaves the model, which saves and reloads plain unquantized
        self.assertFalse(model.is_quantized)
        self.assertFalse(hasattr(model.config, "quantization_config"))
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = Mistral4ForCausalLM.from_pretrained(tmp, dtype=torch.bfloat16)
        self.assertFalse(hasattr(reloaded.config, "quantization_config"))
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[name], weight, msg=name)

    def test_postprocess_drops_dequantized_patterns(self):
        plan = QuantizationPlanHfQuantizer(
            {
                "*.self_attn.*": AutoHfQuantizer.from_config(
                    FineGrainedFP8Config(dequantize=True), pre_quantized=True
                ),
                "*.mlp.*": AutoHfQuantizer.from_config(NVFP4Config(), pre_quantized=True),
            }
        )
        model = LlamaForCausalLM(
            LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1, vocab_size=128)
        )
        with patch("transformers.integrations.finegrained_fp8._disable_deepgemm_on_multi_device") as fp8_hook:
            plan.postprocess_model(model)
        fp8_hook.assert_called_once_with(model)  # each quantizer's after-load hook runs
        self.assertIsNone(model.config.quantization_config["*.self_attn.*"])  # stays, so no broader pattern claims it
        self.assertIsNotNone(model.config.quantization_config["*.mlp.*"])
        self.assertIsNone(plan.get_quantizer("model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(plan.is_serializable())  # NVFP4 can't be saved

    def plan_from_config(self, quantization_config, config=None):
        config = config or DeepseekV4Config(
            num_hidden_layers=2, hidden_size=64, vocab_size=128, n_routed_experts=4, moe_intermediate_size=32
        )
        config.quantization_config = quantization_config
        with patch.object(QuantizationPlanHfQuantizer, "validate_environment"):  # NVFP4 checks for a Blackwell GPU
            return get_hf_quantizer(config, None, None, True, {})[:2]

    def test_modelopt_per_layer_configs(self):
        fp8_static = FineGrainedFP8Config(activation_scheme="static", weight_block_size=None)
        for saved, plan in [
            (
                QWEN,
                {"mtp.layers.0*": None, "mtp*": None} | dict.fromkeys(list(QWEN["quantized_layers"])[:2], fp8_static),
            ),
            (DEEPSEEK, dict.fromkeys(DEEPSEEK["quantized_layers"], NVFP4Config())),
            (MISTRAL, dict.fromkeys(list(MISTRAL["quantized_layers"])[:2], FineGrainedFP8Config())),
        ]:
            hf_quantizer, config = self.plan_from_config(saved)
            # Each layer keeps its own config, one quantizer per distinct config
            self.assertEqual(len(hf_quantizer.all_quantizers), 2)
            self.assertTrue(hf_quantizer.pre_quantized)
            for pattern, expected in plan.items():
                self.assertEqual(repr(config.quantization_config[pattern]), repr(expected), pattern)
        # DeepSeek's base fp8 quantizes everything ModelOpt left out, its `ignore` included
        fallback = self.plan_from_config(DEEPSEEK)[1].quantization_config
        self.assertEqual(list(fallback)[-1], "*")
        self.assertEqual((fallback["*"].quant_method, fallback["*"].scale_fmt), ("fp8", "ue8m0"))
        self.assertNotIn("head", fallback)

    def test_modelopt_checkpoint_names(self):
        hf_quantizer, config = self.plan_from_config(DEEPSEEK)
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
        hf_quantizer.rename_patterns(model)
        # The checkpoint's `layers.0.ffn.experts` is the model's `layers.0.mlp.experts`, its fused experts
        nvfp4, fp8 = hf_quantizer.quantizers["layers.0.mlp.experts"], hf_quantizer.quantizers["*"]
        self.assertEqual(
            (nvfp4.quantization_config.quant_method, fp8.quantization_config.quant_method), ("nvfp4", "fp8")
        )
        self.assertEqual(model.config.quantization_config, hf_quantizer.quantization_config)
        # The exact names win over the `"*"` fallback
        self.assertIs(hf_quantizer.get_quantizer("model.layers.0.mlp.experts.gate_up_proj"), nvfp4)
        self.assertIs(hf_quantizer.get_quantizer("model.layers.1.mlp.experts.down_proj"), nvfp4)
        self.assertIs(hf_quantizer.get_quantizer("model.layers.0.self_attn.q_a_proj.weight"), fp8)
        self.assertIs(hf_quantizer.get_quantizer("model.layers.0.mlp.shared_experts.gate_proj.weight"), fp8)
        self.assertIs(hf_quantizer.get_quantizer("lm_head.weight"), fp8)
        # `None` leaves the modules unquantized, an exact name doesn't match a longer one
        plan = QuantizationPlanHfQuantizer({"layers.1.mlp": nvfp4, "*.mlp.*": None})
        self.assertIs(plan.get_quantizer("model.layers.1.mlp.experts.down_proj"), nvfp4)
        self.assertIsNone(plan.get_quantizer("model.layers.11.mlp.experts.down_proj"))

    @slow
    @require_torch_gpu
    @require_kernels
    def test_quantized(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 needs a Blackwell GPU")
        plan = {
            "*.self_attn.*_proj*": FineGrainedFP8Config(weight_block_size=(32, 32)),
            "*.mlp.experts.*": FineGrainedFP8Config(weight_block_size=(128, 128)),
            "*.mlp.shared_experts.*": NVFP4Config(),
        }
        model = Mistral4ForCausalLM.from_pretrained(
            REPO, dtype=torch.bfloat16, device_map="cuda", quantization_config=plan
        )
        layer = model.model.layers[0]
        self.assertEqual(type(layer.self_attn.q_b_proj).__name__, "FP8Linear")
        self.assertEqual(type(layer.mlp.experts).__name__, "FP8Experts")
        self.assertEqual(type(layer.mlp.shared_experts.gate_proj).__name__, "NVFP4Linear")
        self.assertEqual(type(model.lm_head), torch.nn.Linear)
        self.check_logits(model)

    @slow
    @require_torch_gpu
    @require_kernels
    def test_fp8_quantized_nvfp4_dequantized(self):
        plan = {
            "*.self_attn.*_proj*": FineGrainedFP8Config(weight_block_size=(32, 32)),
            "*.mlp.experts.*": FineGrainedFP8Config(weight_block_size=(128, 128)),
            "*.mlp.shared_experts.*": NVFP4Config(dequantize=True),
        }
        model = Mistral4ForCausalLM.from_pretrained(
            REPO, dtype=torch.bfloat16, device_map="cuda", quantization_config=plan
        )
        layer = model.model.layers[0]
        self.assertEqual(type(layer.mlp.experts).__name__, "FP8Experts")
        self.assertEqual(type(layer.mlp.shared_experts.gate_proj), torch.nn.Linear)
        self.check_logits(model)
