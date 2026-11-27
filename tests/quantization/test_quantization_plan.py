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
    BitsAndBytesConfig,
    DeepseekV4Config,
    FineGrainedFP8Config,
    LlamaConfig,
    LlamaForCausalLM,
    Mistral4ForCausalLM,
    NVFP4Config,
)
from transformers.quantizers.auto import (
    AutoHfQuantizer,
    QuantizationPlanHfQuantizer,
    get_hf_quantizer,
)
from transformers.testing_utils import require_kernels, require_torch, require_torch_gpu, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


# A tiny Mistral4 saved pre-quantized like Mistral-Large-4's NVFP4 release: fp8 attention, nvfp4 shared linears.
# `dense_reference.pt` holds the bf16 weights it was quantized from.
REPO = "hf-internal-testing/tiny-random-Mistral4-fp8-nvfp4"

TINY_DEEPSEEK = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "vocab_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
}


@require_torch
class QuantizationPlanTest(unittest.TestCase):
    def dense_reference(self):
        return torch.load(hf_hub_download(REPO, "dense_reference.pt"))  # TODO: should be a safetensor?

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

    def test_rejects_mixing_formats_swapping_their_modules_themselves(self):
        # bitsandbytes swaps every linear itself: mixed in a plan, it would take the other format's layers
        bnb = AutoHfQuantizer.from_config(BitsAndBytesConfig(load_in_8bit=True))
        fp8 = AutoHfQuantizer.from_config(FineGrainedFP8Config(), pre_quantized=True)
        with self.assertRaisesRegex(ValueError, "can't be mixed"):
            QuantizationPlanHfQuantizer({"*.mlp.*": bnb, "*": fp8})
        QuantizationPlanHfQuantizer({"*": bnb})  # alone, it swaps its own modules as before

    def test_loading_attributes_keep_the_saved_config_class(self):
        # A saved FP8 plan takes `FineGrainedFP8Config(dequantize=True)`'s loading attributes, not another format's
        saved = {"*.mlp.*": {"quant_method": "fp8", "weight_block_size": [128, 128]}, "*": None}
        plan = AutoHfQuantizer.from_plan(saved, FineGrainedFP8Config(dequantize=True), pre_quantized=True)
        self.assertTrue(plan.quantizers["*.mlp.*"].quantization_config.dequantize)
        with self.assertRaisesRegex(ValueError, "quantized with FineGrainedFP8Config but you are passing a"):
            AutoHfQuantizer.from_plan(saved, BitsAndBytesConfig(load_in_8bit=True), pre_quantized=True)

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
        plan.map_module_to_quantizer(model)
        plan.postprocess_model(model)
        self.assertIsNone(model.config.quantization_config["*.self_attn.*"])  # stays, so no broader pattern claims it
        self.assertIsNotNone(model.config.quantization_config["*.mlp.*"])
        self.assertIsNone(plan.get_quantizer("model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(plan.is_serializable())  # NVFP4 can't be saved

    def test_skip_list_folds_into_the_plan(self):
        # An exact name, a prefix and a suffix: `gate` skips the router, not `gate_proj`
        skips = ["model.layers.1.self_attn.q_proj", "model.layers.0.mlp", "gate"]
        plan = AutoHfQuantizer.from_plan({"*": FineGrainedFP8Config(modules_to_not_convert=skips)}, None, True)
        with torch.device("meta"):
            model = LlamaForCausalLM(
                LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, vocab_size=128)
            )
            model.model.layers[1].mlp.gate = torch.nn.Linear(64, 4, bias=False)
        plan.preprocess_model(model)
        skipped = ["model.layers.1.self_attn.q_proj", "model.layers.1.mlp.gate"]
        skipped += [f"model.layers.0.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")]
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or hasattr(module, "source_cls"):
                expected = "Linear" if name in skipped else "FP8Linear"
                self.assertEqual(type(module).__name__, expected, name)
        self.assertEqual(type(model.model.layers[1].mlp.gate_proj).__name__, "FP8Linear")
        self.assertEqual(list(plan.quantizers), ["*"])  # the skips stay out of the saved plan

    def test_param_needs_quantization_asks_the_param_quantizer(self):
        # The loading's shape check asks the plan: it must answer as the param's quantizer, or the quantized weights
        # of bitsandbytes, whose shape differs from the module's, are reported mismatched
        plan = AutoHfQuantizer.from_plan({"*.mlp.*": FineGrainedFP8Config(), "*": None}, None, pre_quantized=False)
        with torch.device("meta"):
            model = LlamaForCausalLM(
                LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1, vocab_size=128)
            )
        plan.preprocess_model(model)
        self.assertTrue(plan.param_needs_quantization(model, "model.layers.0.mlp.up_proj.weight"))
        self.assertFalse(plan.param_needs_quantization(model, "model.layers.0.self_attn.q_proj.weight"))

    def plan_from_config(self, quantization_config):
        config = DeepseekV4Config(**TINY_DEEPSEEK, quantization_config=dict(quantization_config))
        with patch.object(QuantizationPlanHfQuantizer, "validate_environment"):
            return get_hf_quantizer(config, None, None, True, {})[:2]

    def test_quantization_plan_parsing(self):
        # ModelOpt MIXED_PRECISION with FP8 and NVFP4 layers; exact checkpoint names beat the "*" fallback
        saved = {
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "layers.0.ffn.experts": {"quant_algo": "NVFP4", "group_size": 16},
                "layers.1.attention.wq": {"quant_algo": "FP8_PB_WO"},
            },
            "ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
        }
        hf_quantizer, config = self.plan_from_config(saved)
        self.assertEqual(len(hf_quantizer.all_quantizers), 2)
        self.assertIn("layers.0.mlp.experts", hf_quantizer.quantizers)
        self.assertIn("*", hf_quantizer.quantizers)
        nvfp4, fp8 = hf_quantizer.quantizers["layers.0.mlp.experts"], hf_quantizer.quantizers["*"]
        examples = [
            ("model.layers.0.mlp.experts.gate_up_proj", nvfp4),
            ("model.layers.1.mlp.experts.down_proj", fp8),
            ("model.layers.1.self_attn.q_proj.weight", fp8),
        ]
        for name, expected in examples:
            self.assertIs(hf_quantizer.get_quantizer(name), expected, name)
        # The base quantizer covers everything ModelOpt left out, its `ignore` included
        self.assertNotIn("head", hf_quantizer.quantization_config)

    def test_modelopt_single_algo_and_weight_only(self):
        saved = {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16, "ignore": ["lm_head"]}
        plan = self.plan_from_config(saved)[0].quantization_config
        self.assertEqual(list(plan), ["lm_head", "*"])
        self.assertIsNone(plan["lm_head"])
        self.assertEqual(repr(plan["*"]), repr(NVFP4Config()))

        saved = {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": "FP8"}
        with self.assertLogs("transformers.quantizers.schema_readers", level="WARNING") as logs:
            config = self.plan_from_config(saved)[0].quantization_config
        self.assertEqual(repr(config), repr(NVFP4Config(weight_only=True)))
        self.assertTrue(any("kv_cache_quant_algo=FP8" in line for line in logs.output))

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
