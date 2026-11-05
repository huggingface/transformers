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
    Mxfp4Config,
    NVFP4Config,
)
from transformers.quantizers.auto import AutoHfQuantizer, QuantizationPlanHfQuantizer, get_hf_quantizer
from transformers.testing_utils import require_kernels, require_torch, require_torch_gpu, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from safetensors.torch import load_file, save_file

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
# deepseek-ai/DeepSeek-V4-Flash: block fp8, MXFP4 routed experts (`expert_dtype: fp4` in the model config). V4.1 has it
# in its quantization config
DEEPSEEK_V4 = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}
DEEPSEEK_V4_1 = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [32, 32], "expert_dtype": "fp4"}
TINY_DEEPSEEK = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "vocab_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
}
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


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
        plan.match_modules(model)
        plan.postprocess_model(model)
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
            (
                MISTRAL,
                dict.fromkeys(list(MISTRAL["quantized_layers"])[:2], FineGrainedFP8Config(activation_scheme="none")),
            ),
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
        hf_quantizer.match_modules(model)
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
        plan.match_modules(model)
        self.assertIs(plan.get_quantizer("model.layers.1.mlp.experts.down_proj"), nvfp4)
        self.assertIsNone(plan.get_quantizer("model.layers.11.mlp.experts.down_proj"))

    def test_fp4_experts(self):
        for saved, expert_dtype in [(DEEPSEEK_V4, "fp4"), (DEEPSEEK_V4_1, None)]:
            config = DeepseekV4Config(**TINY_DEEPSEEK, expert_dtype=expert_dtype, quantization_config=dict(saved))
            hf_quantizer, config = self.plan_from_config(config.quantization_config, config)
            fp8 = FineGrainedFP8Config(**{k: v for k, v in saved.items() if k != "expert_dtype"})
            plan = {"*.mlp.experts": Mxfp4Config(), "*": fp8}
            self.assertEqual(
                {k: repr(v) for k, v in config.quantization_config.items()}, {k: repr(v) for k, v in plan.items()}
            )
            with torch.device("meta"):
                model = DeepseekV4ForCausalLM(config)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            ):
                hf_quantizer.preprocess_model(model, use_kernels=False)
            # The fused experts the FP8 quantizer used to build, now MXFP4's
            experts = model.model.layers[0].mlp.experts
            self.assertEqual(type(experts).__name__, "Mxfp4Experts")
            self.assertEqual(
                hf_quantizer.get_quantizer("model.layers.0.mlp.experts.gate_up_proj").quantization_config.quant_method,
                "mxfp4",
            )
            self.assertEqual((experts.gate_up_proj.shape, experts.gate_up_proj.dtype), ((4, 64, 32), torch.int8))
            self.assertEqual(
                (experts.down_proj_scale_inv.shape, experts.down_proj_scale_inv.dtype),
                ((4, 64, 1), torch.float8_e8m0fnu),
            )
            self.assertEqual(type(model.model.layers[0].self_attn.q_a_proj).__name__, "FP8Linear")
        # ModelOpt's NVFP4 experts win over DeepSeek's `expert_dtype`, which the config copies into theirs too
        config = DeepseekV4Config(**TINY_DEEPSEEK, expert_dtype="fp4", quantization_config=dict(DEEPSEEK))
        hf_quantizer = self.plan_from_config(config.quantization_config, config)[0]
        methods = {q.quantization_config.quant_method for q in hf_quantizer.all_quantizers}
        self.assertEqual(methods, {"nvfp4", "fp8"})

    def test_fp4_experts_load_dequantize(self):
        # A DeepSeek-V4 checkpoint: per-expert e2m1 pairs in `int8`, a ue8m0 `.scale` per 32
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DEEPSEEK)).save_pretrained(tmp)
            state_dict, expected = load_file(f"{tmp}/model.safetensors"), {}
            for key in [k for k in state_dict if ".experts." in k]:
                rows, cols = state_dict[key].shape
                low, high = torch.randint(0, 16, (2, rows, cols // 2))
                exponents = torch.randint(-8, -2, (rows, cols // 32)).float()
                state_dict[key] = (low | high << 4).to(torch.uint8).view(torch.int8)
                state_dict[key.replace("weight", "scale")] = exponents.exp2().to(torch.float8_e8m0fnu)
                values = torch.stack([torch.tensor(FP4_VALUES)[low], torch.tensor(FP4_VALUES)[high]], -1).flatten(1)
                expected[key] = values * exponents.exp2().repeat_interleave(32, 1)
            save_file(state_dict, f"{tmp}/model.safetensors", metadata={"format": "pt"})
            DeepseekV4Config(**TINY_DEEPSEEK, expert_dtype="fp4", quantization_config=DEEPSEEK_V4).save_pretrained(tmp)
            model, info = DeepseekV4ForCausalLM.from_pretrained(
                tmp, dtype=torch.bfloat16, quantization_config={"dequantize": True}, output_loading_info=True
            )
        self.assertFalse(any(info.values()))
        for i, layer in enumerate(model.model.layers):
            experts = [
                [expected[f"model.layers.{i}.ffn.experts.{e}.w{n}.weight"] for e in range(4)] for n in (1, 2, 3)
            ]
            gate_up = torch.stack([torch.cat(pair) for pair in zip(experts[0], experts[2])])
            torch.testing.assert_close(layer.mlp.experts.gate_up_proj, gate_up.to(torch.bfloat16))
            torch.testing.assert_close(layer.mlp.experts.down_proj, torch.stack(experts[1]).to(torch.bfloat16))

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
