# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from parameterized import parameterized

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DeepseekV4Config,
    FineGrainedFP8Config,
    OPTForCausalLM,
)
from transformers.distributed import DistributedConfig
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.testing_utils import (
    get_device_properties,
    require_accelerate,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_memory_cleanup_mixin import MemoryCleanupMixin, with_grad


if is_torch_available():
    import torch


@contextmanager
def _patch_no_accelerator():
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        if hasattr(torch, "xpu"):
            stack.enter_context(patch("torch.xpu.is_available", return_value=False))
            stack.enter_context(
                patch("transformers.quantizers.quantizer_finegrained_fp8.is_torch_xpu_available", return_value=False)
            )
        yield


@require_torch_accelerator
class FineGrainedFP8ConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = FineGrainedFP8Config()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["lm_head.weight"], "quant_method": "fp8"}
        quantization_config = FineGrainedFP8Config.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)

    def test_megamoe_takes_over_the_experts_ep_rule(self):
        """Mega MoE dispatches inside its kernel, so it replaces the token-dispatch experts rule of the EP plan."""
        config = DeepseekV4Config()
        config._experts_implementation = "deepgemm_megamoe"
        config = FineGrainedFP8HfQuantizer(FineGrainedFP8Config()).update_tp_plan(config)
        self.assertEqual(config.base_model_ep_plan["layers.*.mlp.experts"], "megamoe_experts")
        with self.assertRaisesRegex(ValueError, "Mega MoE expert parallelism requires `ep_size=tp_size`"):
            DistributedConfig(tp_size=1, fsdp_size=8, ep_size=8)._validate_resolved_ep_plan(config.base_model_ep_plan)


@slow
@require_accelerate
@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8QuantizerTest because it is not supported on GPU with capability < 8.9",
)
class FP8QuantizerTest(MemoryCleanupMixin, unittest.TestCase):
    model_name = "meta-llama/Llama-3.2-1B"
    quantized_model_name = "hf-internal-testing/Llama-3.2-1B-Instruct-fp8"
    input_text = "Once upon a time"
    max_new_tokens = 10
    EXPECTED_OUTPUTS = {
        "Once upon a time, there was a little girl who loved to play",
        "Once upon a time, there was a man who was very rich.",
    }
    EXPECTED_DEQUANTIZED_OUTPUT = "Once upon a time, in a small village nestled in the rolling hills"
    device_map = torch_device
    offload_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": "cpu",
        "model.layers.8": "cpu",
        "model.layers.9": "cpu",
        "model.layers.10": "cpu",
        "model.layers.11": "cpu",
        "model.layers.12": "cpu",
        "model.layers.13": "cpu",
        "model.layers.14": "cpu",
        "model.layers.15": "cpu",
        "model.rotary_emb": "cpu",
        "model.norm": "cpu",
        "lm_head": 0,
    }

    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        super().setUpClass()
        cls.quantization_config = FineGrainedFP8Config()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=cls.quantization_config
        )

    @parameterized.expand(
        [
            "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
            "hf-internal-testing/tiny-random-MixtralForCausalLM",
        ]
    )
    def test_moe_conversion_doesnt_raise(self, model_id):
        quantization_config = FineGrainedFP8Config(weight_block_size=(32, 32))
        AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """

        from transformers.integrations import FP8Linear
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = FineGrainedFP8Config()

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1
        FineGrainedFP8HfQuantizer(quantization_config).replace_modules(model)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1
        self.assertEqual(nb_linears, nb_fp8_linear)
        with torch.device("meta"):
            model = OPTForCausalLM(config)
        quantization_config = FineGrainedFP8Config()
        quantizer = FineGrainedFP8HfQuantizer(quantization_config)
        quantizer.modules_to_not_convert = ["fc1"]
        quantizer.replace_modules(model)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1
        self.assertEqual(nb_linears - 24, nb_fp8_linear)

    def test_quantizer_validation_no_accelerator(self):
        """Test quantizer validation when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            config = FineGrainedFP8Config()
            quantizer = FineGrainedFP8HfQuantizer(config)
            quantizer.pre_quantized = False

            with self.assertRaises(RuntimeError):
                quantizer.validate_environment()

    def test_dequantization_no_accelerator(self):
        """Test dequantization when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            config = FineGrainedFP8Config()
            quantizer = FineGrainedFP8HfQuantizer(config)
            quantizer.pre_quantized = True
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertIn(output_tokens, self.EXPECTED_OUTPUTS)

    def test_dequantized_model(self):
        """
        Simple test that checks if the dequantized model is working properly
        """
        quantization_config = FineGrainedFP8Config(dequantize=True)
        dequantized_model = AutoModelForCausalLM.from_pretrained(
            self.quantized_model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
        del dequantized_model

    def test_dequantize_when_no_accelerator(self):
        """
        Simple test that checks if the dequantized model is working properly when no accelerator is available
        """
        with _patch_no_accelerator():
            dequantized_model = AutoModelForCausalLM.from_pretrained(self.quantized_model_name, device_map="cpu")
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to("cpu")
            output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
            del dequantized_model

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_weight_and_weight_scale_inv(self):
        """
        Simple test that checks if the weight and weight_scale_inv are working properly
        """
        weight = self.quantized_model.model.layers[0].self_attn.q_proj.weight
        weight_scale_inv = self.quantized_model.model.layers[0].self_attn.q_proj.weight_scale_inv
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(weight_scale_inv.dtype, torch.float32)
        self.assertEqual(weight.shape, (weight_scale_inv.shape[0] * 128, weight_scale_inv.shape[1] * 128))

    def test_block_size(self):
        """
        Simple test that checks if the block size is working properly
        """
        self.assertEqual(self.quantized_model.config.quantization_config.weight_block_size, (128, 128))
        quantization_config = FineGrainedFP8Config(weight_block_size=(32, 32))
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        self.assertEqual(quantized_model.config.quantization_config.weight_block_size, (32, 32))

    @require_torch_multi_accelerator
    def test_quantized_model_multi_accelerators(self):
        """
        Simple test that checks if the quantized model is working properly with multiple accelerators
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs; or set ZE_AFFINITY_MASK=0,1 if you
        have more than 2 XPUs.
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        quantization_config = FineGrainedFP8Config()
        # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
        # this will lead to put the entire model to device 0.
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory={0: "1GB", 1: "10GB"},
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    @require_torch_multi_accelerator
    def test_save_pretrained_multi_accelerators(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
            # this will lead to put the entire model to device 0.
            model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, device_map="auto", max_memory={0: "1GB", 1: "10GB"}
            )
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_quantized_model_offload(self):
        """
        Simple test that checks if the quantized model returns an error when loading with cpu/disk offloaded
        """
        with self.assertRaisesRegex(
            ValueError, "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
        ):
            AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.offload_device_map, quantization_config=self.quantization_config
            )

    def test_save_pretrained_offload(self):
        """
        Simple test that checks if the saved quantized model is working properly cpu/disk offload
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.offload_device_map)
            output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_compute_module_sizes(self):
        r"""
        Test if we compute the right module sizes needed to generate the device map.
        Also test if we get the right values for `total_byte_count` in `caching_allocator_warmup`.
        """
        from transformers.integrations import FP8Linear
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.modeling_utils import expand_device_map, get_total_byte_count
        from transformers.quantizers import AutoHfQuantizer

        # we need to preprocess the model like that because device_map calculation happens before we load the weights inside the model.
        # For normal wieghts, it's fine but for quantized weights, the tensors dtype might change during loading.
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
            model_size, _ = compute_module_sizes(model, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            total_byte_count = list(get_total_byte_count(model, expanded_device_map).values())[0]

            # testing prequantized = False should be enough, the shape should be the same whether it is pre-quantized or not
            hf_quantizer = AutoHfQuantizer.from_config(FineGrainedFP8Config(), pre_quantized=False)
            hf_quantizer.preprocess_model(model=model, config=model.config)
            quantized_model_size, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            quantized_total_byte_count = list(get_total_byte_count(model, expanded_device_map, hf_quantizer).values())[
                0
            ]

        for name, module in model.named_modules():
            if isinstance(module, FP8Linear):
                # from 16 bits to 8 bits
                assert int(model_size[f"{name}.weight"] // 2) == int(quantized_model_size[f"{name}.weight"])

        # check that we get the same value, as we use `compute_module_sizes` in `get_total_byte_count`
        assert total_byte_count == model_size[""]
        assert quantized_total_byte_count == quantized_model_size[""]

        # we should at least have 1.5 times memory reduction in total
        assert model_size[""] > quantized_model_size[""] * 1.5

    @parameterized.expand(["eager", "batched_mm", "grouped_mm", "deepgemm"])
    def test_quantized_moe_forward(self, experts_implementation):
        """
        Checks implicitly if the moe implementation is correct, i.e. it does not crash for cases
        where the indices go over `top_k` as shown within the Minimax M2 model
        """
        # deepgemm only has CUDA kernels, skip on other devices
        if experts_implementation == "deepgemm" and torch_device != "cuda":
            self.skipTest("deepgemm is only supported on CUDA")

        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/MiniMax-M2-Tiny-FP8",  # single layer version
            experts_implementation=experts_implementation,
            device_map=self.device_map,
        )
        assert model.config._experts_implementation == experts_implementation

        tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]},
        ]
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            self.device_map
        )

        # Only caring about this not crashing
        _ = model.generate(**model_inputs, max_new_tokens=24)

    @parameterized.expand(["eager", "grouped_mm"])
    @with_grad
    def test_quantized_moe_backward(self, experts_implementation):
        """
        Checks that the FP8 layers pass the input gradient back through their dequantized weights, so adapters can
        train on top of them: the gradient matches the one of the dequantized model
        """
        model_id = "hf-internal-testing/MiniMax-M2-Tiny-FP8"
        input_ids = torch.arange(10, 74, device=self.device_map).unsqueeze(0)
        grads = []
        for kwargs in [
            {"experts_implementation": experts_implementation},
            {"quantization_config": FineGrainedFP8Config(dequantize=True)},
        ]:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device_map, **kwargs)
            inputs_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_()
            model(inputs_embeds=inputs_embeds, labels=input_ids).loss.backward()
            grads.append(inputs_embeds.grad.float())

        grad, expected = grads
        # The FP8 forward also quantizes the activations, hence the tolerance
        self.assertLess(((grad - expected).norm() / expected.norm()).item(), 0.1)


@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8LinearTest because it is not supported on GPU with capability < 8.9",
)
class FP8LinearTest(unittest.TestCase):
    device = torch_device

    def test_linear_preserves_shape(self):
        """
        Test that FP8Linear preserves shape when in_features == out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(256, 256, block_size=(128, 128)).to(self.device)
        x = torch.rand((1, 5, 256)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, x.shape)

    def test_linear_with_diff_feature_size_preserves_shape(self):
        """
        Test that FP8Linear generates the correct shape when in_features != out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(128, 256, block_size=(128, 128)).to(self.device)
        x = torch.rand((1, 5, 128)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, (1, 5, 256))

    def test_linear_backward_goes_through_dequantized_weight(self):
        """
        Test that FP8Linear passes the input gradient through its dequantized weight, so adapters can train on top of it.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(256, 384, block_size=(128, 128)).to(self.device)
        linear.weight.data = torch.randn(384, 256).to(self.device, torch.float8_e4m3fn)
        linear.weight_scale_inv.data = torch.rand(3, 2).to(self.device) + 0.5
        x = torch.randn((1, 5, 256), dtype=torch.bfloat16).to(self.device).requires_grad_()

        linear(x).sum().backward()

        scale = linear.weight_scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        expected = torch.ones(1, 5, 384, device=self.device) @ (linear.weight.float() * scale)
        torch.testing.assert_close(x.grad.float(), expected, rtol=1e-2, atol=1e-1)

    def test_grouped_linear_backward_goes_through_dequantized_weight(self):
        """
        Test that FP8GroupedLinear passes the input gradient through its dequantized weight, group by group.
        """
        from transformers.integrations.finegrained_fp8 import FP8GroupedLinear

        linear = FP8GroupedLinear(256, 512, n_groups=2, block_size=(128, 128)).to(self.device)
        linear.weight.data = torch.randn(512, 256).to(self.device, torch.float8_e4m3fn)
        linear.weight_scale_inv.data = torch.rand(4, 2).to(self.device) + 0.5
        x = torch.randn((3, 2, 256), dtype=torch.bfloat16).to(self.device).requires_grad_()

        linear(x).sum().backward()

        scale = linear.weight_scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        weight = (linear.weight.float() * scale).view(2, 256, 256)
        expected = torch.ones(3, 2, 1, 256, device=self.device) @ weight
        torch.testing.assert_close(x.grad.float(), expected.squeeze(2), rtol=1e-2, atol=1e-1)


class FP8EmbeddingTest(unittest.TestCase):
    """
    Some checkpoints quantize an embedding *table* rather than a linear (Qwen4-Exp's n-gram/PLE
    table).
    """

    def test_lookup_applies_the_per_tensor_scale(self):
        """Without the rescale the rows come back as raw FP8 and blow up the next matmul."""
        from transformers.integrations import FP8Embedding

        table = torch.randn(64, 16, dtype=torch.bfloat16)
        scale = table.abs().max().float() / torch.finfo(torch.float8_e4m3fn).max
        embedding = FP8Embedding(64, 16)
        embedding.weight = torch.nn.Parameter((table.float() / scale).to(torch.float8_e4m3fn), requires_grad=False)
        embedding.weight_scale_inv = torch.nn.Parameter(scale.to(torch.bfloat16).reshape(1), requires_grad=False)

        ids = torch.randint(0, 64, (2, 5))
        rows, expected = embedding(ids), table[ids]

        self.assertEqual(rows.dtype, expected.dtype)
        error = (rows.float() - expected.float()).abs().max() / expected.float().abs().max()
        self.assertLess(error, 0.1)

    def test_patterns_match_by_suffix_and_honour_the_skip_list(self):
        """Suffixes match whether or not the text model is nested under a multimodal wrapper."""
        from transformers.integrations import FP8Embedding
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

        model = torch.nn.ModuleDict(
            {n: torch.nn.ModuleDict({"table": torch.nn.Embedding(8, 4)}) for n in ("layers", "language_model")}
        )
        model.other = torch.nn.Embedding(8, 4)
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(modules_to_convert=["table"]), pre_quantized=True)
        quantizer.modules_to_not_convert = ["language_model.table"]
        quantizer.replace_modules(model)

        self.assertIsInstance(model["layers"]["table"], FP8Embedding)  # matched by suffix
        self.assertIs(type(model["language_model"]["table"]), torch.nn.Embedding)  # skip-listed
        self.assertIs(type(model.other), torch.nn.Embedding)  # no pattern match


class FP8DeepGEMMDeviceTest(unittest.TestCase):
    def test_only_the_first_device_runs_deepgemm(self):
        from transformers.integrations import deepgemm

        with patch.object(deepgemm, "_DEEPGEMM_DEVICE", None):
            self.assertTrue(deepgemm.deepgemm_allowed(torch.device("cuda:0")))
            self.assertTrue(deepgemm.deepgemm_allowed(torch.device("cuda:0")))
            self.assertFalse(deepgemm.deepgemm_allowed(torch.device("cuda:1")))


class FP8Ue8m0LoadingTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "needs torch.float8_e8m0fnu")
    @patch.object(FineGrainedFP8HfQuantizer, "validate_environment", lambda *args, **kwargs: None)
    def test_float32_scale_container_loads_as_ue8m0(self):
        # dsv4-flash-base ships power-of-two ue8m0 scales as float32 under `.scale`; they must load as e8m0
        from safetensors.torch import load_file, save_file

        from transformers import MixtralConfig, MixtralForCausalLM

        config = MixtralConfig(
            vocab_size=64, hidden_size=128, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
        )  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            MixtralForCausalLM(config).save_pretrained(f"{tmp}/bf16")
            quantized = AutoModelForCausalLM.from_pretrained(
                f"{tmp}/bf16", quantization_config=FineGrainedFP8Config(scale_fmt="ue8m0"), device_map="cpu"
            )
            expected = {k: v.float() for k, v in quantized.state_dict().items() if k.endswith("scale_inv")}
            quantized.save_pretrained(f"{tmp}/fp8")
            path = f"{tmp}/fp8/model.safetensors"
            state_dict = {
                k.replace("proj.weight_scale_inv", "proj.scale"): v.float() if k.endswith("scale_inv") else v
                for k, v in load_file(path).items()
            }
            save_file(state_dict, path, metadata={"format": "pt"})

            loaded = AutoModelForCausalLM.from_pretrained(f"{tmp}/fp8", device_map="cpu")
        scales = {k: v for k, v in loaded.state_dict().items() if k.endswith("scale_inv")}
        self.assertEqual(scales.keys(), expected.keys())
        for name, scale in scales.items():
            self.assertEqual(scale.dtype, torch.float8_e8m0fnu, name)
            torch.testing.assert_close(scale.float(), expected[name], rtol=0, atol=0)


def _save_modelopt_fp8(path, layers, quantization_config):
    """
    A tiny Llama saved like a ModelOpt FP8 export: each of `layers` as `weight ~= fp8 * weight_scale`, the scale
    `amax / 448` per 128x128 block, or per tensor with a static `input_scale` for `None`. Returns the bf16 weights.
    """
    from safetensors.torch import load_file, save_file

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=256, intermediate_size=384, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, quantization_config=quantization_config,
    )  # fmt: skip
    LlamaForCausalLM(config).save_pretrained(path)
    state_dict, dense = load_file(f"{path}/model.safetensors"), {}
    for name, block in layers.items():
        dense[name] = weight = state_dict[f"{name}.weight"]
        rows, columns = (block, block) if block else weight.shape  # one block per tensor
        blocks = weight.unflatten(1, (-1, columns)).unflatten(0, (-1, rows))
        scale = blocks.abs().amax(dim=(1, 3), keepdim=True) / 448  # saved 4-D per block: (out / 128, 1, in / 128, 1)
        state_dict[f"{name}.weight"] = (blocks / scale).flatten(2).flatten(0, 1).to(torch.float8_e4m3fn)
        state_dict[f"{name}.weight_scale"] = scale if block else scale.reshape(())
        if block is None and name.endswith("proj"):
            state_dict[f"{name}.input_scale"] = torch.tensor(0.02)
    save_file(state_dict, f"{path}/model.safetensors", metadata={"format": "pt"})
    return dense, state_dict


class FP8ModelOptLoadingTest(unittest.TestCase):
    FP8 = ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj")  # per tensor, static activations
    FP8_PB_WO = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj")  # per 128x128 block, weight-only
    LAYERS = dict.fromkeys(FP8) | dict.fromkeys(FP8_PB_WO, 128)
    CONFIG = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            name: {"quant_algo": "FP8" if block is None else "FP8_PB_WO"} for name, block in LAYERS.items()
        },
    }

    def test_load_dequantized(self):
        embedding = {"quant_method": "fp8", "modules_to_convert": ["embed_tokens"]}
        for layers, config, dequantize in [
            (self.LAYERS, self.CONFIG, {"dequantize": True}),
            ({"model.embed_tokens": None}, embedding, FineGrainedFP8Config(dequantize=True)),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                dense, _ = _save_modelopt_fp8(tmp, layers, config)
                model, info = AutoModelForCausalLM.from_pretrained(
                    tmp, quantization_config=dequantize, output_loading_info=True
                )
            self.assertFalse(any(info.values()), info)
            for name, weight in dense.items():  # a plain model, the FP8 embedding table included
                self.assertIn(type(model.get_submodule(name)), (torch.nn.Linear, torch.nn.Embedding), name)
                # Within one e4m3 step: 1/8 of the value, 2**-9 of the scale for subnormals
                error = (model.get_submodule(name).weight - weight).abs()
                self.assertTrue((error <= weight.abs() / 8 + weight.abs().amax() / 448 * 2**-9).all(), name)

    @patch.object(FineGrainedFP8HfQuantizer, "validate_environment", lambda *args, **kwargs: None)
    def test_quantized(self):
        from transformers.integrations import FP8Linear

        with tempfile.TemporaryDirectory() as tmp:
            dense, saved = _save_modelopt_fp8(tmp, self.LAYERS, self.CONFIG)
            model, info = AutoModelForCausalLM.from_pretrained(tmp, device_map="cpu", output_loading_info=True)
        self.assertFalse(any(info.values()), info)
        for name in dense:
            linear = model.get_submodule(name)
            self.assertIsInstance(linear, FP8Linear, name)
            torch.testing.assert_close(linear.weight.view(torch.uint8), saved[f"{name}.weight"].view(torch.uint8))
            scale = saved[f"{name}.weight_scale"]
            torch.testing.assert_close(linear.weight_scale_inv, scale.flatten(1) if scale.ndim == 4 else scale)
        attention = model.model.layers[0].self_attn.q_proj
        self.assertEqual((attention.activation_scheme, attention.block_size), ("static", None))
        torch.testing.assert_close(attention.activation_scale, torch.tensor(0.02))

        # Weight-only: the block-dequantized weight, in the activations' dtype
        mlp = model.model.layers[0].mlp.down_proj
        self.assertEqual((mlp.activation_scheme, mlp.block_size, mlp.activation_scale), ("none", (128, 128), None))
        x = torch.randn(2, 3, 384)
        mlp.bias = torch.nn.Parameter(torch.randn(256))
        dequantized = mlp.weight.float() * mlp.weight_scale_inv.repeat_interleave(128, 0).repeat_interleave(128, 1)
        torch.testing.assert_close(mlp(x), x @ dequantized.T + mlp.bias)
