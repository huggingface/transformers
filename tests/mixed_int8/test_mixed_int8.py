# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import tempfile
import unittest

from packaging import version

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.testing_utils import (
    is_torch_available,
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.utils.versions import importlib_metadata


if is_torch_available():
    import torch
    import torch.nn as nn

    class LoRALayer(nn.Module):
        """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only"""

        def __init__(self, module: nn.Module, rank: int):
            super().__init__()
            self.module = module
            self.adapter = nn.Sequential(
                nn.Linear(module.in_features, rank, bias=False),
                nn.Linear(rank, module.out_features, bias=False),
            )
            small_std = (2.0 / (5 * min(module.in_features, module.out_features))) ** 0.5
            nn.init.normal_(self.adapter[0].weight, std=small_std)
            nn.init.zeros_(self.adapter[1].weight)
            self.adapter.to(module.weight.device)

        def forward(self, input, *args, **kwargs):
            return self.module(input, *args, **kwargs) + self.adapter(input)


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class BaseMixedInt8Test(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "bigscience/bloom-1b7"

    # Constant values
    EXPECTED_RELATIVE_DIFFERENCE = (
        1.540025  # This was obtained on a Quadro RTX 8000 so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of the family.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        # Models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


class MixedInt8Test(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Int8Params

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_8bit = self.model_8bit.get_memory_footprint()

        self.assertAlmostEqual(mem_fp16 / mem_8bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(self.model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)

    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = self.model_8bit.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_generate_quality_config(self):
        r"""
        Test that loading the model with the config is equivalent
        """
        bnb_config = BitsAndBytesConfig()

        model_8bit_from_config = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config, device_map="auto"
        )

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = model_8bit_from_config.generate(
            input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10
        )

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_warns_save_pretrained(self):
        r"""
        Test whether trying to save a model after converting it in 8-bit will throw a warning.
        """
        with self.assertWarns(UserWarning), tempfile.TemporaryDirectory() as tmpdirname:
            self.model_8bit.save_pretrained(tmpdirname)

    def test_raise_if_config_and_load_in_8bit(self):
        r"""
        Test that loading the model with the config and `load_in_8bit` raises an error
        """
        bnb_config = BitsAndBytesConfig()

        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                load_in_8bit=True,
                device_map="auto",
                llm_int8_enable_fp32_cpu_offload=True,
            )

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after converting it in 8-bit will throw an error.
        Checks also if other models are casted correctly.
        """
        with self.assertRaises(ValueError):
            # Tries with `str`
            self.model_8bit.to("cpu")

        with self.assertRaises(ValueError):
            # Tries with a `dtype``
            self.model_8bit.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_8bit.to(torch.device("cuda:0"))

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_8bit.float()

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_8bit.half()

        # Test if we did not break anything
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        self.model_fp16 = self.model_fp16.to(torch.float32)
        _ = self.model_fp16.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Check this does not throw an error
        _ = self.model_fp16.to("cpu")

        # Check this does not throw an error
        _ = self.model_fp16.half()

        # Check this does not throw an error
        _ = self.model_fp16.float()

    def test_fp32_int8_conversion(self):
        r"""
        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", load_in_8bit=True, device_map="auto")
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)

    def test_int8_serialization(self):
        r"""
        Test whether it is possible to serialize a model in 8-bit.
        """
        from bitsandbytes.nn import Int8Params

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_8bit.save_pretrained(tmpdirname)

            # check that the file `quantization_config` is present
            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, "quantization_config"))

            model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, load_in_8bit=True, device_map="auto")

            self.assertTrue(model_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)
            self.assertTrue(hasattr(model_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight, "SCB"))

            # generate
            encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
            output_sequences = model_from_saved.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

            self.assertEqual(
                self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT
            )

    def test_int8_serialization_sharded(self):
        r"""
        Test whether it is possible to serialize a model in 8-bit - sharded version.
        """
        from bitsandbytes.nn import Int8Params

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_8bit.save_pretrained(tmpdirname, max_shard_size="200MB")

            # check that the file `quantization_config` is present
            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, "quantization_config"))

            model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname)

            self.assertTrue(model_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)
            self.assertTrue(hasattr(model_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight, "SCB"))

            # generate
            encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
            output_sequences = model_from_saved.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

            self.assertEqual(
                self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT
            )

    def test_int8_from_pretrained(self):
        r"""
        Test whether loading a 8bit model from the Hub works as expected
        """
        from bitsandbytes.nn import Int8Params

        model_id = "ybelkada/bloom-1b7-8bit"

        model = AutoModelForCausalLM.from_pretrained(model_id)

        self.assertTrue(model.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)
        self.assertTrue(hasattr(model.transformer.h[0].mlp.dense_4h_to_h.weight, "SCB"))

        # generate
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class MixedInt8T5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "t5-small"
        cls.dense_act_model_name = "google/flan-t5-small"  # flan-t5 uses dense-act instead of dense-relu-dense
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "Translate in German: Hello, my dog is cute"

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        gc.collect()
        torch.cuda.empty_cache()

    def test_inference_without_keep_in_fp32(self):
        r"""
        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test
        both cases.
        """
        from transformers import T5ForConditionalGeneration

        T5ForConditionalGeneration._keep_in_fp32_modules = None

        # test with `t5-small`
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(0)
        _ = model.generate(**encoded_input)

        # test with `flan-t5-small`
        model = T5ForConditionalGeneration.from_pretrained(
            self.dense_act_model_name, load_in_8bit=True, device_map="auto"
        )
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(0)
        _ = model.generate(**encoded_input)

    def test_inference_with_keep_in_fp32(self):
        r"""
        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test
        both cases.
        """
        import bitsandbytes as bnb

        from transformers import T5ForConditionalGeneration

        # test with `t5-small`
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")

        # there was a bug with decoders - this test checks that it is fixed
        self.assertTrue(isinstance(model.decoder.block[0].layer[0].SelfAttention.q, bnb.nn.Linear8bitLt))

        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(0)
        _ = model.generate(**encoded_input)

        # test with `flan-t5-small`
        model = T5ForConditionalGeneration.from_pretrained(
            self.dense_act_model_name, load_in_8bit=True, device_map="auto"
        )
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(0)
        _ = model.generate(**encoded_input)

    def test_inference_with_keep_in_fp32_serialized(self):
        r"""
        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly on
        a serialized model.
        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test
        both cases.
        """
        import bitsandbytes as bnb

        from transformers import T5ForConditionalGeneration

        # test with `t5-small`
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            model = T5ForConditionalGeneration.from_pretrained(tmp_dir)

            # there was a bug with decoders - this test checks that it is fixed
            self.assertTrue(isinstance(model.decoder.block[0].layer[0].SelfAttention.q, bnb.nn.Linear8bitLt))

            encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(0)
            _ = model.generate(**encoded_input)

            # test with `flan-t5-small`
            model = T5ForConditionalGeneration.from_pretrained(
                self.dense_act_model_name, load_in_8bit=True, device_map="auto"
            )
            encoded_input = self.tokenizer(self.input_text, return_tensors="pt").to(0)
            _ = model.generate(**encoded_input)


class MixedInt8ModelClassesTest(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()
        # model_name
        self.model_name = "bigscience/bloom-560m"
        self.seq_to_seq_name = "t5-small"

        # Different types of model

        self.base_model = AutoModel.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
        # Sequence classification model
        self.sequence_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, load_in_8bit=True, device_map="auto"
        )
        # CausalLM model
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")
        # Seq2seq model
        self.seq_to_seq_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.seq_to_seq_name, load_in_8bit=True, device_map="auto"
        )

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.base_model
        del self.sequence_model
        del self.model_8bit
        del self.seq_to_seq_model

        gc.collect()
        torch.cuda.empty_cache()

    def test_correct_head_class(self):
        r"""
        A simple test to check if the last modules for some classes (AutoModelForCausalLM or SequenceClassification)
        are kept in their native class.
        """
        from bitsandbytes.nn import Int8Params

        # last param of a base model should be a linear8bit module
        self.assertTrue(self.base_model.h[-1].mlp.dense_4h_to_h.weight.__class__ == Int8Params)

        # Other heads should be nn.Parameter
        self.assertTrue(self.model_8bit.lm_head.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.sequence_model.score.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.seq_to_seq_model.lm_head.weight.__class__ == torch.nn.Parameter)


class MixedInt8TestPipeline(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.pipe

        gc.collect()
        torch.cuda.empty_cache()

    def test_pipeline(self):
        r"""
        The aim of this test is to verify that the mixed int8 is compatible with `pipeline` from transformers. Since
        we used pipline for inference speed benchmarking we want to make sure that this feature does not break anything
        on pipline.
        """
        # self._clear_cuda_cache()
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"device_map": "auto", "load_in_8bit": True},
            max_new_tokens=self.MAX_NEW_TOKENS,
        )

        # Real second forward pass
        pipeline_output = self.pipe(self.input_text)
        self.assertEqual(pipeline_output[0]["generated_text"], self.EXPECTED_OUTPUT)


@require_torch_multi_gpu
class MixedInt8TestMultiGpu(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

    def test_multi_gpu_loading(self):
        r"""
        This tests that the model has been loaded and can be used correctly on a multi-GPU setup.
        Let's just try to load a model on 2 GPUs and see if it works. The model we test has ~2GB of total, 3GB should suffice
        """

        model_parallel = AutoModelForCausalLM.from_pretrained(
            self.model_name, load_in_8bit=True, device_map="balanced"
        )

        # Check correct device map
        self.assertEqual(set(model_parallel.hf_device_map.values()), {0, 1})

        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Second real batch
        output_parallel = model_parallel.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


@require_torch_multi_gpu
class MixedInt8TestCpuGpu(BaseMixedInt8Test):
    def setUp(self):
        super().setUp()

    def check_inference_correctness(self, model):
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_parallel = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        output_text = self.tokenizer.decode(output_parallel[0], skip_special_tokens=True)
        self.assertEqual(output_text, self.EXPECTED_OUTPUT)

    def test_cpu_gpu_loading_random_device_map(self):
        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a random `device_map`.
        """
        device_map = {
            "transformer.word_embeddings": 0,
            "transformer.word_embeddings_layernorm": 0,
            "lm_head": 0,
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": 0,
            "transformer.h.3": 0,
            "transformer.h.4": 0,
            "transformer.h.5": 0,
            "transformer.h.6": 0,
            "transformer.h.7": 0,
            "transformer.h.8": 0,
            "transformer.h.9": 1,
            "transformer.h.10": 0,
            "transformer.h.11": 1,
            "transformer.h.12": 0,
            "transformer.h.13": 0,
            "transformer.h.14": 1,
            "transformer.h.15": 0,
            "transformer.h.16": 0,
            "transformer.h.17": 1,
            "transformer.h.18": 1,
            "transformer.h.19": 0,
            "transformer.h.20": 1,
            "transformer.h.21": 1,
            "transformer.h.22": 0,
            "transformer.h.23": 0,
            "transformer.ln_f": 1,
        }

        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

        model_8bit = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            quantization_config=bnb_config,
        )

        # Check that the model has been correctly set on device 0, 1, and `cpu`.
        self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, "cpu"})

        self.check_inference_correctness(model_8bit)

    def test_cpu_gpu_loading_custom_device_map(self):
        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.
        This time the device map is more organized than the test above and uses the abstraction
        `transformer.h` to encapsulate all the decoder layers.
        """
        device_map = {
            "transformer.word_embeddings": "cpu",
            "transformer.word_embeddings_layernorm": "cpu",
            "lm_head": "cpu",
            "transformer.h": 0,
            "transformer.ln_f": 1,
        }
        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

        # Load model
        model_8bit = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            quantization_config=bnb_config,
        )

        # Check that the model has been correctly set on device 0, 1, and `cpu`.
        self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, "cpu"})

        self.check_inference_correctness(model_8bit)

    def test_cpu_gpu_disk_loading_custom_device_map(self):
        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.
        This time we also add `disk` on the device_map.
        """
        device_map = {
            "transformer.word_embeddings": 0,
            "transformer.word_embeddings_layernorm": "cpu",
            "lm_head": 0,
            "transformer.h": 1,
            "transformer.ln_f": "disk",
        }
        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Load model
            model_8bit = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                quantization_config=bnb_config,
                offload_folder=tmpdirname,
            )

            # Check that the model has been correctly set on device 0, 1, and `cpu`.
            self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, "cpu", "disk"})

            self.check_inference_correctness(model_8bit)

    def test_cpu_gpu_disk_loading_custom_device_map_kwargs(self):
        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.
        This time we also add `disk` on the device_map - using the kwargs directly instead of the quantization config
        """
        device_map = {
            "transformer.word_embeddings": 0,
            "transformer.word_embeddings_layernorm": "cpu",
            "lm_head": 0,
            "transformer.h": 1,
            "transformer.ln_f": "disk",
        }
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Load model
            model_8bit = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                llm_int8_enable_fp32_cpu_offload=True,
                offload_folder=tmpdirname,
            )

            # Check that the model has been correctly set on device 0, 1, and `cpu`.
            self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, "cpu", "disk"})

            self.check_inference_correctness(model_8bit)


class MixedInt8TestTraining(BaseMixedInt8Test):
    def setUp(self):
        self.model_name = "facebook/opt-350m"
        super().setUp()

    def test_training(self):
        if version.parse(importlib_metadata.version("bitsandbytes")) < version.parse("0.37.0"):
            return

        # Step 1: freeze all parameters
        model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map="auto")

        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        # Step 2: add adapters
        for _, module in model.named_modules():
            if "OPTAttention" in repr(type(module)):
                module.q_proj = LoRALayer(module.q_proj, rank=16)
                module.k_proj = LoRALayer(module.k_proj, rank=16)
                module.v_proj = LoRALayer(module.v_proj, rank=16)

        # Step 3: dummy batch
        batch = self.tokenizer("Test batch ", return_tensors="pt").to(0)

        # Step 4: Check if the gradient is not None
        with torch.cuda.amp.autocast():
            out = model.forward(**batch)
            out.logits.norm().backward()

        for module in model.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)
                self.assertTrue(module.adapter[1].weight.grad.norm().item() > 0)
            elif isinstance(module, nn.Embedding):
                self.assertTrue(module.weight.grad is None)
