import gc
import unittest
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import backend_empty_cache, require_compressed_tensors, require_torch, torch_device
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import CompressedTensorsConfig


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class StackCompressedModelTest(unittest.TestCase):
    # Define stubs as class attributes
    compressed_uncompressed_model_stubs = [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-compressed",
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-uncompressed",
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-compressed",
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-uncompressed",
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-stacked-compressed",
            "nm-testing/llama2.c-stories42M-gsm8k-stacked-uncompressed",
        ),
    ]
    # Flatten the list for tests that require a single list of stubs.
    model_stubs = [stub for pair in compressed_uncompressed_model_stubs for stub in pair]

    # For the outputs matching test, use the sparse-only pair.
    sparse_compressed_model = "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-compressed"
    sparse_uncompressed_model = "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-uncompressed"

    prompt = "Paris is the capital of which country?"

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_compressed_uncompressed_model_shapes(self):
        """
        Verify that the weights of an uncompressed model and its decompressed compressed counterpart match.
        Note: Weights for sparsely compressed models may differ due to packing.
        """

        def _has_nested_attr(obj, attr_path):
            attrs = attr_path.split(".")
            for attr in attrs:
                if not hasattr(obj, attr):
                    return None
                obj = getattr(obj, attr)
            return obj

        from compressed_tensors.quantization.utils import iter_named_leaf_modules

        for compressed_model, uncompressed_model in self.compressed_uncompressed_model_stubs:
            with self.subTest(compressed_model=compressed_model, uncompressed_model=uncompressed_model):
                uncompressed = AutoModelForCausalLM.from_pretrained(
                    uncompressed_model,
                    device_map="auto",
                    dtype="auto",
                    quantization_config=CompressedTensorsConfig(run_compressed=False),
                )
                compressed_decompressed = AutoModelForCausalLM.from_pretrained(
                    compressed_model,
                    device_map="auto",
                    dtype="auto",
                    quantization_config=CompressedTensorsConfig(run_compressed=False),
                )

                for name, submodule in iter_named_leaf_modules(uncompressed):
                    comp_decomp_obj = _has_nested_attr(compressed_decompressed, name)
                    if comp_decomp_obj is not None and hasattr(submodule, "weight"):
                        if "sparse-only" in uncompressed_model:
                            self.assertTrue(
                                torch.equal(submodule.weight, comp_decomp_obj.weight),
                                f"Weight mismatch for module '{name}' in sparse-only model.",
                            )
                        else:
                            self.assertTrue(
                                torch.allclose(submodule.weight, comp_decomp_obj.weight, atol=0.2),
                                f"Weight mismatch for module '{name}' in quantized-only or stacked model.",
                            )

    def test_outputs_match(self):
        """
        Ensure that the generated outputs match between the uncompressed model
        and its decompressed compressed counterpart.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.sparse_uncompressed_model)
        input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids

        uncompressed = AutoModelForCausalLM.from_pretrained(
            self.sparse_uncompressed_model,
            device_map="auto",
            dtype="auto",
            quantization_config=CompressedTensorsConfig(run_compressed=False),
        )

        output_uncompressed = uncompressed.generate(input_ids.to(uncompressed.device), max_new_tokens=100)

        decompressed = AutoModelForCausalLM.from_pretrained(
            self.sparse_compressed_model,
            device_map="auto",
            dtype="auto",
            quantization_config=CompressedTensorsConfig(run_compressed=False),
        )
        output_decompressed = decompressed.generate(input_ids.to(decompressed.device), max_new_tokens=100)

        self.assertEqual(
            tokenizer.decode(output_uncompressed[0]),
            tokenizer.decode(output_decompressed[0]),
            "Generated outputs do not match between compressed and uncompressed models.",
        )

    def test_no_warnings_for_all_models(self):
        """
        Confirm that loading any model using compressed tensors does not trigger
        warnings about missing or unexpected keys.
        """
        for model_stub in self.model_stubs:
            with self.subTest(model_stub=model_stub):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    AutoModelForCausalLM.from_pretrained(
                        model_stub,
                        device_map="auto",
                        dtype="auto",
                        quantization_config=CompressedTensorsConfig(run_compressed=False),
                    )
                    for warning in caught_warnings:
                        self.assertNotIn(
                            "missing keys",
                            str(warning.message).lower(),
                            f"'missing keys' found in warnings for model {model_stub}",
                        )
                        self.assertNotIn(
                            "unexpected keys",
                            str(warning.message).lower(),
                            f"'unexpected keys' found in warnings for model {model_stub}",
                        )


@require_compressed_tensors
@require_torch
class RunCompressedTest(unittest.TestCase):
    tinyllama_w4a16 = "nm-testing/tinyllama-w4a16-compressed-hf-quantizer"
    tinyllama_w8a8 = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

    prompt = "Paris is the capital of which country?"

    stubs = [tinyllama_w4a16, tinyllama_w8a8]

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_default_run_compressed__True(self):
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization.utils import iter_named_leaf_modules

        for stub in self.stubs:
            model = AutoModelForCausalLM.from_pretrained(
                stub,
            )
            compressed_linear_counts = 0

            for _, submodule in iter_named_leaf_modules(
                model,
            ):
                if isinstance(submodule, CompressedLinear):
                    compressed_linear_counts += 1

            # some linear models are not compressed - ex. lm_head
            assert compressed_linear_counts > 0

    def test_default_run_compressed__False(self):
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization.utils import iter_named_leaf_modules

        from transformers.utils.quantization_config import CompressedTensorsConfig

        quantization_config = CompressedTensorsConfig(run_compressed=False)

        for stub in self.stubs:
            model = AutoModelForCausalLM.from_pretrained(
                stub,
                quantization_config=quantization_config,
            )
            compressed_linear_counts = 0

            for _, submodule in iter_named_leaf_modules(
                model,
            ):
                if isinstance(submodule, CompressedLinear):
                    compressed_linear_counts += 1

            # No modules should be CompressedLinear
            assert compressed_linear_counts == 0

    def test_run_compressed_outputs_match(self):
        """Check that run_compressed=True/False output are the same"""

        from transformers import AutoTokenizer
        from transformers.utils.quantization_config import CompressedTensorsConfig

        quantization_config = CompressedTensorsConfig(run_compressed=False)

        for stub in self.stubs:
            tokenizer = AutoTokenizer.from_pretrained(stub)
            input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids

            model_run_compressed__True = AutoModelForCausalLM.from_pretrained(
                stub,
            )
            output_rc_true = model_run_compressed__True.generate(input_ids, max_new_tokens=100)

            model_run_compressed__False = AutoModelForCausalLM.from_pretrained(
                stub,
                quantization_config=quantization_config,
            )
            output_rc_false = model_run_compressed__False.generate(input_ids, max_new_tokens=100)

            assert tokenizer.decode(output_rc_true[0]) == tokenizer.decode(output_rc_false[0])
