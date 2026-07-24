import gc
import unittest
import warnings

from transformers import AutoModelForCausalLM
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
    ]
    # Flatten the list for tests that require a single list of stubs.
    model_stubs = [stub for pair in compressed_uncompressed_model_stubs for stub in pair]

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

                for name, submodule in uncompressed.named_modules():
                    if list(submodule.children()):
                        continue
                    comp_decomp_obj = _has_nested_attr(compressed_decompressed, name)
                    if comp_decomp_obj is not None and hasattr(submodule, "weight"):
                        torch.testing.assert_close(
                            submodule.weight.to(torch_device),
                            comp_decomp_obj.weight.to(torch_device),
                            atol=0.2,
                            rtol=1e-5,
                            msg=f"Weight mismatch for module '{name}'.",
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
class DequantizeTest(unittest.TestCase):
    """Non-FP8 schemes have no kernels: the model is always dequantized at load time,
    whether or not `dequantize=True` (or the deprecated `run_compressed=False`) is passed."""

    tinyllama_w4a16 = "nm-testing/tinyllama-w4a16-compressed"
    tinyllama_w8a8 = "nm-testing/tinyllama-w8a8-compressed"

    prompt = "Paris is the capital of which country?"

    stubs = [tinyllama_w4a16, tinyllama_w8a8]

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def test_default_load_is_dequantized(self):
        from compressed_tensors import QuantizationStatus

        for stub in self.stubs:
            model = AutoModelForCausalLM.from_pretrained(stub)

            compressed_count = sum(
                1 for m in model.modules() if getattr(m, "quantization_status", None) == QuantizationStatus.COMPRESSED
            )
            self.assertEqual(compressed_count, 0, "no modules should be left in COMPRESSED state")

            # Weights are back to a dense floating-point layout.
            self.assertFalse(any("weight_packed" in name for name, _ in model.named_parameters()))
            self.assertTrue(model.model.layers[0].self_attn.q_proj.weight.is_floating_point())

    def test_deprecated_run_compressed_maps_to_dequantize(self):
        # `run_compressed=False` is deprecated and must behave as `dequantize=True`.
        quantization_config = CompressedTensorsConfig(run_compressed=False)
        self.assertTrue(quantization_config.dequantize)

        from compressed_tensors import QuantizationStatus

        model = AutoModelForCausalLM.from_pretrained(self.stubs[0], quantization_config=quantization_config)
        compressed_count = sum(
            1 for m in model.modules() if getattr(m, "quantization_status", None) == QuantizationStatus.COMPRESSED
        )
        self.assertEqual(compressed_count, 0)

    def test_dequantize_outputs_match_default(self):
        """An explicit `dequantize=True` must produce the same outputs as the default load
        (both are dequantized)."""
        from transformers import AutoTokenizer

        for stub in self.stubs:
            tokenizer = AutoTokenizer.from_pretrained(stub)
            input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids

            model_default = AutoModelForCausalLM.from_pretrained(stub)
            output_default = model_default.generate(input_ids, max_new_tokens=100)

            model_dequantized = AutoModelForCausalLM.from_pretrained(
                stub, quantization_config=CompressedTensorsConfig(dequantize=True)
            )
            output_dequantized = model_dequantized.generate(input_ids, max_new_tokens=100)

            self.assertEqual(tokenizer.decode(output_default[0]), tokenizer.decode(output_dequantized[0]))
