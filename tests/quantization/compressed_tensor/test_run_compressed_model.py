import gc
import unittest

from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_compressed_tensors, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_compressed_tensors
@require_torch
class CompressedTensorsTest(unittest.TestCase):
    tinyllama_w4a16 = "nm-testing/tinyllama-w4a16-compressed-hf-quantizer"
    tinyllama_w8a8 = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"
    llama3_8b_fp8 = "nm-testing/Meta-Llama-3-8B-Instruct-fp8-hf_compat"

    prompt = "Paris is the capital of which country?"

    stubs = [tinyllama_w4a16, tinyllama_w8a8, llama3_8b_fp8]

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
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
