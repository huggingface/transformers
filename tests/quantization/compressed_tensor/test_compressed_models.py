import gc
import itertools
import warnings

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import require_compressed_tensors, require_torch
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import CompressedTensorsConfig


if is_torch_available():
    import torch


def compressed_uncompressed_model_stubs():
    return [
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


def model_stubs():
    return itertools.chain.from_iterable(compressed_uncompressed_model_stubs())


@pytest.fixture(autouse=True)
def clean_up():
    yield
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.parametrize("compressed_model, uncompressed_model", compressed_uncompressed_model_stubs())
@require_compressed_tensors
@require_torch
def test_compressed_uncompressed_model_shapes(compressed_model, uncompressed_model):
    """
    Verify that the weights of uncompressed and decompressed compressed models are the same.
    Note: Weights of sparsely compressed models may differ due to packing.
    """

    def _has_nested_attr(obj, attr_path):
        attrs = attr_path.split(".")
        for attr in attrs:
            if not hasattr(obj, attr):
                return None
            obj = getattr(obj, attr)
        return obj

    from compressed_tensors.quantization.utils import iter_named_leaf_modules

    uncompressed = AutoModelForCausalLM.from_pretrained(
        uncompressed_model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    compressed_decompressed = AutoModelForCausalLM.from_pretrained(
        compressed_model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )

    for name, submodule in iter_named_leaf_modules(uncompressed):
        if comp_decomp_obj := _has_nested_attr(compressed_decompressed, name):
            if hasattr(submodule, "weight"):
                if "sparse-only" in uncompressed_model:
                    assert torch.equal(submodule.weight, comp_decomp_obj.weight)
                else:
                    assert torch.allclose(submodule.weight, comp_decomp_obj.weight, atol=0.2)


@pytest.mark.parametrize(
    "compressed_model, uncompressed_model",
    [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-compressed",
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-uncompressed",
        ),
    ],
)
@require_compressed_tensors
@require_torch
def test_outputs_match(compressed_model, uncompressed_model):
    """
    Ensure that the output of an uncompressed model matches that of its decompressed compressed counterpart.
    """
    prompt = "Paris is the capital of which country?"

    tokenizer = AutoTokenizer.from_pretrained(uncompressed_model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    uncompressed = AutoModelForCausalLM.from_pretrained(
        uncompressed_model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    output_rc_true = uncompressed.generate(input_ids, max_new_tokens=100)

    compressed_decompressed = AutoModelForCausalLM.from_pretrained(
        compressed_model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    output_rc_false = compressed_decompressed.generate(input_ids, max_new_tokens=100)

    assert tokenizer.decode(output_rc_true[0]) == tokenizer.decode(output_rc_false[0])


@pytest.mark.parametrize("model_stub", model_stubs())
@require_compressed_tensors
@require_torch
def test_no_warnings_for_all_models(model_stub):
    """
    Confirm that loading models saved using compressed tensors
    do not generate warnings regarding missing or unexpected keys.
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        AutoModelForCausalLM.from_pretrained(
            model_stub,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=CompressedTensorsConfig(run_compressed=False),
        )

        for warning in caught_warnings:
            assert "missing keys" not in str(warning.message).lower()
            assert "unexpected keys" not in str(warning.message).lower()
