"""Tests for generation utils with meta tensors."""

from copy import deepcopy

import pytest
import torch

from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.testing_utils import require_torch


@pytest.fixture(scope="module")
def cpu_model():
    """Load a small model on CPU for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=None)
    return model.cpu()


def _assert_tensor_equal(tensor1, tensor2):
    """Helper to compare tensor values handling scalar and 1D cases."""
    val1 = tensor1.item() if tensor1.ndim == 0 else tensor1[0].item()
    val2 = tensor2.item() if tensor2.ndim == 0 else tensor2[0].item()
    assert val1 == val2


@require_torch
def test_prepare_special_tokens_cpu(cpu_model):
    generation_config = deepcopy(cpu_model.generation_config)

    cpu_model._prepare_special_tokens(
        generation_config=generation_config, kwargs_has_attention_mask=True, device=torch.device("cpu")
    )

    assert generation_config._eos_token_tensor is not None
    assert generation_config._eos_token_tensor.device.type == "cpu"

    # Check tensor value matches original config
    if cpu_model.config.eos_token_id is not None:
        expected_eos_id = cpu_model.config.eos_token_id
        actual_eos_id = generation_config._eos_token_tensor

        if actual_eos_id.ndim == 0:
            assert actual_eos_id.item() == expected_eos_id
        else:
            assert actual_eos_id[0].item() == expected_eos_id

    # Verify other special tokens are properly handled
    if generation_config.bos_token_id is not None:
        assert generation_config._bos_token_tensor is not None
        assert generation_config._bos_token_tensor.device.type == "cpu"

    if generation_config.pad_token_id is not None:
        assert generation_config._pad_token_tensor is not None
        assert generation_config._pad_token_tensor.device.type == "cpu"


@require_torch
def test_prepare_special_tokens_meta(cpu_model):
    from transformers.generation.utils import MetaSafeTensorError

    generation_config = GenerationConfig()

    # Manually create special token tensors on meta device to trigger the error
    meta_device = torch.device("meta")
    generation_config.eos_token_id = torch.tensor(50256, device=meta_device, dtype=torch.long)
    generation_config.bos_token_id = torch.tensor(50256, device=meta_device, dtype=torch.long)
    generation_config.pad_token_id = torch.tensor(50256, device=meta_device, dtype=torch.long)

    # Should raise MetaSafeTensorError with meta tensors
    with pytest.raises(MetaSafeTensorError, match="Cannot extract token ID from scalar meta tensor"):
        cpu_model._prepare_special_tokens(
            generation_config=generation_config, kwargs_has_attention_mask=True, device=torch.device("cpu")
        )


@require_torch
def test_prepare_special_tokens_consistency(cpu_model):
    """Test that CPU tensors work while meta tensors fail consistently."""
    from transformers.generation.utils import MetaSafeTensorError

    # Define consistent token IDs to use for both tests
    eos_token_id = 50256
    bos_token_id = 50256
    pad_token_id = 50256

    # Test 1: CPU tensors - should work normally
    cpu_config = GenerationConfig()
    cpu_config.eos_token_id = eos_token_id
    cpu_config.bos_token_id = bos_token_id
    cpu_config.pad_token_id = pad_token_id

    cpu_model._prepare_special_tokens(
        generation_config=cpu_config, kwargs_has_attention_mask=True, device=torch.device("cpu")
    )

    # Verify CPU tensors are created successfully
    assert cpu_config._eos_token_tensor.device.type == "cpu"
    assert cpu_config._eos_token_tensor.item() == eos_token_id

    # Test 2: Meta tensors should raise MetaSafeTensorError
    meta_config = GenerationConfig()
    meta_device = torch.device("meta")
    meta_config.eos_token_id = torch.tensor(eos_token_id, device=meta_device, dtype=torch.long)
    meta_config.bos_token_id = torch.tensor(bos_token_id, device=meta_device, dtype=torch.long)
    meta_config.pad_token_id = torch.tensor(pad_token_id, device=meta_device, dtype=torch.long)

    # Should raise MetaSafeTensorError for meta tensors
    with pytest.raises(MetaSafeTensorError, match="Cannot extract token ID from scalar meta tensor"):
        cpu_model._prepare_special_tokens(
            generation_config=meta_config, kwargs_has_attention_mask=True, device=torch.device("cpu")
        )


@require_torch
def test_no_drift_after_prepare(cpu_model):
    generation_config = GenerationConfig()
    generation_config.eos_token_id = 50256
    generation_config.bos_token_id = 50256
    generation_config.pad_token_id = 50256
    generation_config.decoder_start_token_id = 50256

    # Snapshot original values before calling _prepare_special_tokens
    original_eos = deepcopy(generation_config.eos_token_id)
    original_bos = deepcopy(generation_config.bos_token_id)
    original_pad = deepcopy(generation_config.pad_token_id)
    original_decoder_start = deepcopy(generation_config.decoder_start_token_id)

    # Also snapshot other important config attributes
    original_max_length = deepcopy(getattr(generation_config, "max_length", None))
    original_do_sample = deepcopy(getattr(generation_config, "do_sample", None))

    cpu_model._prepare_special_tokens(
        generation_config=generation_config, kwargs_has_attention_mask=True, device=torch.device("cpu")
    )

    # Assert original config values are unchanged (no drift)
    assert generation_config.eos_token_id == original_eos, "eos_token_id should not be mutated"
    assert generation_config.bos_token_id == original_bos, "bos_token_id should not be mutated"
    assert generation_config.pad_token_id == original_pad, "pad_token_id should not be mutated"
    assert generation_config.decoder_start_token_id == original_decoder_start, (
        "decoder_start_token_id should not be mutated"
    )

    # Check other config attributes remain unchanged
    assert getattr(generation_config, "max_length", None) == original_max_length, "max_length should not be mutated"
    assert getattr(generation_config, "do_sample", None) == original_do_sample, "do_sample should not be mutated"

    # Verify only tensor versions were added (new attributes)
    assert hasattr(generation_config, "_eos_token_tensor"), "_eos_token_tensor should be added"
    assert hasattr(generation_config, "_bos_token_tensor"), "_bos_token_tensor should be added"
    assert hasattr(generation_config, "_pad_token_tensor"), "_pad_token_tensor should be added"

    # Ensure tensor versions are properly created
    if generation_config._eos_token_tensor is not None:
        assert isinstance(generation_config._eos_token_tensor, torch.Tensor), (
            "_eos_token_tensor should be torch.Tensor"
        )
        assert generation_config._eos_token_tensor.device.type == "cpu", "_eos_token_tensor should be on CPU"
