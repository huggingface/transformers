import torch
import pytest
from transformers.modeling_utils import get_torch_context_manager_or_global_device

def test_get_torch_context_manager_or_global_device():
    device = get_torch_context_manager_or_global_device()
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == expected_device, f"Expected device {expected_device}, got {device}"

@pytest.mark.skipif(hasattr(torch, "get_default_device"), reason="get_default_device exists in this PyTorch version")
def test_get_default_device_raises_attribute_error():
    with pytest.raises(AttributeError, match="module 'torch' has no attribute 'get_default_device'"):
        torch.get_default_device()