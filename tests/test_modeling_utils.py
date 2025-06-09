import torch 
import pytest
from transformers.modeling_utils import get_torch_context_manager_or_global_device


def test_get_torch_context_manager_or_global_device():
    """
    Tests that get_torch_context_manager_or_global_device returns the correct device.

    Verifies the function returns 'cuda' if available, else 'cpu', or the current device
    context if set. Ensures compatibility with PyTorch device management for model loading.
    """
    # Calling the function to get the current device
    device = get_torch_context_manager_or_global_device()
    # Defining the expected device based on CUDA availability
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assert the returned device matches the expected device
    assert device == expected_device, f"Expected device {expected_device}, got {device}"


@pytest.mark.skipif(
    hasattr(torch, "get_default_device"),
    reason="get_default_device exists in this PyTorch version",
)
def test_get_default_device_raises_attribute_error():
    """
    Tests that torch.get_default_device raises AttributeError in PyTorch 2.2.
    Ensures the function is not available in PyTorch 2.2, validating the need for
    the get_torch_context_manager_or_global_device workaround.
    """
    with pytest.raises(AttributeError, match="module 'torch' has no attribute 'get_default_device'"):
        torch.get_default_device()