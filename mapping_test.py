import torch
from unittest.mock import MagicMock

# Import the classes from your local version
from transformers import KernelConfig


def test_fix_on_mac():
    print("Testing KernelConfig Fix")
    kernel_mapping = {
        "RMSNorm": {
            "cuda": "kernels-community/layer_norm:LlamaRMSNorm",
            "rocm": "kernels-community/layer_norm:LlamaRMSNorm",
        }
    }

    # 3. Create the config
    kernel_config = KernelConfig(kernel_mapping)

    # 4. Create a MOCK model
    # We pretend this is a model on a CUDA device so we don't need the real Llama model
    mock_model = MagicMock()
    mock_model.training = False

    # Mock the parameter device to return 'cuda'
    mock_param = MagicMock()
    mock_param.device.type = "cuda"
    mock_model.parameters.return_value = iter([mock_param])

    # Mock named_modules to register the layer name "RMSNorm"
    mock_layer = MagicMock()
    mock_layer.kernel_layer_name = "RMSNorm"
    mock_model.named_modules.return_value = [("layers.0", mock_layer)]

    print("Simulating model load...")

    # 5. Trigger the logic you fixed
    try:
        kernel_config.create_compatible_mapping(mock_model)
    except Exception as e:
        print(f"Execution crashed: {e}")
        return

    # 6. Verify the result
    result_mapping = kernel_config.kernel_mapping

    print("\n--- Result ---")
    if "RMSNorm" in result_mapping:
        backends = result_mapping["RMSNorm"].keys()
        print(f"Registered Backends: {list(backends)}")

        if "cuda" in backends and "rocm" not in backends:
            print("PASS: The fix worked! ROCm was ignored, preserving CUDA.")
        elif "rocm" in backends:
            print("FAIL: ROCm is present. It overwrote CUDA (The bug is still there).")
        else:
            print("FAIL: Mapping is empty.")
    else:
        print("FAIL: RMSNorm not found in mapping.")


if __name__ == "__main__":
    test_fix_on_mac()
