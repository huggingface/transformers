import sys
import pytest

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import clear_import_cache


@run_test_using_subprocess
def test_clear_import_cache():
    """Test the clear_import_cache function."""

    # Save initial state
    initial_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
    assert len(initial_modules) > 0, "No transformers modules loaded before test"

    # Execute clear_import_cache() function
    clear_import_cache()

    # Verify modules were removed
    remaining_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
    assert len(remaining_modules) < len(initial_modules), "No modules were removed"

    # Import and verify module exists
    from transformers.models.auto import modeling_auto

    assert "transformers.models.auto.modeling_auto" in sys.modules
    assert modeling_auto.__name__ == "transformers.models.auto.modeling_auto"

@pytest.mark.parametrize(
    "torch_version,enable,expected",
    [
        ("2.9.0", False, "ieee"),
        ("2.10.0", True, "tf32"),
        ("2.10.0", False, "ieee"),
        ("2.8.1", True, True),
        ("2.8.1", False, False),
        ("2.9.0", True, "tf32"),
    ],
)
def test_set_tf32_mode(torch_version, enable, expected):
    # Use the full module path for patch
    with patch("transformers.utils.import_utils.get_torch_version", return_value=torch_version):
        # Mock torch.backends inside the module
        mock_torch = MagicMock()
        with patch("transformers.utils.import_utils.torch", mock_torch):
            _set_tf32_mode(enable)
            pytorch_ver = version.parse(torch_version)
            if pytorch_ver >= version.parse("2.9.0"):
                assert mock_torch.backends.cuda.matmul.fp32_precision == expected
                assert mock_torch.backends.cudnn.fp32_precision == expected
            else:
                assert mock_torch.backends.cuda.matmul.allow_tf32 == expected
                assert mock_torch.backends.cudnn.allow_tf32 == expected
