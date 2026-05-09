import sys
from unittest.mock import patch

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import clear_import_cache, is_mlx_available


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


def test_is_mlx_available_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HF_USE_MLX", "0")
    is_mlx_available.cache_clear()

    with patch("transformers.utils.import_utils._is_package_available") as mock_is_package_available:
        assert not is_mlx_available()

    mock_is_package_available.assert_not_called()
    is_mlx_available.cache_clear()


def test_is_mlx_available_checks_package_by_default(monkeypatch):
    monkeypatch.delenv("HF_USE_MLX", raising=False)
    is_mlx_available.cache_clear()

    with patch(
        "transformers.utils.import_utils._is_package_available", return_value=(True, None)
    ) as mock_is_package_available:
        assert is_mlx_available()

    mock_is_package_available.assert_called_once_with("mlx")
    is_mlx_available.cache_clear()
