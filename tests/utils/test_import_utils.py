import sys

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


def test_is_torch_hpu_available_warns_before_patching_torch(monkeypatch):
    import torch

    from transformers.utils import import_utils

    import_utils.is_torch_hpu_available.cache_clear()

    def fake_is_package_available(package_name, return_version=False):
        if package_name == "accelerate" and return_version:
            return True, "1.5.0"
        if package_name in {"habana_frameworks", "habana_frameworks.torch"}:
            return True
        return False

    class FakeHPU:
        @staticmethod
        def is_available():
            return True

    original_gather = torch.gather
    original_tensor_gather = torch.Tensor.gather
    original_take_along_dim = torch.take_along_dim
    original_cholesky = torch.linalg.cholesky
    original_scatter = torch.scatter
    original_tensor_scatter = torch.Tensor.scatter
    original_compile = torch.compile

    monkeypatch.setattr(import_utils, "is_torch_available", lambda: True)
    monkeypatch.setattr(import_utils, "_is_package_available", fake_is_package_available)
    monkeypatch.setattr(torch, "hpu", FakeHPU(), raising=False)
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    warnings = []
    monkeypatch.setattr(import_utils.logger, "warning_once", lambda message, *args, **kwargs: warnings.append(message))

    try:
        assert import_utils.is_torch_hpu_available() is True
    finally:
        torch.gather = original_gather
        torch.Tensor.gather = original_tensor_gather
        torch.take_along_dim = original_take_along_dim
        torch.linalg.cholesky = original_cholesky
        torch.scatter = original_scatter
        torch.Tensor.scatter = original_tensor_scatter
        torch.compile = original_compile
        import_utils.is_torch_hpu_available.cache_clear()

    assert any("transformers will patch selected torch functions globally" in warning for warning in warnings)
