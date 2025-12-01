import builtins
import sys

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import clear_import_cache, is_torchcodec_importable


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


def test_torchcodec_importable_handles_import_failure(monkeypatch):
    """
    Simulates an environment where the 'torchcodec' package is installed
    (find_spec succeeds) but importing it fails (e.g., missing DLL on Windows).
    The function should safely return False without raising.
    """

    # Pretend torchcodec is installed
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "torchcodec" else None,
    )

    # Simulate 'import torchcodec' raising a runtime error
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchcodec":
            raise RuntimeError("simulated DLL load failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Should now return False (fallback) rather than raise
    assert is_torchcodec_importable() is False
