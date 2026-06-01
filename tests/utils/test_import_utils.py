import sys

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils import import_utils
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


def test_is_package_available_falls_back_to_package_name_metadata(monkeypatch):
    monkeypatch.setattr(import_utils.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", {})
    monkeypatch.setattr(
        import_utils.importlib.metadata,
        "version",
        lambda name: "0.18.0"
        if name == "gguf"
        else (_ for _ in ()).throw(import_utils.importlib.metadata.PackageNotFoundError()),
    )

    is_available, package_version = import_utils._is_package_available("gguf", return_version=True)

    assert is_available is True
    assert package_version == "0.18.0"
