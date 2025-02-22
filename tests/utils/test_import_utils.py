import sys

from transformers.utils.import_utils import clear_import_cache


def test_clear_import_cache():
    # Import some transformers modules

    # Get initial module count
    initial_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}

    # Verify we have some modules loaded
    assert len(initial_modules) > 0

    # Clear cache
    clear_import_cache()

    # Check modules were removed
    remaining_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
    assert len(remaining_modules) < len(initial_modules)

    # Verify we can reimport
    assert "transformers" in sys.modules
