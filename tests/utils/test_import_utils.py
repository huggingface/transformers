import importlib
import sys

from transformers.utils.import_utils import _LazyModule, clear_import_cache


def test_clear_import_cache():
    # Save initial state
    initial_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}

    # Run the test
    clear_import_cache()

    # Verify modules were removed
    remaining_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
    assert len(remaining_modules) < len(initial_modules)

    # Verify we can reimport a module
    assert "transformers.models.auto.modeling_auto" in sys.modules

    # Restore initial state
    for name, module in initial_modules.items():
        sys.modules[name] = module
        if isinstance(module, _LazyModule):
            # Re-initialize lazy module cache
            importlib.reload(module)
