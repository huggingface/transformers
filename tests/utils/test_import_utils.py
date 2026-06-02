import sys
from types import ModuleType
from unittest.mock import patch

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import _is_package_available, clear_import_cache


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


def test_is_package_available_edge_cases():
    pkg_name = "definitely_not_a_real_pkg_xyz"

    namespace_shadow = ModuleType(pkg_name)
    versionless_install = ModuleType(pkg_name)
    versionless_install.__file__ = f"/path/to/site-packages/{pkg_name}/__init__.py"
    with_version = ModuleType(pkg_name)
    with_version.__version__ = "1.2.3"

    cases = [
        (namespace_shadow, (False, "N/A")),
        (versionless_install, (True, "N/A")),
        (with_version, (True, "1.2.3")),
    ]
    for fake_module, expected in cases:
        with (
            patch("transformers.utils.import_utils.importlib.util.find_spec", return_value=object()),
            patch("transformers.utils.import_utils.importlib.import_module", return_value=fake_module),
        ):
            assert _is_package_available(pkg_name, return_version=True) == expected
