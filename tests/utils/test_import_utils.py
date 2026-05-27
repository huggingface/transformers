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


def test_is_package_available_namespace_shadow_marked_unavailable():
    pkg_name = "definitely_not_a_real_pkg_xyz"
    fake_module = ModuleType(pkg_name)

    with (
        patch("transformers.utils.import_utils.importlib.util.find_spec", return_value=object()),
        patch("transformers.utils.import_utils.importlib.import_module", return_value=fake_module),
    ):
        assert _is_package_available(pkg_name, return_version=True) == (False, "N/A")


def test_is_package_available_versionless_install_marked_available():
    pkg_name = "definitely_not_a_real_pkg_xyz"
    fake_module = ModuleType(pkg_name)
    fake_module.__file__ = "/path/to/site-packages/definitely_not_a_real_pkg_xyz/__init__.py"

    with (
        patch("transformers.utils.import_utils.importlib.util.find_spec", return_value=object()),
        patch("transformers.utils.import_utils.importlib.import_module", return_value=fake_module),
    ):
        assert _is_package_available(pkg_name, return_version=True) == (True, "N/A")


def test_is_package_available_frozen_install_marked_available():
    pkg_name = "definitely_not_a_real_pkg_xyz"
    fake_module = ModuleType(pkg_name)
    fake_module.__version__ = "1.2.3"

    with (
        patch("transformers.utils.import_utils.importlib.util.find_spec", return_value=object()),
        patch("transformers.utils.import_utils.importlib.import_module", return_value=fake_module),
    ):
        assert _is_package_available(pkg_name, return_version=True) == (True, "1.2.3")
