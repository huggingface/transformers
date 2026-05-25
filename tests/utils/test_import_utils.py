import sys
import unittest
from unittest.mock import patch

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import (
    PACKAGE_DISTRIBUTION_MAPPING,
    clear_import_cache,
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_4_available,
)


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


class TestFlashAttnDistributionMapMissing(unittest.TestCase):
    """
    Regression tests for https://github.com/huggingface/transformers/issues/45520.

    When flash_attn is importable but its distribution name is not in
    PACKAGE_DISTRIBUTION_MAPPING (e.g. installed via a non-standard wheel),
    is_flash_attn_*_available() must return False rather than raising a KeyError.
    """

    def _map_without_flash_attn(self):
        return {
            k: v for k, v in PACKAGE_DISTRIBUTION_MAPPING.items() if k not in ("flash_attn", "flash_attn_interface")
        }

    def test_is_flash_attn_2_available_no_keyerror(self):
        with patch("transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING", self._map_without_flash_attn()):
            try:
                result = is_flash_attn_2_available()
                self.assertFalse(result)
            except KeyError as exc:
                self.fail(f"is_flash_attn_2_available raised KeyError: {exc}")

    def test_is_flash_attn_3_available_no_keyerror(self):
        with patch("transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING", self._map_without_flash_attn()):
            try:
                result = is_flash_attn_3_available()
                self.assertFalse(result)
            except KeyError as exc:
                self.fail(f"is_flash_attn_3_available raised KeyError: {exc}")

    def test_is_flash_attn_4_available_no_keyerror(self):
        with patch("transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING", self._map_without_flash_attn()):
            try:
                result = is_flash_attn_4_available()
                self.assertFalse(result)
            except KeyError as exc:
                self.fail(f"is_flash_attn_4_available raised KeyError: {exc}")
