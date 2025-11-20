import os
import unittest
from unittest.mock import patch

from transformers.testing_utils import require_kernels


@require_kernels
class HubKernelsTests(unittest.TestCase):
    def test_disable_hub_kernels(self):
        """
        Test that _kernels_enabled is False when USE_HUB_KERNELS when USE_HUB_KERNELS=OFF
        """
        with patch.dict(os.environ, {"USE_HUB_KERNELS": "ON"}):
            # Re-import to ensure the environment variable takes effect
            import importlib

            from transformers.integrations import hub_kernels

            importlib.reload(hub_kernels)

            # Verify that kernels are disabled
            self.assertFalse(hub_kernels._kernels_enabled)

    def test_enable_hub_kernels_default(self):
        """
        Test that _kernels_enabled is True when USE_HUB_KERNELS is not provided (default behavior)
        """
        # Remove USE_HUB_KERNELS from the environment if it exists
        env_without_hub_kernels = {k: v for k, v in os.environ.items() if k != "USE_HUB_KERNELS"}
        with patch.dict(os.environ, env_without_hub_kernels, clear=True):
            # Re-import to ensure the environment variable change takes effect
            import importlib

            from transformers.integrations import hub_kernels

            importlib.reload(hub_kernels)

            # Verify that kernels are enabled by default
            self.assertTrue(hub_kernels._kernels_enabled)

    def test_enable_hub_kernels_on(self):
        """
        Test that _kernels_enabled is True when USE_HUB_KERNELS=ON
        """
        with patch.dict(os.environ, {"USE_HUB_KERNELS": "ON"}):
            # Re-import to ensure the environment variable takes effect
            import importlib

            from transformers.integrations import hub_kernels

            importlib.reload(hub_kernels)

            # Verify that kernels are enabled
            self.assertTrue(hub_kernels._kernels_enabled)

    @patch("kernels.use_kernel_forward_from_hub")
    def test_use_kernel_forward_from_hub_not_called_when_disabled(self, mocked_use_kernel_forward):
        """
        Test that kernels.use_kernel_forward_from_hub is not called when USE_HUB_KERNELS is disabled
        """
        # Set environment variable to disable hub kernels
        with patch.dict(os.environ, {"USE_HUB_KERNELS": "OFF"}):
            # Re-import to ensure the environment variable takes effect
            import importlib

            from transformers.integrations import hub_kernels

            importlib.reload(hub_kernels)

            # Call the function with a test layer name
            decorator = hub_kernels.use_kernel_forward_from_hub("DummyLayer")

            # Verify that the kernels function was never called
            mocked_use_kernel_forward.assert_not_called()

            # Verify that we get a no-op decorator
            class FooClass:
                pass

            result = decorator(FooClass)
            self.assertIs(result, FooClass)

    @patch("kernels.use_kernel_forward_from_hub")
    def test_use_kernel_forward_from_hub_called_when_enabled_default(self, mocked_use_kernel_forward):
        """
        Test that kernels.use_kernel_forward_from_hub is called when USE_HUB_KERNELS is not set (default)
        """
        # Remove USE_HUB_KERNELS from the environment if it exists
        env_without_hub_kernels = {k: v for k, v in os.environ.items() if k != "USE_HUB_KERNELS"}
        with patch.dict(os.environ, env_without_hub_kernels, clear=True):
            # Re-import to ensure the environment variable change takes effect
            import importlib

            from transformers.integrations import hub_kernels

            importlib.reload(hub_kernels)

            # Call the function with a test layer name
            hub_kernels.use_kernel_forward_from_hub("FooLayer")

            # Verify that the kernels function was called once with the correct argument
            mocked_use_kernel_forward.assert_called_once_with("FooLayer")

    @patch("kernels.use_kernel_forward_from_hub")
    def test_use_kernel_forward_from_hub_called_when_enabled_on(self, mocked_use_kernel_forward):
        """
        Test that kernels.use_kernel_forward_from_hub is called when USE_HUB_KERNELS=ON
        """
        with patch.dict(os.environ, {"USE_HUB_KERNELS": "ON"}):
            # Re-import to ensure the environment variable change takes effect
            import importlib

            from transformers.integrations import hub_kernels

            importlib.reload(hub_kernels)

            # Call the function with a test layer name
            hub_kernels.use_kernel_forward_from_hub("FooLayer")

            # Verify that the kernels function was called once with the correct argument
            mocked_use_kernel_forward.assert_called_once_with("FooLayer")
