import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

from packaging.version import parse as parse_version
from parameterized import parameterized

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import (
    _is_package_available,
    clear_import_cache,
    is_flash_attn_2_available,
    is_flash_attn_3_available,
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


@contextmanager
def mock_flash_attn_env(
    installed_packages: dict[str, str] | None = None,
    cuda_available: bool = False,
    kernels_available: bool = False,
    kernel_download_fails: bool = False,
):
    """Mock the environment probed by `is_flash_attn_{2,3}_available`. Args:
    - `installed_packages`: maps import names to versions, e.g. `{"flash_attn": "2.6.0"}`. The distribution name is
      assumed to match the import name (with underscores replaced by hyphens), except for `flash_attn_interface`
      which is distributed as `flash-attn-3`.
    - `cuda_available`: whether CUDA is available or not.
    - `kernels_available`: whether the kernels library is available.
    - `kernel_download_fails`: if this flag is set to True, the get_kernel method of the fake kernels module will raise
        a RuntimeError to simulate a kernel download failure.
    """
    installed_packages = {} if installed_packages is None else installed_packages
    distribution_names = {"flash_attn_interface": "flash-attn-3"}

    def fake_is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str]:
        is_available = pkg_name in installed_packages
        version = installed_packages.get(pkg_name, "N/A") if return_version else None
        return is_available, version

    fake_distribution_mapping = {
        pkg: [distribution_names.get(pkg, pkg.replace("_", "-"))] for pkg in installed_packages
    }
    fake_kernels_module = ModuleType("kernels")
    fake_kernels_module.get_kernel = MagicMock(
        side_effect=RuntimeError("kernel unavailable") if kernel_download_fails else None
    )

    is_flash_attn_2_available.cache_clear()
    is_flash_attn_3_available.cache_clear()
    try:
        with (
            patch("transformers.utils.import_utils._is_package_available", side_effect=fake_is_package_available),
            patch("transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING", fake_distribution_mapping),
            patch("transformers.utils.import_utils.is_torch_cuda_available", return_value=cuda_available),
            patch("transformers.utils.import_utils.is_torch_mlu_available", return_value=False),
            patch("transformers.utils.import_utils.is_kernels_available", return_value=kernels_available),
            patch.dict(sys.modules, {"kernels": fake_kernels_module}),
        ):
            yield fake_kernels_module.get_kernel
    finally:
        is_flash_attn_2_available.cache_clear()
        is_flash_attn_3_available.cache_clear()


@parameterized.expand([("2.0.0",), ("2.3.3",), ("2.6.0",)])
def test_flash_attn_2_available_with_package(version: str):
    # If the package version is below 2.3.3, the package is too old, and FA should be unavailable
    expected = parse_version(version) >= parse_version("2.3.3")

    with mock_flash_attn_env(installed_packages={"flash_attn": version}, cuda_available=True) as get_kernel:
        # Check the result is the expected one
        is_available = is_flash_attn_2_available()
        assert is_available == expected, (
            f"Expected is_flash_attn_2_available() to be {expected} but got {is_available}"
        )
        # Check the kernels fallback was not probed (kernels_fallback_ok default value is False)
        get_kernel.assert_not_called()
        # Ensure the kernels fallback is not probed (should not happen when the package is present and cuda available)
        assert is_flash_attn_2_available(kernels_fallback_ok=True) == expected
        get_kernel.assert_not_called()


def test_flash_attn_3_available_with_package():
    with mock_flash_attn_env(installed_packages={"flash_attn_interface": "3.0.0"}, cuda_available=True) as get_kernel:
        assert is_flash_attn_3_available()
        assert is_flash_attn_3_available(kernels_fallback_ok=True)
        get_kernel.assert_not_called()


@parameterized.expand(
    [(2, False, False), (2, True, False), (2, True, True), (3, False, False), (3, True, False), (3, True, True)]
)
def test_flash_attn_cuda_kernels_fallback(fa_version: int, kernels_available: bool, download_fails: bool):
    from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK

    # Test is expected to pass only if the kernels library is available and the kernel download does not fail
    expected = kernels_available and not download_fails

    # Mock an env where the package is not available and kernels availability depends on the parameters
    with mock_flash_attn_env(kernels_available=kernels_available, kernel_download_fails=download_fails) as get_kernel:
        # Ensure the FA is not available without kernels fallback
        if fa_version == 2:
            assert not is_flash_attn_2_available()
        elif fa_version == 3:
            assert not is_flash_attn_3_available()
        else:
            raise ValueError(f"Invalid FA version: {fa_version}")

        # Check expected value
        if fa_version == 2:
            is_available = is_flash_attn_2_available(kernels_fallback_ok=True)
        elif fa_version == 3:
            is_available = is_flash_attn_3_available(kernels_fallback_ok=True)
        else:
            raise ValueError(f"Invalid FA version: {fa_version}")

        if is_available != expected:
            raise RuntimeError(
                f"Expected is_flash_attn_{fa_version}_available() to be {expected} but got {is_available}"
            )

        # Check the number of calls to get_kernel
        if kernels_available:
            key = f"flash_attention_{fa_version}"
            get_kernel.assert_called_once_with(FLASH_ATTN_KERNEL_FALLBACK[key], version=1)
        else:
            get_kernel.assert_not_called()


def test_flash_attn_2_fallback_rescues_non_cuda_platform():
    # Package installed but no CUDA/MLU device (e.g. XPU): the kernels fallback should still kick in
    with mock_flash_attn_env(installed_packages={"flash_attn": "2.6.0"}, cuda_available=False, kernels_available=True):
        assert not is_flash_attn_2_available()
        assert is_flash_attn_2_available(kernels_fallback_ok=True)


def test_require_flash_attn_decorators_accept_kernels_fallback():
    # Smoke test: these decorators call is_flash_attn_2_available(kernels_fallback_ok=True) and must not raise
    from transformers.testing_utils import require_all_flash_attn, require_flash_attn

    class DummyTest:
        pass

    with mock_flash_attn_env(kernels_available=True):
        assert require_flash_attn(DummyTest) is not None
        assert require_all_flash_attn(DummyTest) is not None
