import sys
from pathlib import Path

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import clear_import_cache


LOCAL_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _prepare_local_transformers_import():
    if str(LOCAL_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_SRC_DIR))

    for module_name in list(sys.modules):
        if module_name == "transformers" or module_name.startswith("transformers."):
            del sys.modules[module_name]


def _imports_tqdm(imported_modules: set[str]) -> bool:
    return any(module_name == "tqdm" or module_name.startswith("tqdm.") for module_name in imported_modules)


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


@run_test_using_subprocess
def test_import_transformers_keeps_heavy_modules_lazy():
    _prepare_local_transformers_import()
    initial_modules = set(sys.modules)

    import transformers  # noqa: F401

    imported_modules = set(sys.modules) - initial_modules
    assert "numpy" not in imported_modules
    assert "huggingface_hub.utils" not in imported_modules
    assert "huggingface_hub.hf_api" not in imported_modules
    assert not _imports_tqdm(imported_modules)


@run_test_using_subprocess
def test_importing_cached_file_keeps_hf_api_lazy():
    _prepare_local_transformers_import()
    initial_modules = set(sys.modules)

    from transformers.utils import cached_file  # noqa: F401

    imported_modules = set(sys.modules) - initial_modules
    assert "huggingface_hub.utils" not in imported_modules
    assert "huggingface_hub.hf_api" not in imported_modules
    assert not _imports_tqdm(imported_modules)


@run_test_using_subprocess
def test_importing_logging_keeps_tqdm_lazy_until_use():
    _prepare_local_transformers_import()
    initial_modules = set(sys.modules)

    from transformers import logging

    logging._get_tqdm_lib.cache_clear()

    imported_modules = set(sys.modules) - initial_modules
    assert not _imports_tqdm(imported_modules)
    assert logging._get_tqdm_lib.cache_info().currsize == 0

    logging.enable_progress_bar()
    list(logging.tqdm(range(0)))

    assert logging._get_tqdm_lib.cache_info().currsize == 1
