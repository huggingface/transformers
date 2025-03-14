import importlib
import os
import subprocess
import sys

import pytest

import transformers.utils.logging
from transformers.utils.import_utils import _LazyModule


@pytest.fixture(scope="function", autouse=True)
def isolate_module_cache():
    """Fixture to save and restore the module cache state before and after each test."""
    # Save initial state
    initial_modules = dict(sys.modules)

    # Save logging state
    initial_verbosity = transformers.utils.logging.get_verbosity()
    initial_env_vars = {}
    for env_var in ["TRANSFORMERS_VERBOSITY"]:
        if env_var in os.environ:
            initial_env_vars[env_var] = os.environ[env_var]

    yield

    # Restore initial state after test
    current_modules = set(sys.modules.keys())
    added_modules = current_modules - set(initial_modules.keys())

    # Remove any modules that were added during the test
    for name in added_modules:
        if name in sys.modules:
            del sys.modules[name]

    # Restore original modules
    for name, module in initial_modules.items():
        sys.modules[name] = module
        if isinstance(module, _LazyModule):
            # Re-initialize lazy module cache
            importlib.reload(module)

    # Restore logging state
    transformers.utils.logging.set_verbosity(initial_verbosity)
    transformers.utils.logging._reset_library_root_logger()

    # Restore environment variables
    for env_var in ["TRANSFORMERS_VERBOSITY"]:
        if env_var in initial_env_vars:
            os.environ[env_var] = initial_env_vars[env_var]
        elif env_var in os.environ:
            del os.environ[env_var]


def run_in_subprocess(func_name):
    """Run a test function in a separate subprocess to fully isolate it."""
    test_script = f"""
import os
import sys
import importlib
from transformers.utils.import_utils import clear_import_cache

def {func_name}():
    # Import the function to test
    from transformers.utils.import_utils import clear_import_cache

    # First, ensure we have some transformers modules loaded
    import transformers.models.auto.modeling_auto

    # Save initial state
    initial_modules = {{name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}}
    assert len(initial_modules) > 0, "No transformers modules loaded before test"

    # Run the test
    clear_import_cache()

    # Verify modules were removed
    remaining_modules = {{name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}}
    assert len(remaining_modules) < len(initial_modules), "No modules were removed"

    # Import and verify module exists
    from transformers.models.auto import modeling_auto
    assert "transformers.models.auto.modeling_auto" in sys.modules
    assert modeling_auto.__name__ == "transformers.models.auto.modeling_auto"

    return True

# Run the test function directly
result = {func_name}()
sys.exit(0 if result else 1)
"""
    # Create a temporary script file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(test_script.encode("utf-8"))
        temp_script = f.name

    try:
        # Run the script in a subprocess
        result = subprocess.run([sys.executable, temp_script], capture_output=True, text=True)
        # Check if the test passed
        passed = result.returncode == 0
        if not passed:
            print(f"Subprocess test failed with output:\n{result.stdout}\n{result.stderr}")
        return passed
    finally:
        # Clean up the temporary file
        os.unlink(temp_script)


def test_clear_import_cache():
    """Wrapper that runs the actual test in a subprocess."""
    assert run_in_subprocess("test_clear_import_cache_impl")
