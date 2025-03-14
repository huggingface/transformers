import importlib
import os
import sys

import pytest

from transformers.utils.import_utils import _LazyModule


@pytest.mark.skipif(
    "PYTEST_XDIST_WORKER" in os.environ,
    reason="This test should not run under pytest-xdist workers",
)
def test_clear_import_cache(monkeypatch):
    """Test the clear_import_cache function in a way that doesn't affect other tests."""
    # Create a temporary Python script that runs the test
    import tempfile
    import subprocess
    
    test_script = """
import sys
import os

# Import the function to test
from transformers.utils.import_utils import clear_import_cache

# First, ensure we have some transformers modules loaded
import transformers.models.auto.modeling_auto

# Save initial state
initial_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
assert len(initial_modules) > 0, "No transformers modules loaded before test"

# Run the test
clear_import_cache()

# Verify modules were removed
remaining_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
assert len(remaining_modules) < len(initial_modules), "No modules were removed"

# Import and verify module exists
from transformers.models.auto import modeling_auto
assert "transformers.models.auto.modeling_auto" in sys.modules
assert modeling_auto.__name__ == "transformers.models.auto.modeling_auto"

# Exit with success
sys.exit(0)
"""
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(test_script.encode("utf-8"))
        temp_script = f.name
    
    try:
        # Run the script in a subprocess with a clean environment
        env = os.environ.copy()
        # Clear any transformers-related environment variables
        for key in list(env.keys()):
            if key.startswith("TRANSFORMERS_"):
                del env[key]
        
        result = subprocess.run(
            [sys.executable, temp_script],
            env=env,
            capture_output=True,
            text=True
        )
        
        # Check if the test passed
        if result.returncode != 0:
            print(f"Subprocess test failed with output:\n{result.stdout}\n{result.stderr}")
        
        assert result.returncode == 0, "Subprocess test failed"
    finally:
        # Clean up the temporary file
        os.unlink(temp_script)
