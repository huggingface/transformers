import subprocess
import sys

def test_from_pretrained_with_oo_flag():
    """Tests that AutoModel.from_pretrained works with the -OO flag."""
    script = 'from transformers import AutoModel; model = AutoModel.from_pretrained("google-bert/bert-base-cased")'

    # Run the script using the -OO flag
    result = subprocess.run(
        [sys.executable, "-OO", "-c", script],
        capture_output=True
    )

    # The test passes if the command returns an exit code of 0 (meaning success)
    assert result.returncode == 0, f"Script failed with -OO flag: {result.stderr.decode()}"