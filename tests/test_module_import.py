import transformers
import importlib

def test_module_spec():
    assert transformers.__spec__ is not None
    assert importlib.util.find_spec("transformers") is not None
