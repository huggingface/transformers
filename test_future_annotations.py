from __future__ import annotations
from transformers.utils.auto_docstring import _process_kwargs_parameters
import inspect


def test_with_future_annotations():
    # This should fail without fix
    def dummy_func(**kwargs: "ImagesKwargs"):
        pass

    sig = inspect.signature(dummy_func)
    # This line should trigger the bug
    result = _process_kwargs_parameters(sig, dummy_func, None, {}, 0, [])
    print("Success!")


if __name__ == "__main__":
    test_with_future_annotations()
