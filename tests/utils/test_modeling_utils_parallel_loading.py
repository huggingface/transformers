import multiprocessing
import os

from .test_modeling_utils import ModelUtilsTest


original_setUp = ModelUtilsTest.setUp


# We're monkey patching the original tests, as we want to run them, but now with the parallel loader enabled
def patched_setUp(self):
    # Call the original setUp first
    original_setUp(self)

    # Set the env variable to enable parallel loading
    os.environ.setdefault("ENABLE_PARALLEL_LOADING", "true")
    # Set multiprocessing to spawn, which is required due to torch contraints
    multiprocessing.set_start_method("spawn", force=True)


try:
    # Monkey patch the setUp method
    ModelUtilsTest.setUp = patched_setUp
finally:
    ModelUtilsTest.setUp = original_setUp
