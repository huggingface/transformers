import multiprocessing
import os


# Set the env variable to enable parallel loading
os.environ["HF_HF_ENABLE_PARALLEL_LOADING"] = "true"

# Set multiprocessing to spawn, which is required due to cuda constraints
multiprocessing.set_start_method("spawn", force=True)

# Declare the normal model_utils.py test as a sideffect of importing the module
from .test_modeling_utils import ModelUtilsTest  # noqa
