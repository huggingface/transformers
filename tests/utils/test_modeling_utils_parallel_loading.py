import os


# Set the env variable to enable parallel loading
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"


# Declare the normal model_utils.py test as a sideffect of importing the module
from .test_modeling_utils import ModelUtilsTest  # noqa
