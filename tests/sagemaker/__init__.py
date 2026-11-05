import importlib


def is_sagemaker_available():
    return importlib.util.find_spec("sagemaker") is not None
