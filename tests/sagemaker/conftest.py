# we define a fixture function below and it will be "used" by
# referencing its name from tests

import os

import pytest
from attr import dataclass


os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # defaults region


@dataclass
class SageMakerTestEnvironment:
    role = "arn:aws:iam::558105141721:role/sagemaker_execution_role"
    hyperparameters = {
        "task_name": "mnli",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "do_train": True,
        "do_eval": True,
        "do_predict": True,
        "output_dir": "/opt/ml/model",
        "overwrite_output_dir": True,
        "max_steps": 500,
        "save_steps": 5500,
    }
    distributed_hyperparameters = {**hyperparameters, "max_steps": 1000}

    @property
    def metric_definitions(self) -> str:
        return [
            {"Name": "train_runtime", "Regex": r"train_runtime.*=\D*(.*?)$"},
            {"Name": "eval_accuracy", "Regex": r"eval_accuracy.*=\D*(.*?)$"},
            {"Name": "eval_loss", "Regex": r"eval_loss.*=\D*(.*?)$"},
        ]

    @property
    def base_job_name(self) -> str:
        return "pytorch-transformers-test"

    @property
    def test_path(self) -> str:
        return "./tests/sagemaker/scripts/pytorch"

    @property
    def image_uri(self) -> str:
        return "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04"


@pytest.fixture(scope="class")
def sm_env(request):
    request.cls.env = SageMakerTestEnvironment()
