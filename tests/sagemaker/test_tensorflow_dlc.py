import os
import pytest
import subprocess
from ast import literal_eval
from sagemaker.huggingface import HuggingFace
from sagemaker import TrainingJobAnalytics

# from sagemaker.huggingface import HuggingFace

# TODO add in setup.py there is a release
# | bert-finetuning-pytorch       | testbertfinetuningusingBERTfromtransformerlib+PT                  | SageMaker createTrainingJob | 1        | Accuracy, time to train |
# |-------------------------------|-------------------------------------------------------------------|-----------------------------|----------|-------------------------|
# | bert-finetuning-pytorch-ddp   | test bert finetuning using BERT from transformer lib+ PT DPP      | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
# | bert-finetuning-pytorch-smddp | test bert finetuning using BERT from transformer lib+ PT SM DDP   | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
# | deberta-finetuning-smmp       | test deberta finetuning using BERT from transformer lib+ PT SM MP | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # current DLCs are only in us-east-1 available
os.environ["AWS_PROFILE"] = "hf-sm"  # local profile FIXME: needs to be removed to work in the pipeline

SAGEMAKER_ROLE = "arn:aws:iam::558105141721:role/sagemaker_execution_role"
ECR_IMAGE = "564829616587.dkr.ecr.us-east-1.amazonaws.com/huggingface-training:tensorflow2.3.1-transformers4.3.1-tokenizers0.10.1-datasets1.2.1-py37-gpu-cu110"
BASE_NAME = "sm-tf-transfromers-test"
TEST_PATH = "./tests/sagemaker/scripts/tensorflow"

HYPERPARAMETER = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "epochs": 1,
    "output_dir": "/opt/ml/model",
}

DISTRIBUTED_HYPERPARAMETER = {}

METRIC_DEFINITIONS = [
    {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
    {"Name": "eval_loss", "Regex": "loss.*=\D*(.*?)]$"},
    {"Name": "eval_accuracy", "Regex": "sparse_categorical_accuracy.*=\D*(.*?)]$"},
]
DISTRIBUTED_METRIC_DEFINITIONS = METRIC_DEFINITIONS + []


@pytest.mark.skipif(
    literal_eval(os.getenv("TEST_SAGEMAKER", "False")) is not True,
    reason="Skipping test because should only be run when releasing minor transformers version",
)
@pytest.mark.parametrize("model_name_or_path", ["distilbert-base-cased"])
@pytest.mark.parametrize("instance_count", [1])
@pytest.mark.parametrize("instance_type", ["ml.g4dn.xlarge"])
def test_single_node_fine_tuning(instance_type, instance_count, model_name_or_path):

    # defines hyperparameters
    hyperparameters = {"model_name_or_path": model_name_or_path, **HYPERPARAMETER}

    # creates estimator
    estimator = HuggingFace(
        entry_point="run_tf.py",
        source_dir=TEST_PATH,
        role=SAGEMAKER_ROLE,
        image_uri=ECR_IMAGE,
        base_job_name=f"{BASE_NAME}-single-node",
        instance_count=instance_count,
        instance_type=instance_type,
        debugger_hook_config=False,
        hyperparameters=hyperparameters,
        metric_definitions=METRIC_DEFINITIONS,
        py_version="py3",
    )
    # run training
    estimator.fit()

    # test csv
    TrainingJobAnalytics(estimator.latest_training_job.name).export_csv(
        f"{TEST_PATH}/{BASE_NAME}_single_node_metrics.csv"
    )

    result_metrics_df = TrainingJobAnalytics(estimator.latest_training_job.name).dataframe()

    train_runtime = list(result_metrics_df[result_metrics_df.metric_name == "train_runtime"]["value"])
    eval_accuracy = list(result_metrics_df[result_metrics_df.metric_name == "eval_accuracy"]["value"])
    eval_loss = list(result_metrics_df[result_metrics_df.metric_name == "eval_loss"]["value"])

    assert all(t <= 300 for t in train_runtime)
    assert all(t >= 0.7 for t in eval_accuracy)
    assert all(t <= 0.5 for t in eval_loss)
