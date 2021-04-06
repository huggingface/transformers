import os
import subprocess
import unittest
from ast import literal_eval

import pytest

from parameterized import parameterized, parameterized_class

from . import is_sagemaker_available


if is_sagemaker_available():
    from sagemaker import TrainingJobAnalytics
    from sagemaker.huggingface import HuggingFace


@pytest.mark.skipif(
    literal_eval(os.getenv("TEST_SAGEMAKER", "False")) is not True,
    reason="Skipping test because should only be run when releasing minor transformers version",
)
@pytest.mark.usefixtures("sm_env")
@parameterized_class(
    [
        {
            "framework": "pytorch",
            "script": "run_glue.py",
            "model_name_or_path": "distilbert-base-cased",
            "instance_type": "ml.p3dn.24xlarge",
            "results": {"train_runtime": 300, "eval_accuracy": 0.7, "eval_loss": 0.6},
        },
        {
            "framework": "pytorch",
            "script": "run_ddp.py",
            "model_name_or_path": "distilbert-base-cased",
            "instance_type": "ml.p3dn.24xlarge",
            "results": {"train_runtime": 300, "eval_accuracy": 0.7, "eval_loss": 0.6},
        },
        {
            "framework": "tensorflow",
            "script": "run_tf_dist.py",
            "model_name_or_path": "distilbert-base-cased",
            "instance_type": "ml.p3dn.24xlarge",
            "results": {"train_runtime": 500, "eval_accuracy": 0.6, "eval_loss": 0.7},
        },
    ]
)
class MultiNodeTest(unittest.TestCase):
    def setUp(self):
        if self.framework == "pytorch":
            subprocess.run(
                f"cp ./examples/text-classification/run_glue.py {self.env.test_path}/run_glue.py".split(),
                encoding="utf-8",
                check=True,
            )
        assert hasattr(self, "env")

    def create_estimator(self, instance_count):
        job_name = f"{self.env.base_job_name}-{instance_count}-{'ddp' if 'ddp' in self.script else 'smd'}"
        # distributed data settings
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}} if self.script != "run_ddp.py" else None

        # creates estimator
        return HuggingFace(
            entry_point=self.script,
            source_dir=self.env.test_path,
            role=self.env.role,
            image_uri=self.env.image_uri,
            base_job_name=job_name,
            instance_count=instance_count,
            instance_type=self.instance_type,
            debugger_hook_config=False,
            hyperparameters={**self.env.distributed_hyperparameters, "model_name_or_path": self.model_name_or_path},
            metric_definitions=self.env.metric_definitions,
            distribution=distribution,
            py_version="py36",
        )

    def save_results_as_csv(self, job_name):
        TrainingJobAnalytics(job_name).export_csv(f"{self.env.test_path}/{job_name}_metrics.csv")

    # @parameterized.expand([(2,), (4,),])
    @parameterized.expand([(2,)])
    def test_script(self, instance_count):
        # create estimator
        estimator = self.create_estimator(instance_count)

        # run training
        estimator.fit()

        # save csv
        self.save_results_as_csv(estimator.latest_training_job.name)
        # result dataframe
        result_metrics_df = TrainingJobAnalytics(estimator.latest_training_job.name).dataframe()

        # extract kpis
        train_runtime = list(result_metrics_df[result_metrics_df.metric_name == "train_runtime"]["value"])
        eval_accuracy = list(result_metrics_df[result_metrics_df.metric_name == "eval_accuracy"]["value"])
        eval_loss = list(result_metrics_df[result_metrics_df.metric_name == "eval_loss"]["value"])

        # assert kpis
        assert all(t <= self.results["train_runtime"] for t in train_runtime)
        assert any(t >= self.results["eval_accuracy"] for t in eval_accuracy)
        assert all(t <= self.results["eval_loss"] for t in eval_loss)
