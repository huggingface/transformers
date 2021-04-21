import json
import os
import subprocess
import unittest
from ast import literal_eval

import pytest

from parameterized import parameterized, parameterized_class

from . import is_sagemaker_available


if is_sagemaker_available():
    from sagemaker import Session, TrainingJobAnalytics
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
            "script": "run_glue_model_parallelism.py",
            "model_name_or_path": "roberta-large",
            "instance_type": "ml.p3dn.24xlarge",
            "results": {"train_runtime": 1600, "eval_accuracy": 0.3, "eval_loss": 1.2},
        },
        {
            "framework": "pytorch",
            "script": "run_glue.py",
            "model_name_or_path": "roberta-large",
            "instance_type": "ml.p3dn.24xlarge",
            "results": {"train_runtime": 1600, "eval_accuracy": 0.3, "eval_loss": 1.2},
        },
    ]
)
class MultiNodeTest(unittest.TestCase):
    def setUp(self):
        if self.framework == "pytorch":
            subprocess.run(
                f"cp ./examples/pytorch/text-classification/run_glue.py {self.env.test_path}/run_glue.py".split(),
                encoding="utf-8",
                check=True,
            )
        assert hasattr(self, "env")

    def create_estimator(self, instance_count):

        # configuration for running training on smdistributed Model Parallel
        mpi_options = {
            "enabled": True,
            "processes_per_host": 8,
        }
        smp_options = {
            "enabled": True,
            "parameters": {
                "microbatches": 4,
                "placement_strategy": "spread",
                "pipeline": "interleaved",
                "optimize": "speed",
                "partitions": 4,
                "ddp": True,
            },
        }

        distribution = {"smdistributed": {"modelparallel": smp_options}, "mpi": mpi_options}

        name_extension = "trainer" if self.script == "run_glue.py" else "smtrainer"
        # creates estimator
        return HuggingFace(
            entry_point=self.script,
            source_dir=self.env.test_path,
            role=self.env.role,
            image_uri=self.env.image_uri,
            base_job_name=f"{self.env.base_job_name}-{instance_count}-smp-{name_extension}",
            instance_count=instance_count,
            instance_type=self.instance_type,
            debugger_hook_config=False,
            hyperparameters={
                **self.env.hyperparameters,
                "model_name_or_path": self.model_name_or_path,
                "max_steps": 500,
            },
            metric_definitions=self.env.metric_definitions,
            distribution=distribution,
            py_version="py36",
        )

    def save_results_as_csv(self, job_name):
        TrainingJobAnalytics(job_name).export_csv(f"{self.env.test_path}/{job_name}_metrics.csv")

    # @parameterized.expand([(2,), (4,),])
    @parameterized.expand([(1,)])
    def test_scripz(self, instance_count):
        # create estimator
        estimator = self.create_estimator(instance_count)

        # run training
        estimator.fit()

        # result dataframe
        result_metrics_df = TrainingJobAnalytics(estimator.latest_training_job.name).dataframe()

        # extract kpis
        eval_accuracy = list(result_metrics_df[result_metrics_df.metric_name == "eval_accuracy"]["value"])
        eval_loss = list(result_metrics_df[result_metrics_df.metric_name == "eval_loss"]["value"])
        # get train time from SageMaker job, this includes starting, preprocessing, stopping
        train_runtime = (
            Session().describe_training_job(estimator.latest_training_job.name).get("TrainingTimeInSeconds", 999999)
        )

        # assert kpis
        assert train_runtime <= self.results["train_runtime"]
        assert all(t >= self.results["eval_accuracy"] for t in eval_accuracy)
        assert all(t <= self.results["eval_loss"] for t in eval_loss)

        # dump tests result into json file to share in PR
        with open(f"{estimator.latest_training_job.name}.json", "w") as outfile:
            json.dump({"train_time": train_runtime, "eval_accuracy": eval_accuracy, "eval_loss": eval_loss}, outfile)
