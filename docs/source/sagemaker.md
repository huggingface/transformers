<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Run training on Amazon SageMaker

Hugging Face and Amazon are introducing new [Hugging Face Deep Learning Containers (DLCs)](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers) to make it easier than ever to train Hugging Face Transformer models in [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

To learn how to access and use the new Hugging Face DLCs with the Amazon SageMaker Python SDK, check out the guides and resources below.

---

## Deep Learning Container (DLC) overview

The Deep Learning Container are in every available where Amazon SageMaker is available. You can see the [AWS region table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) for all AWS global infrastructure. To get an detailed overview of all included packages look [here in the release notes](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html).

| 🤗 Transformers version | 🤗 Datasets version | PyTorch/TensorFlow version | type     | device | Python Version | Example `image_uri`                                                                                                               |
| ----------------------- | ------------------- | -------------------------- | -------- | ------ | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 4.4.2                   | 1.5.0               | PyTorch 1.6.0              | training | GPU    | 3.6            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04`    |
| 4.4.2                   | 1.5.0               | TensorFlow 2.4.1           | training | GPU    | 3.7            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-tensorflow-training:2.4.1-transformers4.4.2-gpu-py37-cu110-ubuntu18.04` |

---

## Getting Started: Train a 🤗 Transformers Model

To train a 🤗 Transformers model by using the `HuggingFace` SageMaker Python SDK you need to:

- [Prepare a training script](#prepare-a-transformers-fine-tuning-script)
- [Create a `HuggingFace` Estimator](#create-an-huggingface-estimator)
- [Run training by calling the `fit` method](#execute-training)
- [Access you model](#access-trained-model)

### Setup & Installation

Before you can train a transformers models with Amazon SageMaker you need to sign up for an AWS account. If you do not have an AWS account yet learn more [here](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).

After you complete these tasks you can get started using either [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html), [SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html), or a local environment. To start training locally you need configure the right [IAM permission](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

Upgrade to the latest `sagemaker` version.

```bash
pip install sagemaker --upgrade
```

**SageMaker environment**

_Note: The execution role is intended to be available only when running a notebook within SageMaker. If you run `get_execution_role` in a notebook not on SageMaker, expect a "region" error._

```python
import sagemaker
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
```

**Local environment**

```python
import sagemaker
import boto3

iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName='role-name-of-your-iam-role-with-right-permissions')['Role']['Arn']
sess = sagemaker.Session()
```

### Prepare a 🤗 Transformers fine-tuning script.

The training script is very similar to a training script you might run outside of SageMaker, but you can access useful properties about the training environment through various environment variables, including the following:

- `SM_MODEL_DIR`: A string that represents the path where the training job writes the model artifacts to. After training, artifacts in this directory are uploaded to S3 for model hosting. `SM_MODEL_DIR` is always set to `/opt/ml/model`.

- `SM_NUM_GPUS`: An integer representing the number of GPUs available to the host.

- `SM_CHANNEL_XXXX:` A string that represents the path to the directory that contains the input data for the specified channel. For example, if you specify two input channels in the HuggingFace estimator’s fit call, named `train` and `test`, the environment variables `SM_CHANNEL_TRAIN` and `SM_CHANNEL_TEST` are set.

You can find a full list of the exposed environment variables [here](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md).

Later we define `hyperparameters` in the [HuggingFace Estimator](#create-an-huggingface-estimator), which are passed in as named arguments and and can be processed with the `ArgumentParser()`.

```python
import transformers
import datasets
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--model_name_or_path", type=str)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
```

_Note that SageMaker doesn’t support argparse actions. For example, if you want to use a boolean hyperparameter, specify `type` as `bool` in your script and provide an explicit `True` or `False` value._

For a complete example of a 🤗 Transformers training script, see [train.py](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/scripts/train.py)

### Create an HuggingFace Estimator

You run 🤗 Transformers training scripts on SageMaker by creating `HuggingFace` Estimators. The Estimator handles end-to-end Amazon SageMaker training. The training of your script is invoked when you call `fit` on a `HuggingFace` Estimator. In the Estimator you define, which fine-tuning script should be used as `entry_point`, which `instance_type` should be used, which `hyperparameters` are passed in, you can find all possible `HuggingFace` Parameter [here](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#huggingface-estimator). and an example of a fine-tuning script [here](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/scripts/train.py).
You can find all useable `instance_types` [here](https://aws.amazon.com/de/sagemaker/pricing/).

The following code sample shows how you train a custom `HuggingFace` script `train.py`, passing in three hyperparameters (`epochs`, `per_device_train_batch_size`, and `model_name_or_path`).

```python
from sagemaker.huggingface import HuggingFace


# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'per_device_train_batch_size': 32,
                 'model_name_or_path': 'distilbert-base-uncased'
                 }

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters = hyperparameters
)
```

To run the `TrainingJob` locally you can define `instance_type='local'` or `instance_type='local-gpu'` for gpu usage. _Note: this does not working within SageMaker Studio_

### Execute Training

You start your `TrainingJob` by calling `fit` on a `HuggingFace` Estimator. In the `fit` method you specify your input training data, like a string S3 URI `s3://my-bucket/my-training-data` or a `FileSystemInput` for [EFS or FSx Lustre](https://sagemaker.readthedocs.io/en/stable/overview.html?highlight=FileSystemInput#use-file-systems-as-training-inputs), see [here](https://sagemaker.readthedocs.io/en/stable/overview.html?highlight=FileSystemInput#use-file-systems-as-training-inputs).

```python
huggingface_estimator.fit(
  {'train': 's3://sagemaker-us-east-1-558105141721/samples/datasets/imdb/train',
   'test': 's3://sagemaker-us-east-1-558105141721/samples/datasets/imdb/test'}
)

```

SageMaker takes care of starting and managing all the required ec2 instances for ands starts the training job by running.

```bash
/opt/conda/bin/python train.py --epochs 1 --model_name_or_path distilbert-base-uncased --per_device_train_batch_size 32
```

### Access trained model

After training is done you can access your model either through the [AWS console](https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin) or downloading it directly from S3.

```python
from sagemaker.s3 import S3Downloader

S3Downloader.download(
    s3_uri=huggingface_estimator.model_data, # s3 uri where the trained model is located
    local_path='.', # local path where *.targ.gz is saved
    sagemaker_session=sess # sagemaker session used for training the model
)
```

---

## Sample Notebooks

You can find here a list of the official notebooks provided by Hugging Face.

| Notebook                                                                                                                                                                                        | Description                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [Getting Started Pytorch](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb)                                                   | End-to-End binary Text-Classification example using `Trainer` and `imdb` dataset                                 |
| [Getting Started Tensorflow](https://github.com/huggingface/notebooks/blob/master/sagemaker/02_getting_started_tensorflow/sagemaker-notebook.ipynb)                                             | End-to-End binary Text-Classification example using `Keras` and `imdb` dataset                                   |
| [Distributed Training Data Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/03_distributed_training_data_parallelism/sagemaker-notebook.ipynb)                       | End-to-End distributed Question-Answering example using `Trainer` and 🤗 Transformers example script for `SQAuD` |
| [Distributed Training Model Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)                     | End-to-End model parallelism example using `SageMakerTrainer` and `run_glue.py` script                           |
| [Spot Instances and continues training](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb)                                              | End-to-End to Text-Classification example using spot instances with continued training.                          |
| [SageMaker Metrics](https://github.com/huggingface/notebooks/blob/master/sagemaker/06_sagemaker_metrics/sagemaker-notebook.ipynb)                                                               | End-to-End to Text-Classification example using SageMaker Metrics to extract and log metrics during training     |
| [Distributed Training Data Parallelism Tensorflow](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb) | End-to-End distributed binary Text-Classification example using `Keras` and `TensorFlow`                    
| [Distributed Seq2Seq Training with Data Parallelism and BART](https://github.com/huggingface/notebooks/blob/master/sagemaker/08_distributed_summarization_bart_t5/sagemaker-notebook.ipynb) | End-to-End distributed summarization example `BART-large` and 🤗 Transformers example script for `summarization`                        |


---

## Advanced Features

In addition to the Deep Learning Container and the SageMaker SDK, we have implemented other additional features.

### Distributed Training: Data-Parallel

You can use [SageMaker Data Parallelism Library](https://aws.amazon.com/blogs/aws/managed-data-parallelism-in-amazon-sagemaker-simplifies-training-on-large-datasets/) out of the box for distributed training. We added the functionality of Data Parallelism directly into the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html). If your `train.py` uses the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) API you only need to define the distribution parameter in the HuggingFace Estimator.

- [Example Notebook PyTorch](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)
- [Example Notebook TensorFlow](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb)

```python
# configuration for running training on smdistributed Data Parallel
distribution = {'smdistributed':{'dataparallel':{ 'enabled': True }}}

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3dn.24xlarge',
        instance_count=2,
        role=role,
        transformers_version='4.4.2',
        pytorch_version='1.6.0',
        py_version='py36',
        hyperparameters = hyperparameters
        distribution = distribution
)

```

### Distributed Training: Model-Parallel

You can use [SageMaker Model Parallelism Library](https://aws.amazon.com/blogs/aws/amazon-sagemaker-simplifies-training-deep-learning-models-with-billions-of-parameters/) out of the box for distributed training. We added the functionality of Model Parallelism directly into the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html). If your `train.py` uses the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) API you only need to define the distribution parameter in the HuggingFace Estimator.  
For detailed information about the adjustments take a look [here](https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_general.html?highlight=modelparallel#required-sagemaker-python-sdk-parameters).


- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)


```python
# configuration for running training on smdistributed Model Parallel
mpi_options = {
    "enabled" : True,
    "processes_per_host" : 8
}

smp_options = {
    "enabled":True,
    "parameters": {
        "microbatches": 4,
        "placement_strategy": "spread",
        "pipeline": "interleaved",
        "optimize": "speed",
        "partitions": 4,
        "ddp": True,
    }
}

distribution={
    "smdistributed": {"modelparallel": smp_options},
    "mpi": mpi_options
}

 # create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3dn.24xlarge',
        instance_count=2,
        role=role,
        transformers_version='4.4.2',
        pytorch_version='1.6.0',
        py_version='py36',
        hyperparameters = hyperparameters,
        distribution = distribution
)
```

### Spot Instances

With the creation of HuggingFace Framework extension for the SageMaker Python SDK we can also leverage the benefit of [fully-managed EC2 spot instances](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) and save up to 90% of our training cost.

_Note: Unless your training job completes quickly, we recommend you use [checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) with managed spot training, therefore you need to define the `checkpoint_s3_uri`._

To use spot instances with the `HuggingFace` Estimator we have to set the `use_spot_instances` parameter to `True` and define your `max_wait` and `max_run` time. You can read more about the [managed spot training lifecycle here](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html).

- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb)

```python
# hyperparameters, which are passed into the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'model_name':'distilbert-base-uncased',
                 'output_dir':'/opt/ml/checkpoints'
                 }
# create the Estimator

huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
	    checkpoint_s3_uri=f's3://{sess.default_bucket()}/checkpoints'
        use_spot_instances=True,
        max_wait=3600, # This should be equal to or greater than max_run in seconds'
        max_run=1000,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters = hyperparameters
)

# Training seconds: 874
# Billable seconds: 262
# Managed Spot Training savings: 70.0%

```

### Git Repository

When you create a `HuggingFace` Estimator, you can specify a [training script that is stored in a GitHub repository](https://sagemaker.readthedocs.io/en/stable/overview.html#use-scripts-stored-in-a-git-repository) as the entry point for the estimator, so that you don’t have to download the scripts locally. If Git support is enabled, the `entry_point` and `source_dir` should be relative paths in the Git repo if provided. 

If you are using `git_config` to run the [🤗 Transformers examples scripts](https://github.com/huggingface/transformers/tree/master/examples) keep in mind that you need to configure the right `'branch'` for you `transformers_version`, e.g. if you use `transformers_version='4.4.2` you have to use `'branch':'v4.4.2'`. 

As an example to use `git_config` with an [example script from the transformers repository](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification).

_Tip: define `output_dir` as `/opt/ml/model` in the hyperparameter for the script to save your model to S3 after training._

- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/02_getting_started_tensorflow/sagemaker-notebook.ipynb)

```python
# configure git settings
git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.4.2'} # v4.4.2 is referring to the `transformers_version you use in the estimator.

 # create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='run_glue.py',
        source_dir='./examples/pytorch/text-classification',
        git_config=git_config,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters=hyperparameters
)

```

### SageMaker Metrics

[SageMaker Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html#define-train-metrics) can automatically parse the logs for metrics and send those metrics to CloudWatch. If you want SageMaker to parse logs you have to specify the metrics that you want SageMaker to send to CloudWatch when you configure the training job. You specify the name of the metrics that you want to send and the regular expressions that SageMaker uses to parse the logs that your algorithm emits to find those metrics.

- [Example Notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/06_sagemaker_metrics/sagemaker-notebook.ipynb)

```python
# define metrics definitions

metric_definitions = [
{"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
{"Name": "eval_accuracy", "Regex": "eval_accuracy.*=\D*(.*?)$"},
{"Name": "eval_loss", "Regex": "eval_loss.*=\D*(.*?)$"},
]

# create the Estimator

huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        metric_definitions=metric_definitions,
        hyperparameters = hyperparameters)

```

## Additional Resources

- [Announcement Blog Post](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face)

- [AWS and Hugging Face collaborate to simplify and accelerate adoption of natural language processing](https://aws.amazon.com/blogs/machine-learning/aws-and-hugging-face-collaborate-to-simplify-and-accelerate-adoption-of-natural-language-processing-models/)

- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
