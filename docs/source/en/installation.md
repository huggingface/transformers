<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Installation

Transformers works with [PyTorch](https://pytorch.org/get-started/locally/). It has been tested on Python 3.10+ and PyTorch 2.4+.

## Virtual environment

[uv](https://docs.astral.sh/uv/) is an extremely fast Rust-based Python package and project manager and requires a [virtual environment](https://docs.astral.sh/uv/pip/environments/) by default to manage different projects and avoids compatibility issues between dependencies.

It can be used as a drop-in replacement for [pip](https://pip.pypa.io/en/stable/), but if you prefer to use pip, remove `uv` from the commands below.

> [!TIP]
> Refer to the uv [installation](https://docs.astral.sh/uv/guides/install-python/) docs to install uv.

Create a virtual environment to install Transformers in.

```bash
uv venv .env
source .env/bin/activate
```

## Python

Install Transformers with the following command.

[uv](https://docs.astral.sh/uv/) is a fast Rust-based Python package and project manager.

```bash
uv pip install transformers
```

For GPU acceleration, install the appropriate CUDA drivers for [PyTorch](https://pytorch.org/get-started/locally).

Run the command below to check if your system detects an NVIDIA GPU.

```bash
nvidia-smi
```

To install a CPU-only version of Transformers, run the following command.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

Test whether the install was successful with the following command. It should return a label and score for the provided text.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Source install

Installing from source installs the *latest* version rather than the *stable* version of the library. It ensures you have the most up-to-date changes in Transformers and it's useful for experimenting with the latest features or fixing a bug that hasn't been officially released in the stable version yet.

The downside is that the latest version may not always be stable. If you encounter any problems, please open a [GitHub Issue](https://github.com/huggingface/transformers/issues) so we can fix it as soon as possible.

Install from source with the following command.

```bash
uv pip install git+https://github.com/huggingface/transformers
```

Check if the install was successful with the command below. It should return a label and score for the provided text.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Editable install

An [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) is useful if you're developing locally with Transformers. It links your local copy of Transformers to the Transformers [repository](https://github.com/huggingface/transformers) instead of copying the files. The files are added to Python's import path.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> You must keep the local Transformers folder to keep using it.

Update your local version of Transformers with the latest changes in the main repository with the following command.

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#) is a language-agnostic package manager. Install Transformers from the [conda-forge](https://anaconda.org/conda-forge/transformers) channel in your newly created virtual environment.

```bash
conda install conda-forge::transformers
```

## Set up

After installation, you can configure the Transformers cache location or set up the library for offline usage.

### Cache directory

When you load a pretrained model with [`~PreTrainedModel.from_pretrained`], the model is downloaded from the Hub and locally cached.

Every time you load a model, it checks whether the cached model is up-to-date. If it's the same, then the local model is loaded. If it's not the same, the newer model is downloaded and cached.

The default directory given by the shell environment variable `HF_HUB_CACHE` is `~/.cache/huggingface/hub`. On Windows, the default directory is `C:\Users\username\.cache\huggingface\hub`.

Cache a model in a different directory by changing the path in the following shell environment variables (listed by priority).

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) (default)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (only if `HF_HOME` is not set)

### Offline mode

To use Transformers in an offline or firewalled environment requires the downloaded and cached files ahead of time. Download a model repository from the Hub with the [`~huggingface_hub.snapshot_download`] method.

> [!TIP]
> Refer to the [Download files from the Hub](https://hf.co/docs/huggingface_hub/guides/download) guide for more options for downloading files from the Hub. You can download files from specific revisions, download from the CLI, and even filter which files to download from a repository.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

Set the environment variable `HF_HUB_OFFLINE=1` to prevent HTTP calls to the Hub when loading a model.

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext ...
```

Another option for only loading cached files is to set `local_files_only=True` in [`~PreTrainedModel.from_pretrained`].

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```

### Loading weights from S3-compatible object storage

[`~PreTrainedModel.from_pretrained`] expects either a Hub repo id or a local directory. It does not fetch weights over `s3://` directly. To keep private checkpoints or fine-tune outputs in an S3-compatible bucket, such as Amazon S3 or [Backblaze B2](https://www.backblaze.com/cloud-storage?utm_source=transformers-docs&utm_medium=referral&utm_campaign=ai_artifacts&utm_content=installation), download the model snapshot to a local directory with the [`s3fs`](https://s3fs.readthedocs.io/) [`fsspec`](https://filesystem-spec.readthedocs.io/) backend, then point `from_pretrained` at that local path.

```bash
pip install s3fs
```

`s3fs` is built on `boto3`, so it reads the standard `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` credentials. For a non-AWS endpoint, pass the bucket's S3 URL through the `endpoint_url` argument (or the `AWS_ENDPOINT_URL` environment variable). The Backblaze B2 endpoint follows `https://s3.<region>.backblazeb2.com`, and a B2 application key maps onto the AWS access key id and secret access key.

```bash
export AWS_ACCESS_KEY_ID="$B2_APPLICATION_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$B2_APPLICATION_KEY"
export B2_ENDPOINT_URL="https://s3.us-west-004.backblazeb2.com"
```

```py
import os
import s3fs
from transformers import AutoModelForCausalLM, AutoTokenizer

fs = s3fs.S3FileSystem(endpoint_url=os.environ["B2_ENDPOINT_URL"])
fs.get("my-bucket/models/llama-2-7b/", "./llama-2-7b/", recursive=True)

model = AutoModelForCausalLM.from_pretrained("./llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("./llama-2-7b")
```

The same credentials work for streaming a training dataset stored in the bucket with the [`datasets`](https://hf.co/docs/datasets) library, which also uses `fsspec`. Pass the endpoint through `storage_options` so the request reaches the right backend.

```py
import os
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="s3://my-bucket/datasets/wikitext/train-*.parquet",
    storage_options={"endpoint_url": os.environ["B2_ENDPOINT_URL"]},
    streaming=True,
)
```

Only the endpoint value changes between providers, so the same script runs against Amazon S3 by dropping `endpoint_url` and against Backblaze B2 by setting it to the bucket's region URL.
