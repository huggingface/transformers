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

# Installation

🤗 Transformers is tested on Python 3.6+, and PyTorch 1.1.0+ or TensorFlow 2.0+.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're
unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Create a virtual environment with the version of Python you're going
to use and activate it.

Now, if you want to use 🤗 Transformers, you can install it with pip. If you'd like to play with the examples, you
must install it from source.

## Installation with pip

First you need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available),
[PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) and/or
[Flax installation page](https://github.com/google/flax#quick-install)
regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```

Alternatively, for CPU-support only, you can install 🤗 Transformers and PyTorch in one line with:

```bash
pip install transformers[torch]
```

or 🤗 Transformers and TensorFlow 2.0 in one line with:

```bash
pip install transformers[tf-cpu]
```

or 🤗 Transformers and Flax in one line with:

```bash
pip install transformers[flax]
```

To check 🤗 Transformers is properly installed, run the following command:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

It should download a pretrained model then print something like

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

(Note that TensorFlow will print additional stuff before that last statement.)

## Installing from source

Here is how to quickly install `transformers` from source:

```bash
pip install git+https://github.com/huggingface/transformers
```

Note that this will install not the latest released version, but the bleeding edge `master` version, which you may want to use in case a bug has been fixed since the last official release and a new release hasn't  been yet rolled out.

While we strive to keep `master` operational at all times, if you notice some issues, they usually get fixed within a few hours or a day and and you're more than welcome to help us detect any problems by opening an [Issue](https://github.com/huggingface/transformers/issues) and this way, things will get fixed even sooner.

Again, you can run:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
```

to check 🤗 Transformers is properly installed.

## Editable install

If you want to constantly use the bleeding edge `master` version of the source code, or if you want to contribute to the library and need to test the changes in the code you're making, you will need an editable install. This is done by cloning the repository and installing with the following commands:

``` bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

This command performs a magical link between the folder you cloned the repository to and your python library paths, and it'll look inside this folder in addition to the normal library-wide paths. So if normally your python packages get installed into:
```
~/anaconda3/envs/main/lib/python3.7/site-packages/
```
now this editable install will reside where you clone the folder to, e.g. `~/transformers/` and python will search it too.

Do note that you have to keep that `transformers` folder around and not delete it to continue using the  `transformers` library.

Now, let's get to the real benefit of this installation approach. Say, you saw some new feature has been just committed into `master`. If you have already performed all the steps above, to update your transformers to include all the latest commits, all you need to do is to `cd` into that cloned repository folder and update the clone to the latest version:

```
cd ~/transformers/
git pull
```

There is nothing else to do. Your python environment will find the bleeding edge version of `transformers` on the next run.


## With conda

Since Transformers version v4.0.0, we now have a conda channel: `huggingface`.

🤗 Transformers can be installed using conda as follows:

```
conda install -c huggingface transformers
```

Follow the installation pages of TensorFlow, PyTorch or Flax to see how to install them with conda.

## Caching models

This library provides pretrained models that will be downloaded and cached locally. Unless you specify a location with
`cache_dir=...` when you use methods like `from_pretrained`, these models will automatically be downloaded in the
folder given by the shell environment variable ``TRANSFORMERS_CACHE``. The default value for it will be the Hugging
Face cache home followed by ``/transformers/``. This is (by order of priority):

  * shell environment variable ``HF_HOME``
  * shell environment variable ``XDG_CACHE_HOME`` + ``/huggingface/``
  * default: ``~/.cache/huggingface/``

So if you don't have any specific environment variable set, the cache directory will be at
``~/.cache/huggingface/transformers/``.

**Note:** If you have set a shell environment variable for one of the predecessors of this library
(``PYTORCH_TRANSFORMERS_CACHE`` or ``PYTORCH_PRETRAINED_BERT_CACHE``), those will be used if there is no shell
environment variable for ``TRANSFORMERS_CACHE``.

### Offline mode

It's possible to run 🤗 Transformers in a firewalled or a no-network environment.

Setting environment variable `TRANSFORMERS_OFFLINE=1` will tell 🤗 Transformers to use local files only and will not try to look things up.

Most likely you may want to couple this with `HF_DATASETS_OFFLINE=1` that performs the same for 🤗 Datasets if you're using the latter.

Here is an example of how this can be used on a filesystem that is shared between a normally networked and a firewalled to the external world instances.

On the instance with the normal network run your program which will download and cache models (and optionally datasets if you use 🤗 Datasets). For example:

```
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

and then with the same filesystem you can now run the same program on a firewalled instance:
```
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```
and it should succeed without any hanging waiting to timeout.

#### Fetching models and tokenizers to use offline

When running a script the first time like mentioned above, the downloaded files will be cached for future reuse. 
However, it is also possible to download files and point to their local path instead.

Downloading files can be done through the Web Interface by clicking on the "Download" button, but it can also be handled
programmatically using the `huggingface_hub` library that is a dependency to `transformers`:

- Using `snapshot_download` to download an entire repository
- Using `hf_hub_download` to download a specific file

See the reference for these methods in the huggingface_hub
[documentation](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub).

## Do you want to run a Transformer model on a mobile device?

You should check out our [swift-coreml-transformers](https://github.com/huggingface/swift-coreml-transformers) repo.

It contains a set of tools to convert PyTorch or TensorFlow 2.0 trained Transformer models (currently contains `GPT-2`,
`DistilGPT-2`, `BERT`, and `DistilBERT`) to CoreML models that run on iOS devices.

At some point in the future, you'll be able to seamlessly move from pretraining or fine-tuning models in PyTorch or
TensorFlow 2.0 to productizing them in CoreML, or prototype a model or an app in CoreML then research its
hyperparameters or architecture from PyTorch or TensorFlow 2.0. Super exciting!
