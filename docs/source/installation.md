# Installation

🤗 Transformers is tested on Python 3.6+, and PyTorch 1.1.0+ or TensorFlow 2.0+.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're
unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Create a virtual environment with the version of Python you're going 
to use and activate it.

Now, if you want to use 🤗 Transformers, you can install it with pip. If you'd like to play with the examples, you
must install it from source.

## Installation with pip

First you need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) 
and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific 
install command for your platform.

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

To install from source, clone the repository and install with the following commands:

``` bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Again, you can run 

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
```

to check 🤗 Transformers is properly installed.

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

### Note on model downloads (Continuous Integration or large-scale deployments)

If you expect to be downloading large volumes of models (more than 1,000) from our hosted bucket (for instance through
your CI setup, or a large-scale production deployment), please cache the model files on your end. It will be way
faster, and cheaper. Feel free to contact us privately if you need any help.

## Do you want to run a Transformer model on a mobile device?

You should check out our [swift-coreml-transformers](https://github.com/huggingface/swift-coreml-transformers) repo.

It contains a set of tools to convert PyTorch or TensorFlow 2.0 trained Transformer models (currently contains `GPT-2`, 
`DistilGPT-2`, `BERT`, and `DistilBERT`) to CoreML models that run on iOS devices.

At some point in the future, you'll be able to seamlessly move from pretraining or fine-tuning models in PyTorch or
TensorFlow 2.0 to productizing them in CoreML, or prototype a model or an app in CoreML then research its
hyperparameters or architecture from PyTorch or TensorFlow 2.0. Super exciting!
