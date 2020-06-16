# Installation

Transformers is tested on Python 3.6+ and PyTorch 1.1.0

## With pip

PyTorch Transformers can be installed using pip as follows:

``` bash
pip install transformers
```

## From source

To install from source, clone the repository and install with:

``` bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
```

## Caching models

This library provides pretrained models that will be downloaded and cached locally. Unless you specify a location with
`cache_dir=...` when you use the `from_pretrained` method, these models will automatically be downloaded in the 
folder given by the shell environment variable ``TRANSFORMERS_CACHE``. The default value for it will be the PyTorch 
cache home followed by ``/transformers/`` (even if you don't have PyTorch installed). This is (by order of priority):

  * shell environment variable ``ENV_TORCH_HOME``
  * shell environment variable ``ENV_XDG_CACHE_HOME`` + ``/torch/``
  * default: ``~/.cache/torch/``

So if you don't have any specific environment variable set, the cache directory will be at 
``~/.cache/torch/transformers/``.

**Note:** If you have set a shell enviromnent variable for one of the predecessors of this library 
(``PYTORCH_TRANSFORMERS_CACHE`` or ``PYTORCH_PRETRAINED_BERT_CACHE``), those will be used if there is no shell 
enviromnent variable for ``TRANSFORMERS_CACHE``.

## Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in the [tests folder](https://github.com/huggingface/transformers/tree/master/tests) and examples tests in the [examples folder](https://github.com/huggingface/transformers/tree/master/examples).

Refer to the [contributing guide](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md#tests) for details about running tests.

## OpenAI GPT original tokenization workflow

If you want to reproduce the original tokenization process of the `OpenAI GPT` paper, you will need to install `ftfy` and `SpaCy`:

``` bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

If you don't install `ftfy` and `SpaCy`, the `OpenAI GPT` tokenizer will default to tokenize using BERT's `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).

## Note on model downloads (Continuous Integration or large-scale deployments)

If you expect to be downloading large volumes of models (more than 1,000) from our hosted bucket (for instance through your CI setup, or a large-scale production deployment), please cache the model files on your end. It will be way faster, and cheaper. Feel free to contact us privately if you need any help.

## Do you want to run a Transformer model on a mobile device?

You should check out our [swift-coreml-transformers](https://github.com/huggingface/swift-coreml-transformers) repo.

It contains a set of tools to convert PyTorch or TensorFlow 2.0 trained Transformer models (currently contains `GPT-2`, `DistilGPT-2`, `BERT`, and `DistilBERT`) to CoreML models that run on iOS devices.

At some point in the future, you'll be able to seamlessly move from pre-training or fine-tuning models in PyTorch to productizing them in CoreML,
or prototype a model or an app in CoreML then research its hyperparameters or architecture from PyTorch. Super exciting!
