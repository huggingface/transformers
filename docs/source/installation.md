# Installation

Transformers is tested on Python 2.7 and 3.5+ (examples are tested only on python 3.5+) and PyTorch 1.1.0

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
pip install [--editable] .
```

## Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in the [tests folder](https://github.com/huggingface/transformers/tree/master/transformers/tests) and examples tests in the [examples folder](https://github.com/huggingface/transformers/tree/master/examples).

Tests can be run using `unittest` or `pytest` (install pytest if needed with `pip install pytest`).

Run all the tests from the root of the cloned repository with the commands:

```bash
python -m unittest discover -s transformers/tests -p "*test.py" -t .
python -m unittest discover -s examples -p "*test.py" -t examples
```

or

``` bash
python -m pytest -sv ./transformers/tests/
python -m pytest -sv ./examples/
```

By default, slow tests are skipped. Set the `RUN_SLOW` environment variable to `yes` to run them.

## OpenAI GPT original tokenization workflow

If you want to reproduce the original tokenization process of the `OpenAI GPT` paper, you will need to install `ftfy` (use version 4.4.3 if you are using Python 2) and `SpaCy`:

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
