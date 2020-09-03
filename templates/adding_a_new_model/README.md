# How to add a new model in 🤗 Transformers

This folder describes the process to add a new model in 🤗 Transformers and provide templates for the required files.

The library is designed to incorporate a variety of models and code bases. As such the process for adding a new model
usually mostly consists in copy-pasting to relevant original code in the various sections of the templates included in
the present repository.

One important point though is that the library has the following goals impacting the way models are incorporated:

- One specific feature of the API is the capability to run the model and tokenizer inline. The tokenization code thus
  often have to be slightly adapted to allow for running in the python interpreter.
- the package is also designed to be as self-consistent and with a small and reliable set of packages dependencies. In
  consequence, additional dependencies are usually not allowed when adding a model but can be allowed for the
  inclusion of a new tokenizer (recent examples of dependencies added for tokenizer specificities include
  `sentencepiece` and `sacremoses`). Please make sure to check the existing dependencies when possible before adding a
  new one.

For a quick overview of the general philosphy of the library and its organization, please check the
[QuickStart section of the documentation](https://huggingface.co/transformers/philosophy.html).

# Typical workflow for including a model

Here an overview of the general workflow: 

- [ ] Add model/configuration/tokenization classes.
- [ ] Add conversion scripts.
- [ ] Add tests and a @slow integration test.
- [ ] Document your model.
- [ ] Finalize.

Let's detail what should be done at each step.

## Adding model/configuration/tokenization classes

Here is the workflow for adding model/configuration/tokenization classes:

- [ ] Copy the python files from the present folder to the main folder and rename them, replacing `xxx` with your model
  name.
- [ ] Edit the files to replace `XXX` (with various casing) with your model name.
- [ ] Copy-paste or create a simple configuration class for your model in the `configuration_...` file.
- [ ] Copy-paste or create the code for your model in the `modeling_...` files (PyTorch and TF 2.0).
- [ ] Copy-paste or create a tokenizer class for your model in the `tokenization_...` file.

## Adding conversion scripts

Here is the workflow for the conversion scripts:

- [ ] Copy the conversion script (`convert_...`) from the present folder to the main folder.
- [ ] Edit this script to convert your original checkpoint weights to the current pytorch ones.

## Adding tests:

Here is the workflow for the adding tests:

- [ ] Copy the python files from the `tests` sub-folder of the present folder to the `tests` subfolder of the main
  folder and rename them, replacing `xxx` with your model name.
- [ ] Edit the tests files to replace `XXX` (with various casing) with your model name.
- [ ] Edit the tests code as needed.

## Documenting your model:

Here is the workflow for documentation:

- [ ] Make sure all your arguments are properly documented in your configuration and tokenizer.
- [ ] Most of the documentation of the models is automatically generated, you just have to make sure that
  `XXX_START_DOCSTRING` contains an introduction to the model you're adding and a link to the original
  article and that `XXX_INPUTS_DOCSTRING` contains all the inputs of your model.
- [ ] Create a new page `xxx.rst` in the folder `docs/source/model_doc` and add this file in `docs/source/index.rst`.

Make sure to check you have no sphinx warnings when building the documentation locally and follow our
[documentaiton guide](https://github.com/huggingface/transformers/tree/master/docs#writing-documentation---specification).

## Final steps

You can then finish the addition step by adding imports for your classes in the common files:

- [ ] Add import for all the relevant classes in `__init__.py`.
- [ ] Add your configuration in `configuration_auto.py`.
- [ ] Add your PyTorch and TF 2.0 model respectively in `modeling_auto.py` and `modeling_tf_auto.py`.
- [ ] Add your tokenizer in `tokenization_auto.py`.
- [ ] Add a link to your conversion script in the main conversion utility (in `commands/convert.py`)
- [ ] Edit the PyTorch to TF 2.0 conversion script to add your model in the `convert_pytorch_checkpoint_to_tf2.py`
  file.
- [ ] Add a mention of your model in the doc: `README.md` and the documentation itself
  in `docs/source/index.rst` and `docs/source/pretrained_models.rst`.
- [ ] Upload the pretrained weights, configurations and vocabulary files.
- [ ] Create model card(s) for your models on huggingface.co. For those last two steps, check the
  [model sharing documentation](https://huggingface.co/transformers/model_sharing.html).
