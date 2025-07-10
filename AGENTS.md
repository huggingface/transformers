# AGENTS.md Guide for Hugging Face Transformers

This AGENTS.md file provides guidance for code agents working with this codebase.

## Core Project Structure

- `/src/transformers`: This contains the core source code for the library
  - `/models`: Code for individual models. Models inherit from base classes in the root `/src/transformers` directory.
- `/tests`: This contains the core test classes for the library. These are usually inherited rather than directly run.
  - `/models`: Tests for individual models. Model tests inherit from common tests in the root `/tests` directory.
- `/docs`: This contains the documentation for the library, including guides, tutorials, and API references.

## Coding Conventions for Hugging Face Transformers

- PRs should be as brief as possible. Bugfix PRs in particular can often be only one or two lines long, and do not need large comments, docstrings or new functions in this case. Aim to minimize the size of the diff.
- When writing tests, they should be added to an existing file. The only exception is for PRs to add a new model, when a new test directory should be created for that model.
- Code style is enforced in the CI. You can install the style tools with `pip install -e .[quality]`. You can then run `make fixup` to apply style and consistency fixes to your code.

## Copying and inheritance

Many models in the codebase have similar code, but it is not shared by inheritance because we want each model file to be self-contained.
We use two mechanisms to keep this code in sync:

- "Copied from" syntax. Functions or entire classes can have a comment at the top like this: `# Copied from transformers.models.llama.modeling_llama.rotate_half` or `# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5`
  These comments are actively checked by the style tools, and copies will automatically be updated when the base code is updated. If you need to update a copied function, you should
  either update the base function and use `make fixup` to propagate the change to all copies, or simply remove the `# Copied from` comment if that is inappropriate.
- "Modular" files. These files briefly define models by composing them using inheritance from other models. They are not meant to be used directly. Instead, the style tools
  automatically generate a complete modeling file, like `modeling_bert.py`, from the modular file like `modular_bert.py`. If a model has a modular file, the modeling file
  should never be edited directly! Instead, changes should be made in the modular file, and then you should run `make fixup` to update the modeling file automatically.

When adding new models, you should prefer `modular` style.

## Testing

After making changes, you should usually run `make fixup` to ensure any copies and modular files are updated, and then test all affected models. This includes both
the model you made the changes in and any other models that were updated by `make fixup`. Tests can be run with `pytest tests/models/[name]/test_modeling_[name].py`
If your changes affect code in other classes like tokenizers or processors, you should run those tests instead, like `test_processing_[name].py` or `test_tokenization_[name].py`.

In order to run tests, you may need to install dependencies. You can do this with `pip install -e .[testing]`. You will probably also need to `pip install torch accelerate` if your environment does not already have them.