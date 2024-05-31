# Using the `diff_converter` linter

`pip install libcst` is a must!

# `sh examples/diff-conversion/convert_examples.sh` to get the converted outputs

The diff converter is a new `linter` specific to `transformers`. It allows us to unpack inheritance in python to convert a modular `diff` file like `diff_gemma.py` into a `single model single file`. 

Examples of possible usage are available in the `examples/diff-conversion`, or `diff_gemma` for a full model usage.

`python utils/diff_model_converter.py --files_to_parse "/Users/arthurzucker/Work/transformers/examples/diff-conversion/diff_my_new_model2.py"`

## How it works
We use the `libcst` parser to produce an AST representation of the `diff_xxx.py` file. For any imports that are made from `transformers.models.modeling_xxxx` we parse the source code of that module, and build a class dependency mapping, which allows us to unpack the difference dependencies.

The code from the `diff` file and the class dependency mapping are "merged" to produce the single model single file. 
We use ruff to automatically remove the potential duplicate imports.

## Why we use libcst instead of the native AST?
AST is super powerful, but it does not keep the `docstring`, `comment` or code formatting. Thus we decided to go with `libcst`