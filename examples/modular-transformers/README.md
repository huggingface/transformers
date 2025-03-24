# Using the `modular_converter` linter

`pip install libcst` is a must!

# `sh examples/modular-transformers/convert_examples.sh` to get the converted outputs

The modular converter is a new `linter` specific to `transformers`. It allows us to unpack inheritance in python to convert a modular file like `modular_gemma.py` into a `single model single file`. 

Examples of possible usage are available in the `examples/modular-transformers`, or `modular_gemma` for a full model usage.

`python utils/modular_model_converter.py --files_to_parse "/Users/arthurzucker/Work/transformers/examples/modular-transformers/modular_my_new_model2.py"`

## How it works
We use the `libcst` parser to produce an AST representation of the `modular_xxx.py` file. For any imports that are made from `transformers.models.modeling_xxxx` we parse the source code of that module, and build a class dependency mapping, which allows us to unpack the modularerence dependencies.

The code from the `modular` file and the class dependency mapping are "merged" to produce the single model single file. 
We use ruff to automatically remove the potential duplicate imports.

## Why we use libcst instead of the native AST?
AST is super powerful, but it does not keep the `docstring`, `comment` or code formatting. Thus we decided to go with `libcst`