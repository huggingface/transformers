<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Import Utilities

This page goes through the transformers utilities to enable lazy and fast object import.
While we strive for minimal dependencies, some models have specific dependencies requirements that cannot be
worked around. We don't want for all users of `transformers` to have to install those dependencies to use other models,
we therefore mark those as soft dependencies rather than hard dependencies.

The transformers toolkit is not made to error-out on import of a model that has a specific dependency; instead, an
object for which you are lacking a dependency will error-out when calling any method on it. As an example, if 
`torchvision` isn't installed, the fast image processors will not be available. 

This object is still importable:

```python
>>> from transformers import DetrImageProcessorFast
>>> print(DetrImageProcessorFast)
<class 'DetrImageProcessorFast'>
```

However, no method can be called on that object:

```python
>>> DetrImageProcessorFast.from_pretrained()
ImportError: 
DetrImageProcessorFast requires the Torchvision library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
```

Let's see how to specify specific object dependencies.

## Specifying Object Dependencies

### Filename-based

All objects under a given filename have an automatic dependency to the tool linked to the filename

**TensorFlow**: All files starting with `modeling_tf_` have an automatic TensorFlow dependency.

**Flax**: All files starting with `modeling_flax_` have an automatic Flax dependency

**PyTorch**: All files starting with `modeling_` and not valid with the above (TensorFlow and Flax) have an automatic 
PyTorch dependency

**Tokenizers**: All files starting with `tokenization_` and ending with `_fast` have an automatic `tokenizers` dependency

**Vision**: All files starting with `image_processing_` have an automatic dependency to the `vision` dependency group; 
at the time of writing, this only contains the `pillow` dependency.

**Vision + Torch + Torchvision**: All files starting with `image_processing_` and ending with `_fast` have an automatic
dependency to `vision`, `torch`, and `torchvision`.

All of these automatic dependencies are added on top of the explicit dependencies that are detailed below.

### Explicit Object Dependencies

We add a method called `requires` that is used to explicitly specify the dependencies of a given object. As an
example, the `Trainer` class has two hard dependencies: `torch` and `accelerate`. Here is how we specify these 
required dependencies:

```python
from .utils.import_utils import requires

@requires(backends=("torch", "accelerate"))
class Trainer:
    ...
```

Backends that can be added here are all the backends that are available in the `import_utils.py` module.

Additionally, specific versions can be specified in each backend. For example, this is how you would specify
a requirement on torch>=2.6 on the `Trainer` class:

```python
from .utils.import_utils import requires

@requires(backends=("torch>=2.6", "accelerate"))
class Trainer:
    ...
```

You can specify the following operators: `==`, `>`, `>=`, `<`, `<=`, `!=`.

## Methods

[[autodoc]] utils.import_utils.define_import_structure

[[autodoc]] utils.import_utils.requires
