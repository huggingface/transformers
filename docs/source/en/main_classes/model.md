<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Models

The base classes [`PreTrainedModel`], [`TFPreTrainedModel`], and
[`FlaxPreTrainedModel`] implement the common methods for loading/saving a model either from a local
file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's AWS
S3 repository).

[`PreTrainedModel`] and [`TFPreTrainedModel`] also implement a few methods which
are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary
- prune the attention heads of the model.

The other methods that are common to each model are defined in [`~modeling_utils.ModuleUtilsMixin`]
(for the PyTorch models) and [`~modeling_tf_utils.TFModuleUtilsMixin`] (for the TensorFlow models) or
for text generation, [`~generation.GenerationMixin`] (for the PyTorch models),
[`~generation.TFGenerationMixin`] (for the TensorFlow models) and
[`~generation.FlaxGenerationMixin`] (for the Flax/JAX models).


## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

Custom models should also include a `_supports_assign_param_buffer`, which determines if superfast init can apply
on the particular model. Signs that your model needs this are if `test_save_and_load_from_pretrained` fails. If so,
set this to `False`.

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## Pushing to the Hub

[[autodoc]] utils.PushToHubMixin

## Sharded checkpoints

[[autodoc]] modeling_utils.load_sharded_checkpoint
