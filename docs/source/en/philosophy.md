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

# Philosophy

Transformers is a PyTorch-first library. It provides models that are faithful to their papers, easy to use, and easy to hack.

A longer, in-depth article with examples, visualizations and timelines is available [here](https://huggingface.co/spaces/transformers-community/Transformers-tenets) as our canonical reference.

> [!NOTE]
> Our philosophy evolves through practice. What follows are our current, stable principles.

## Who this library is for

- Researchers and educators exploring or extending model architectures.
- Practitioners fine-tuning, evaluating, or serving models.
- Engineers who want a pretrained model that “just works” with a predictable API.

## What you can expect

- Three core classes are required for each model: [configuration](main_classes/configuration),
    [models](main_classes/model), and a preprocessing class. [Tokenizers](main_classes/tokenizer) handle NLP, [image processors](main_classes/image_processor) handle images, [video processors](main_classes/video_processor) handle videos, [feature extractors](main_classes/feature_extractor) handle audio, and [processors](main_classes/processors) handle multimodal inputs.

- All of these classes can be initialized in a simple and unified way from pretrained instances by using a common
    `from_pretrained()` method which downloads (if needed), caches and
    loads the related class instance and associated data (configurations' hyperparameters, tokenizers' vocabulary, processors' parameters
    and models' weights) from a pretrained checkpoint provided on [Hugging Face Hub](https://huggingface.co/models) or your own saved checkpoint.
- On top of those three base classes, the library provides two APIs: [`pipeline`] for quickly
    using a model for inference on a given task and [`Trainer`] to quickly train or fine-tune a PyTorch model.

## Core tenets

The following tenets solidified over time, and they're detailed in our new philosophy [blog post](https://huggingface.co/spaces/transformers-community/Transformers-tenets). They guide maintainer decisions when reviewing PRs and contributions.

> - **Source of Truth.** Implementations must be faithful to official results and intended behavior.
> - **One Model, One File.** Core inference/training logic is visible top-to-bottom in the model file users read.
> - **Code is the Product.** Optimize for reading and diff-ing. Prefer explicit names over clever indirection.
> - **Standardize, Don’t Abstract.** Keep model-specific behavior in the model. Use shared interfaces only for generic infra.
> - **DRY\*** (Repeat when it helps users). End-user modeling files remain self-contained. Infra is factored out.
> - **Minimal User API.** Few codepaths, predictable kwargs, stable methods.
> - **Backwards Compatibility.** Public surfaces should not break. Old Hub artifacts have to keep working..
> - **Consistent Public Surface.** Naming, outputs, and optional diagnostics are aligned and tested.

## Main classes

- [**Configuration classes**](main_classes/configuration) store the hyperparameters required to build a model. These include the number of layers and hidden size. You don't always need to instantiate these yourself. When using a pretrained model without modification, creating the model automatically instantiates the configuration.
- **Model classes** are PyTorch models ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)), wrapped by at least a [PreTrainedModel](https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/model#transformers.PreTrainedModel).

- **Modular transformers.** Contributors write a small `modular_*.py` shard that declares reuse from existing components. The library auto-expands this into the visible `modeling_*.py` file that users read/debug. Maintainers review the shard; users hack the expanded file. This preserves “One Model, One File” without boilerplate drift. See [the contributing documentation](https://huggingface.co/docs/transformers/en/modular_transformers) for more information.

- **Preprocessing classes** convert the raw data into a format accepted by the model. A [tokenizer](main_classes/tokenizer) stores the vocabulary for each model and provides methods for encoding and decoding strings in a list of token embedding indices. [Image processors](main_classes/image_processor) preprocess vision inputs, [video processors](https://huggingface.co/docs/transformers/en/main_classes/video_processor) preprocess videos inputs, [feature extractors](main_classes/feature_extractor) preprocess audio inputs, and [processors](main_classes/processors) preprocess multimodal inputs.

All these classes can be instantiated from pretrained instances, saved locally, and shared on the Hub with three methods:

- `from_pretrained()` lets you instantiate a model, configuration, and preprocessing class from a pretrained version either
  provided by the library itself (the supported models can be found on the [Model Hub](https://huggingface.co/models)) or
  stored locally (or on a server) by the user.
- `save_pretrained()` lets you save a model, configuration, and preprocessing class locally so that it can be reloaded using
  `from_pretrained()`.
- `push_to_hub()` lets you share a model, configuration, and a preprocessing class to the Hub, so it is easily accessible to everyone.
