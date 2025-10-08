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

🤗 Transformers is a PyTorch-first library for people who want models that are faithful to their papers, easy to use, and easy to hack.

A longer, in-depth article with examples, visualisations and timelines is available [here as our canonical reference](https://huggingface.co/spaces/transformers-community/Transformers-tenets).

> Note: our philosophy has evolved through practice. What follows is the current, stable set of principles.

## Who this library is for

- Researchers and educators exploring or extending model architectures.
- Practitioners fine-tuning, evaluating, or serving models.
- Engineers who want a pretrained model that “just works” with a predictable API.

## What you can expect

- Three core classes required to use each model: [configuration](main_classes/configuration),
    [models](main_classes/model), and a preprocessing class ([tokenizer](main_classes/tokenizer) for NLP, [image processor](main_classes/image_processor) for images, [video processors](main_classes/video_processor) for videos, [feature extractor](main_classes/feature_extractor) for audio, and [processor](main_classes/processors) for multimodal inputs).

- All of these classes can be initialized in a simple and unified way from pretrained instances by using a common
    `from_pretrained()` method which downloads (if needed), caches and
    loads the related class instance and associated data (configurations' hyperparameters, tokenizers' vocabulary, processors' parameters
    and models' weights) from a pretrained checkpoint provided on [Hugging Face Hub](https://huggingface.co/models) or your own saved checkpoint.
- On top of those three base classes, the library provides two APIs: [`pipeline`] for quickly
    using a model for inference on a given task and [`Trainer`] to quickly train or fine-tune a PyTorch model.


## Core tenets

These tenets solidified over time, and are more detailed in  [our new philosophy blog post.](https://huggingface.co/spaces/transformers-community/Transformers-tenets) They should guide maintainers decisions when reviewing PRs and contributions to the library.

- **Source of Truth.** Implementations must be faithful to official results and intended behavior.
- **One Model, One File.** Core inference/training logic is visible top-to-bottom in the model file users read.
- **Code is the Product.** Optimize for reading and diff-ing; prefer explicit names over clever indirection.
- **Standardize, Don’t Abstract.** Keep model-specific behavior in the model; use shared interfaces only for generic infra.
- **DRY\*** (Repeat when it helps users). End-user modeling files remain self-contained; infra is factored out.
- **Minimal User API.** Few codepaths, predictable kwargs, stable methods.
- **Backwards Compatibility.** Public surfaces should not break; old Hub artifacts have to keep working.
- **Consistent Public Surface.** Naming, outputs, and optional diagnostics are aligned and tested.

## Main classes

- [**Configuration classes**](main_classes/configuration) store the hyperparameters required to build a model (such as the number of layers and hidden size). You don't always need to instantiate these yourself. In particular, if you are using a pretrained model without any modification, creating the model will automatically take care of instantiating the configuration (which is part of the model).
- **Model classes** are PyTorch models ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)), wrapped by at least a [PreTrainedModel](https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/model#transformers.PreTrainedModel).

- **Modular transformers.** Contributors write a small `modular_*.py` shard that declares reuse from existing components. The library auto-expands this into the visible `modeling_*.py` file that users read/debug. Maintainers review the shard; users hack the expanded file. This preserves “One Model, One File” without boilerplate drift. See [the contributing documentation](https://huggingface.co/docs/transformers/en/modular_transformers) for more information.

- **Preprocessing classes** convert the raw data into a format accepted by the model. A [tokenizer](main_classes/tokenizer) stores the vocabulary for each model and provide methods for encoding and decoding strings in a list of token embedding indices to be fed to a model. [Image processors](main_classes/image_processor) preprocess vision inputs, [video processors](https://huggingface.co/docs/transformers/en/main_classes/video_processor) preprocess videos inputs, [feature extractors](main_classes/feature_extractor) preprocess audio inputs, and a [processor](main_classes/processors) handles multimodal inputs.


All these classes can be instantiated from pretrained instances, saved locally, and shared on the Hub with three methods:

- `from_pretrained()` lets you instantiate a model, configuration, and preprocessing class from a pretrained version either
  provided by the library itself (the supported models can be found on the [Model Hub](https://huggingface.co/models)) or
  stored locally (or on a server) by the user.
- `save_pretrained()` lets you save a model, configuration, and preprocessing class locally so that it can be reloaded using
  `from_pretrained()`.
- `push_to_hub()` lets you share a model, configuration, and a preprocessing class to the Hub, so it is easily accessible to everyone.
