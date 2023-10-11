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

🤗 Transformers is an opinionated library built for:

- machine learning researchers and educators seeking to use, study or extend large-scale Transformers models.
- hands-on practitioners who want to fine-tune those models or serve them in production, or both.
- engineers who just want to download a pretrained model and use it to solve a given machine learning task.

The library was designed with two strong goals in mind:

1. Be as easy and fast to use as possible:

  - We strongly limited the number of user-facing abstractions to learn, in fact, there are almost no abstractions,
    just three standard classes required to use each model: [configuration](main_classes/configuration),
    [models](main_classes/model), and a preprocessing class ([tokenizer](main_classes/tokenizer) for NLP, [image processor](main_classes/image_processor) for vision, [feature extractor](main_classes/feature_extractor) for audio, and [processor](main_classes/processors) for multimodal inputs).
  - All of these classes can be initialized in a simple and unified way from pretrained instances by using a common
    `from_pretrained()` method which downloads (if needed), caches and
    loads the related class instance and associated data (configurations' hyperparameters, tokenizers' vocabulary,
    and models' weights) from a pretrained checkpoint provided on [Hugging Face Hub](https://huggingface.co/models) or your own saved checkpoint.
  - On top of those three base classes, the library provides two APIs: [`pipeline`] for quickly
    using a model for inference on a given task and [`Trainer`] to quickly train or fine-tune a PyTorch model (all TensorFlow models are compatible with `Keras.fit`).
  - As a consequence, this library is NOT a modular toolbox of building blocks for neural nets. If you want to
    extend or build upon the library, just use regular Python, PyTorch, TensorFlow, Keras modules and inherit from the base
    classes of the library to reuse functionalities like model loading and saving. If you'd like to learn more about our coding philosophy for models, check out our [Repeat Yourself](https://huggingface.co/blog/transformers-design-philosophy) blog post.

2. Provide state-of-the-art models with performances as close as possible to the original models:

  - We provide at least one example for each architecture which reproduces a result provided by the official authors
    of said architecture.
  - The code is usually as close to the original code base as possible which means some PyTorch code may be not as
    *pytorchic* as it could be as a result of being converted TensorFlow code and vice versa.

A few other goals:

- Expose the models' internals as consistently as possible:

  - We give access, using a single API, to the full hidden-states and attention weights.
  - The preprocessing classes and base model APIs are standardized to easily switch between models.

- Incorporate a subjective selection of promising tools for fine-tuning and investigating these models:

  - A simple and consistent way to add new tokens to the vocabulary and embeddings for fine-tuning.
  - Simple ways to mask and prune Transformer heads.

- Easily switch between PyTorch, TensorFlow 2.0 and Flax, allowing training with one framework and inference with another.

## Main concepts

The library is built around three types of classes for each model:

- **Model classes** can be PyTorch models ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)), Keras models ([tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)) or JAX/Flax models ([flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html)) that work with the pretrained weights provided in the library.
- **Configuration classes** store the hyperparameters required to build a model (such as the number of layers and hidden size). You don't always need to instantiate these yourself. In particular, if you are using a pretrained model without any modification, creating the model will automatically take care of instantiating the configuration (which is part of the model).
- **Preprocessing classes** convert the raw data into a format accepted by the model. A [tokenizer](main_classes/tokenizer) stores the vocabulary for each model and provide methods for encoding and decoding strings in a list of token embedding indices to be fed to a model. [Image processors](main_classes/image_processor) preprocess vision inputs, [feature extractors](main_classes/feature_extractor) preprocess audio inputs, and a [processor](main_classes/processors) handles multimodal inputs.

All these classes can be instantiated from pretrained instances, saved locally, and shared on the Hub with three methods:

- `from_pretrained()` lets you instantiate a model, configuration, and preprocessing class from a pretrained version either
  provided by the library itself (the supported models can be found on the [Model Hub](https://huggingface.co/models)) or
  stored locally (or on a server) by the user.
- `save_pretrained()` lets you save a model, configuration, and preprocessing class locally so that it can be reloaded using
  `from_pretrained()`.
- `push_to_hub()` lets you share a model, configuration, and a preprocessing class to the Hub, so it is easily accessible to everyone.

