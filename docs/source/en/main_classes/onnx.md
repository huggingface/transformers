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

# Exporting 🤗 Transformers models to ONNX

🤗 Transformers provides a `transformers.onnx` package that enables you to
convert model checkpoints to an ONNX graph by leveraging configuration objects.

See the [guide](../serialization) on exporting 🤗 Transformers models for more
details.

## ONNX Configurations

We provide three abstract classes that you should inherit from, depending on the
type of model architecture you wish to export:

* Encoder-based models inherit from [`~onnx.config.OnnxConfig`]
* Decoder-based models inherit from [`~onnx.config.OnnxConfigWithPast`]
* Encoder-decoder models inherit from [`~onnx.config.OnnxSeq2SeqConfigWithPast`]

### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX Features

Each ONNX configuration is associated with a set of _features_ that enable you
to export models for different types of topologies or tasks.

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager

