<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Exporters

New export backends can be added to Transformers by subclassing [`HfExporter`] and implementing its two
hooks, `export_artifact` and `save_artifact`.

<Tip>

Learn how to use the built-in exporters in the [Exporters](../exporters) guide.

</Tip>

## AutoHfExporter

[[autodoc]] exporters.auto.AutoHfExporter

## AutoExportConfig

[[autodoc]] exporters.auto.AutoExportConfig

## HfExporter

[[autodoc]] exporters.base.HfExporter

## DynamoExporter

[[autodoc]] exporters.exporter_dynamo.DynamoExporter
    - export_artifact
    - save_artifact

## OnnxExporter

[[autodoc]] exporters.exporter_onnx.OnnxExporter
    - export_artifact
    - save_artifact

## ExecutorchExporter

[[autodoc]] exporters.exporter_executorch.ExecutorchExporter
    - export_artifact
    - save_artifact

## ExporterOutput

[[autodoc]] exporters.base.ExporterOutput

## Running an export

What an export is loaded back as: a decomposed, cache-driven export is driven through `generate`, a single
graph is called. [`AutoExportedModel`] reads the manifest and picks between them.

## AutoExportedModel

[[autodoc]] exporters.auto.AutoExportedModel

## ExportedGenerator

[[autodoc]] exporters.generator.ExportedGenerator
    - from_pretrained
    - from_runners

## ExportedModel

[[autodoc]] exporters.base.ExportedModel
    - from_pretrained
    - __call__

## ModelRunner

One exported graph, bound to the runtime that runs it.

[[autodoc]] exporters.base.ModelRunner

[[autodoc]] exporters.runner_dynamo.DynamoModelRunner

[[autodoc]] exporters.runner_onnx.OnnxModelRunner

[[autodoc]] exporters.runner_executorch.ExecutorchModelRunner

## DynamoConfig

[[autodoc]] exporters.configs.DynamoConfig

## OnnxConfig

[[autodoc]] exporters.configs.OnnxConfig

## ExecutorchConfig

[[autodoc]] exporters.configs.ExecutorchConfig

## Utilities

Lower-level functions that power `export_for_generation`, useful when you need to intervene
between decomposing a model and exporting each component.

[[autodoc]] exporters.utils.get_leaf_tensors

[[autodoc]] exporters.utils.prepare_for_export

[[autodoc]] exporters.utils.decompose_prefill_decode

[[autodoc]] exporters.utils.decompose_multimodal

[[autodoc]] exporters.utils.decompose_for_generation

[[autodoc]] exporters.utils.is_multimodal
