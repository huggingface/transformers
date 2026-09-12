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

New export formats can be added to Transformers by subclassing [`HfExporter`]. ExecuTorch backends
extend the existing exporter with `register_executorch_backend` and an `ExecutorchBackendRecipe`;
see [Extending the exporters](../exporters_extend#register-an-executorch-backend).

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
    - export

## OnnxExporter

[[autodoc]] exporters.exporter_onnx.OnnxExporter
    - export

## ExecutorchExporter

[[autodoc]] exporters.exporter_executorch.ExecutorchExporter
    - export
    - capture
    - lower

## ExecuTorch backend recipes

[[autodoc]] exporters.exporter_executorch.register_executorch_backend

[[autodoc]] exporters.exporter_executorch.ExecutorchBackendRecipe

[[autodoc]] exporters.exporter_executorch.ExecutorchBackendPreparation

[[autodoc]] exporters.exporter_executorch.ExecutorchCompatibilityPolicy

[[autodoc]] exporters.exporter_executorch.ExecutorchCapture

[[autodoc]] exporters.exporter_executorch.ExecutorchExportPatch

[[autodoc]] exporters.exporter_executorch.ExecutorchAttention

[[autodoc]] exporters.exporter_executorch.scoped_executorch_attention

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
