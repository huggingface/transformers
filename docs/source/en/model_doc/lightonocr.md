<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->

# LightOnOCR

LightOnOCR is a multimodal model designed for optical character recognition (OCR) tasks. It combines a vision encoder for processing document images with a text decoder for generating text sequences.

The model architecture consists of:
- **Vision Encoder**: Processes document images into visual embeddings
- **Text Decoder**: Generates text sequences from the visual embeddings

You can use LightOnOCR for various document understanding tasks including text extraction, document question answering, and structured information extraction.

## LightOnOCRConfig

[[autodoc]] LightOnOCRConfig

## LightOnOCRTextConfig

[[autodoc]] LightOnOCRTextConfig

## LightOnOCRVisionConfig

[[autodoc]] LightOnOCRVisionConfig

## LightOnOCRProcessor

[[autodoc]] LightOnOCRProcessor
    - __call__

## LightOnOCRText

[[autodoc]] LightOnOCRText
    - forward

## LightOnOCRVision

[[autodoc]] LightOnOCRVision
    - forward

## LightOnOCRModel

[[autodoc]] LightOnOCRModel
    - forward

## LightOnOCRForConditionalGeneration

[[autodoc]] LightOnOCRForConditionalGeneration
    - forward
