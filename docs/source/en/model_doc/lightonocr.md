<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-11-18.*

# LightOnOCR


**LightOnOCR** is a compact, end-to-end vision–language model for Optical Character Recognition (OCR) and document understanding. It achieves state-of-the-art accuracy in its weight class while being several times faster and cheaper than larger general-purpose VLMs.

📝 **[Read the full blog post](https://huggingface.co/blog/lightonai/lightonocr/)** | 📓 **[Finetuning notebook](https://colab.research.google.com/drive/1WjbsFJZ4vOAAlKtcCauFLn_evo5UBRNa?usp=sharing)**

**Model Overview**

LightOnOCR combines a Vision Transformer encoder(Pixtral-based) with a lightweight text decoder(Qwen3-based) distilled from high-quality open VLMs. It is optimized for document parsing tasks, producing accurate, layout-aware text extraction from high-resolution pages.




## LightOnOCRConfig

[[autodoc]] LightOnOCRConfig

## LightOnOCRTextConfig

[[autodoc]] LightOnOCRTextConfig

## LightOnOCRVisionConfig

[[autodoc]] LightOnOCRVisionConfig

## LightOnOCRProcessor

[[autodoc]] LightOnOCRProcessor
    - __call__

## LightOnOCRTextModel

[[autodoc]] LightOnOCRTextModel
    - forward

## LightOnOCRVisionModel

[[autodoc]] LightOnOCRVisionModel
    - forward

## LightOnOCRModel

[[autodoc]] LightOnOCRModel
    - forward

## LightOnOCRForConditionalGeneration

[[autodoc]] LightOnOCRForConditionalGeneration
    - forward
