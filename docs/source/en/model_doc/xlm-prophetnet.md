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
*This model was released on 2020-01-13 and added to Hugging Face Transformers on 2023-06-20.*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# XLM-ProphetNet

[XLM-ProphetNet](https://huggingface.co/papers/2001.04063) is an encoder-decoder model designed for future n-gram prediction, a novel self-supervised objective that predicts the next n tokens simultaneously based on previous context tokens. This approach encourages the model to plan for future tokens and reduces overfitting on local correlations. Trained on the multilingual XGLUE dataset, XLM-ProphetNet achieves state-of-the-art results on abstractive summarization and question generation tasks using both base and large-scale pretraining datasets.

## XLMProphetNetConfig

[[autodoc]] XLMProphetNetConfig

## XLMProphetNetTokenizer

[[autodoc]] XLMProphetNetTokenizer

## XLMProphetNetModel

[[autodoc]] XLMProphetNetModel

## XLMProphetNetEncoder

[[autodoc]] XLMProphetNetEncoder

## XLMProphetNetDecoder

[[autodoc]] XLMProphetNetDecoder

## XLMProphetNetForConditionalGeneration

[[autodoc]] XLMProphetNetForConditionalGeneration

## XLMProphetNetForCausalLM

[[autodoc]] XLMProphetNetForCausalLM

