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
*This model was released on 2019-01-09 and added to Hugging Face Transformers on 2023-06-20 and contributed by [thomwolf](https://huggingface.co/thomwolf).*

> [!WARNING]
> This model is in maintenance mode only, so we won’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.35.0. You can do so by running the following command: pip install -U transformers==4.35.0.
>
> This model was deprecated due to security issues linked to `pickle.load`. To continue using TransfoXL, use a specific revision to ensure you're downloading safe files from the Hub and set the environment variable `TRUST_REMOTE_CODE` to `True`.
>
> ```py
> import os
> from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
> 
> os.environ["TRUST_REMOTE_CODE"] = "True"
> 
> checkpoint = 'transfo-xl/transfo-xl-wt103'
> revision = '40a186da79458c9f9de846edfaea79c412137f97'
> 
> tokenizer = TransfoXLTokenizer.from_pretrained(checkpoint, revision=revision)
> model = TransfoXLLMHeadModel.from_pretrained(checkpoint, revision=revision)
> ```

# Transformer XL

[Transformer-XL](https://huggingface.co/papers/1901.02860) extends the Transformer architecture with a segment-level recurrence mechanism and relative positional encoding to handle longer-term dependencies without losing temporal coherence. It achieves significant improvements in capturing long-range dependencies, outperforming RNNs and vanilla Transformers in both short and long sequences. Transformer-XL demonstrates state-of-the-art results on various benchmarks, including enwiki8, text8, WikiText-103, One Billion Word, and Penn Treebank, and can generate coherent text articles with thousands of tokens.

## TransfoXLConfig

[[autodoc]] TransfoXLConfig

## TransfoXLTokenizer

[[autodoc]] TransfoXLTokenizer
    - save_vocabulary

## TransfoXL specific outputs

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput

## TransfoXLModel

[[autodoc]] TransfoXLModel
    - forward

## TransfoXLLMHeadModel

[[autodoc]] TransfoXLLMHeadModel
    - forward

## TransfoXLForSequenceClassification

[[autodoc]] TransfoXLForSequenceClassification
    - forward

## Internal Layers

[[autodoc]] AdaptiveEmbedding

[[autodoc]] TFAdaptiveEmbedding

