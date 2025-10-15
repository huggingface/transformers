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
> This model is in maintenance mode only, so we won’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.35.0. You can do so by running the following command: pip install -U transformers==4.35.0.

# Transformer XL

[Transformer-XL](https://huggingface.co/papers/1901.02860) extends the Transformer architecture with a segment-level recurrence mechanism and relative positional encoding to handle longer-term dependencies without losing temporal coherence. It achieves significant improvements in capturing long-range dependencies, outperforming RNNs and vanilla Transformers in both short and long sequences. Transformer-XL demonstrates state-of-the-art results on various benchmarks, including enwiki8, text8, WikiText-103, One Billion Word, and Penn Treebank, and can generate coherent text articles with thousands of tokens.

## Usage tips

- Transformer-XL uses relative sinusoidal positional embeddings. Pad inputs on the left or right. The original implementation trains on SQuAD with left padding, so padding defaults to left.
- Transformer-XL has no sequence length limit, unlike most other models.
- Transformer-XL works like a regular GPT model but introduces a recurrence mechanism for consecutive segments. A segment is a number of consecutive tokens (like 512) that may span across multiple documents. Segments are fed in order to the model.
- The model concatenates hidden states from the previous segment to the current input to compute attention scores. This lets the model attend to information from both the previous and current segments. Stacking multiple attention layers increases the receptive field to multiple previous segments.
- This changes positional embeddings to relative positional embeddings. Regular positional embeddings would give the same results for the current input and current hidden state at a given position. The model makes adjustments in how attention scores are computed.
- Transformer-XL doesn't work with `torch.nn.DataParallel` due to a bug in PyTorch. See [issue #36035](https://github.com/pytorch/pytorch/issues/36035).
- This model was deprecated due to security issues with `pickle.load`. Use a specific revision to download safe files from the Hub. Set `TRUST_REMOTE_CODE=True` as an environment variable.

## TransfoXLConfig

[[autodoc]] TransfoXLConfig

## TransfoXLTokenizer

[[autodoc]] TransfoXLTokenizer
    - save_vocabulary

## TransfoXL specific outputs

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput

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
