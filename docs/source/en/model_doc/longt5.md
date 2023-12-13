<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LongT5

## Overview

The LongT5 model was proposed in [LongT5: Efficient Text-To-Text Transformer for Long Sequences](https://arxiv.org/abs/2112.07916)
by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung and Yinfei Yang. It's an
encoder-decoder transformer pre-trained in a text-to-text denoising generative setting. LongT5 model is an extension of
T5 model, and it enables using one of the two different efficient attention mechanisms - (1) Local attention, or (2)
Transient-Global attention.


The abstract from the paper is the following:

*Recent work has shown that either (1) increasing the input length or (2) increasing model size can improve the
performance of Transformer-based neural models. In this paper, we present a new model, called LongT5, with which we
explore the effects of scaling both the input length and model size at the same time. Specifically, we integrated
attention ideas from long-input transformers (ETC), and adopted pre-training strategies from summarization pre-training
(PEGASUS) into the scalable T5 architecture. The result is a new attention mechanism we call {\em Transient Global}
(TGlobal), which mimics ETC's local/global attention mechanism, but without requiring additional side-inputs. We are
able to achieve state-of-the-art results on several summarization tasks and outperform the original T5 models on
question answering tasks.*

This model was contributed by [stancld](https://huggingface.co/stancld).
The original code can be found [here](https://github.com/google-research/longt5).

## Usage tips

- [`LongT5ForConditionalGeneration`] is an extension of [`T5ForConditionalGeneration`] exchanging the traditional
encoder *self-attention* layer with efficient either *local* attention or *transient-global* (*tglobal*) attention.
- Unlike the T5 model, LongT5 does not use a task prefix. Furthermore, it uses a different pre-training objective
inspired by the pre-training of [`PegasusForConditionalGeneration`].
- LongT5 model is designed to work efficiently and very well on long-range *sequence-to-sequence* tasks where the
input sequence exceeds commonly used 512 tokens. It is capable of handling input sequences of a length up to 16,384 tokens.
- For *Local Attention*, the sparse sliding-window local attention operation allows a given token to attend only `r`
tokens to the left and right of it (with `r=127` by default). *Local Attention* does not introduce any new parameters
to the model. The complexity of the mechanism is linear in input sequence length `l`: `O(l*r)`.
- *Transient Global Attention* is an extension of the *Local Attention*. It, furthermore, allows each input token to
interact with all other tokens in the layer. This is achieved via splitting an input sequence into blocks of a fixed
length `k` (with a default `k=16`). Then, a global token for such a block is obtained via summing and normalizing the embeddings of every token
in the block. Thanks to this, the attention allows each token to attend to both nearby tokens like in Local attention, and
also every global token like in the case of standard global attention (*transient* represents the fact the global tokens
are constructed dynamically within each attention operation).  As a consequence, *TGlobal* attention introduces
a few new parameters -- global relative position biases and a layer normalization for global token's embedding.
The complexity of this mechanism is `O(l(r + l/k))`.
- An example showing how to evaluate a fine-tuned LongT5 model on the [pubmed dataset](https://huggingface.co/datasets/scientific_papers) is below.

```python
>>> import evaluate
>>> from datasets import load_dataset
>>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

>>> dataset = load_dataset("scientific_papers", "pubmed", split="validation")
>>> model = (
...     LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
...     .to("cuda")
...     .half()
... )
>>> tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")


>>> def generate_answers(batch):
...     inputs_dict = tokenizer(
...         batch["article"], max_length=16384, padding="max_length", truncation=True, return_tensors="pt"
...     )
...     input_ids = inputs_dict.input_ids.to("cuda")
...     attention_mask = inputs_dict.attention_mask.to("cuda")
...     output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=2)
...     batch["predicted_abstract"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
...     return batch


>>> result = dataset.map(generate_answer, batched=True, batch_size=2)
>>> rouge = evaluate.load("rouge")
>>> rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"])
```


## Resources

- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## LongT5Config

[[autodoc]] LongT5Config

<frameworkcontent>
<pt>

## LongT5Model

[[autodoc]] LongT5Model
    - forward

## LongT5ForConditionalGeneration

[[autodoc]] LongT5ForConditionalGeneration
    - forward

## LongT5EncoderModel

[[autodoc]] LongT5EncoderModel
    - forward

</pt>
<jax>

## FlaxLongT5Model

[[autodoc]] FlaxLongT5Model
    - __call__
    - encode
    - decode

## FlaxLongT5ForConditionalGeneration

[[autodoc]] FlaxLongT5ForConditionalGeneration
    - __call__
    - encode
    - decode

</jax>
</frameworkcontent>
