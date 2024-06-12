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

# Reformer

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=reformer">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-reformer-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/reformer-crime-and-punishment">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The Reformer model was proposed in the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451.pdf) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

The abstract from the paper is the following:

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can
be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its
complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only once in the training process instead of
N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models
while being much more memory-efficient and much faster on long sequences.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors' code can be
found [here](https://github.com/google/trax/tree/master/trax/models/reformer).

## Usage tips

- Reformer does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035).
- Use Axial position encoding (see below for more details). It’s a mechanism to avoid having a huge positional encoding matrix (when the sequence length is very big) by factorizing it into smaller matrices.
- Replace traditional attention by LSH (local-sensitive hashing) attention (see below for more details). It’s a technique to avoid computing the full product query-key in the attention layers.
- Avoid storing the intermediate results of each layer by using reversible transformer layers to obtain them during the backward pass (subtracting the residuals from the input of the next layer gives them back) or recomputing them for results inside a given layer (less efficient than storing them but saves memory).
- Compute the feedforward operations by chunks and not on the whole batch.

### Axial Positional Encodings

Axial Positional Encodings were first implemented in Google's [trax library](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29)
and developed by the authors of this model's paper. In models that are treating very long input sequences, the
conventional position id encodings store an embeddings vector of size \\(d\\) being the `config.hidden_size` for
every position \\(i, \ldots, n_s\\), with \\(n_s\\) being `config.max_embedding_size`. This means that having
a sequence length of \\(n_s = 2^{19} \approx 0.5M\\) and a `config.hidden_size` of \\(d = 2^{10} \approx 1000\\)
would result in a position encoding matrix:

$$X_{i,j}, \text{ with } i \in \left[1,\ldots, d\right] \text{ and } j \in \left[1,\ldots, n_s\right]$$

which alone has over 500M parameters to store. Axial positional encodings factorize \\(X_{i,j}\\) into two matrices:

$$X^{1}_{i,j}, \text{ with } i \in \left[1,\ldots, d^1\right] \text{ and } j \in \left[1,\ldots, n_s^1\right]$$

and

$$X^{2}_{i,j}, \text{ with } i \in \left[1,\ldots, d^2\right] \text{ and } j \in \left[1,\ldots, n_s^2\right]$$

with:

$$d = d^1 + d^2 \text{ and } n_s = n_s^1 \times n_s^2 .$$

Therefore the following holds:

$$X_{i,j} = \begin{cases}
X^{1}_{i, k}, & \text{if }\ i < d^1 \text{ with } k = j \mod n_s^1 \\
X^{2}_{i - d^1, l}, & \text{if } i \ge d^1 \text{ with } l = \lfloor\frac{j}{n_s^1}\rfloor
\end{cases}$$

Intuitively, this means that a position embedding vector \\(x_j \in \mathbb{R}^{d}\\) is now the composition of two
factorized embedding vectors: \\(x^1_{k, l} + x^2_{l, k}\\), where as the `config.max_embedding_size` dimension
\\(j\\) is factorized into \\(k \text{ and } l\\). This design ensures that each position embedding vector
\\(x_j\\) is unique.

Using the above example again, axial position encoding with \\(d^1 = 2^9, d^2 = 2^9, n_s^1 = 2^9, n_s^2 = 2^{10}\\)
can drastically reduced the number of parameters from 500 000 000 to \\(2^{18} + 2^{19} \approx 780 000\\) parameters, this means 85% less memory usage.

In practice, the parameter `config.axial_pos_embds_dim` is set to a tuple \\((d^1, d^2)\\) which sum has to be
equal to `config.hidden_size` and `config.axial_pos_shape` is set to a tuple \\((n_s^1, n_s^2)\\) which
product has to be equal to `config.max_embedding_size`, which during training has to be equal to the *sequence
length* of the `input_ids`.


### LSH Self Attention

In Locality sensitive hashing (LSH) self attention the key and query projection weights are tied. Therefore, the key
query embedding vectors are also tied. LSH self attention uses the locality sensitive hashing mechanism proposed in
[Practical and Optimal LSH for Angular Distance](https://arxiv.org/abs/1509.02897) to assign each of the tied key
query embedding vectors to one of `config.num_buckets` possible buckets. The premise is that the more "similar"
key query embedding vectors (in terms of *cosine similarity*) are to each other, the more likely they are assigned to
the same bucket.

The accuracy of the LSH mechanism can be improved by increasing `config.num_hashes` or directly the argument
`num_hashes` of the forward function so that the output of the LSH self attention better approximates the output
of the "normal" full self attention. The buckets are then sorted and chunked into query key embedding vector chunks
each of length `config.lsh_chunk_length`. For each chunk, the query embedding vectors attend to its key vectors
(which are tied to themselves) and to the key embedding vectors of `config.lsh_num_chunks_before` previous
neighboring chunks and `config.lsh_num_chunks_after` following neighboring chunks.

For more information, see the [original Paper](https://arxiv.org/abs/2001.04451) or this great [blog post](https://www.pragmatic.ml/reformer-deep-dive/).

Note that `config.num_buckets` can also be factorized into a list \\((n_{\text{buckets}}^1,
n_{\text{buckets}}^2)\\). This way instead of assigning the query key embedding vectors to one of \\((1,\ldots,
n_{\text{buckets}})\\) they are assigned to one of \\((1-1,\ldots, n_{\text{buckets}}^1-1, \ldots,
1-n_{\text{buckets}}^2, \ldots, n_{\text{buckets}}^1-n_{\text{buckets}}^2)\\). This is crucial for very long sequences to
save memory.

When training a model from scratch, it is recommended to leave `config.num_buckets=None`, so that depending on the
sequence length a good value for `num_buckets` is calculated on the fly. This value will then automatically be
saved in the config and should be reused for inference.

Using LSH self attention, the memory and time complexity of the query-key matmul operation can be reduced from
\\(\mathcal{O}(n_s \times n_s)\\) to \\(\mathcal{O}(n_s \times \log(n_s))\\), which usually represents the memory
and time bottleneck in a transformer model, with \\(n_s\\) being the sequence length.


### Local Self Attention

Local self attention is essentially a "normal" self attention layer with key, query and value projections, but is
chunked so that in each chunk of length `config.local_chunk_length` the query embedding vectors only attends to
the key embedding vectors in its chunk and to the key embedding vectors of `config.local_num_chunks_before`
previous neighboring chunks and `config.local_num_chunks_after` following neighboring chunks.

Using Local self attention, the memory and time complexity of the query-key matmul operation can be reduced from
\\(\mathcal{O}(n_s \times n_s)\\) to \\(\mathcal{O}(n_s \times \log(n_s))\\), which usually represents the memory
and time bottleneck in a transformer model, with \\(n_s\\) being the sequence length.


### Training

During training, we must ensure that the sequence length is set to a value that can be divided by the least common
multiple of `config.lsh_chunk_length` and `config.local_chunk_length` and that the parameters of the Axial
Positional Encodings are correctly set as described above. Reformer is very memory efficient so that the model can
easily be trained on sequences as long as 64000 tokens.

For training, the [`ReformerModelWithLMHead`] should be used as follows:

```python
input_ids = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
loss = model(input_ids, labels=input_ids)[0]
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)

## ReformerConfig

[[autodoc]] ReformerConfig

## ReformerTokenizer

[[autodoc]] ReformerTokenizer
    - save_vocabulary

## ReformerTokenizerFast

[[autodoc]] ReformerTokenizerFast

## ReformerModel

[[autodoc]] ReformerModel
    - forward

## ReformerModelWithLMHead

[[autodoc]] ReformerModelWithLMHead
    - forward

## ReformerForMaskedLM

[[autodoc]] ReformerForMaskedLM
    - forward

## ReformerForSequenceClassification

[[autodoc]] ReformerForSequenceClassification
    - forward

## ReformerForQuestionAnswering

[[autodoc]] ReformerForQuestionAnswering
    - forward
