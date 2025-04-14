<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Introduction to Attention Mechanisms in Transformers
Attention is at the core of every transformer architecture. The title of the groundbreaking paper "**Attention Is All You Need**" by Vaswani et al. is no coincidence. In this work, they presented the "Scaled Dot-Product Attention" mechanism, which was then extended by the splitting of the tensors into multiple heads (Multi-Head Attention), enabling parallelization and improved feature extraction. Since then, many variations have emerged, each addressing specific challenges such as computational efficiency, sequence length limitations, and others.

At a high level, attention is an ingenious mechanism through which the model learns to focus on the most relevant parts of the input data when making predictions or generating inputs. Here, we provide a brief introduction on the workings of the most popular attention mechanisms.

## Query, Key, and Value

Imagine you are particularly interested in learning about the Big Bang — this is your query, $Q$; your focus. You walk into a library, and each book has a detailed index that summarizes its content — this is the key, $K$. The value, $V$, is the actual information you extract from the books; paragraphs, or whole chapters. First, you compare the match between your query and key. This comparison provides you with attention scores, i.e., which books are the most relevant. Finally, to accomplish your task, you gather information from various different books that may have some similarity with your query — for example, cosmology, biology, and even some religious texts.

Technically, $Q$, $K$, $V$ are individually obtained by multiplying the input embedding $X$ with their respective projection matrices $W^Q$ for the query, $W^K$ for the key, and $W^V$ for the value. Additionally, $Q, K \in \mathbb{R}^{t\times d_k}$, and $V \in \mathbb{R}^{t\times d_v}$. To simplify the discussion and notation, we will assume $d_v = d_k = d$.

## Scaled Dot-Product Attention

The dot product between two vectors is a measure of their similarity. Orthogonal vectors have a dot product equal to 0, as their projection onto each other is null. This notion of similarity can be extended to matrices, as their rows and columns can be viewed as vectors. 

The first step in the Scaled Dot-Product Attention (SDPA) involves computing the *attention scores* $Q\cdot K^T$, or the level of similarity between the query and the key. This product is then divided by a scaling factor $\sqrt{d_k}$, yielding a zero-mean and unit-variance product (provided the elements of $Q$ and $K$ are already normalized). The *softmax* function, which acts as a normalizer, is then applied to this product to yield the *attention weights*: $\text{softmax}(\frac{Q\cdot K^T}{\sqrt{d_k}})$. 
Finally, the attention weights matrix is multiplied by the value $V$:
$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q\cdot K^T}{\sqrt{d_k}})V$$

In the context of sequence processing, for a sequence of length $t$, the attention weights matrix will be $t \times t$. Note that the $ij$ element of this matrix comes from the multiplication between the $i$-th row vector of $Q$ and the $j$-th column vector of $K^T$, i.e., the $j$-th row of $K$. Since $Q$, $K$ and $V$ have dimensions $t \times d$, each row is a vector of a token embedded in a $d$-dimensional space. Thus, the $ij$ element of the attention weight matrix represents the similarity between the latent (embedded) representation between the $i$-th query token and the $j$-th key token.

<div align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/a5fce677dfff6da5a76534e53a7c89ea1ae19bd9/query-key-matmul.svg" alt="Query and Key Matrix multiplication">
</div>

Finally, the attention weight matrix multiplies the value matrix $V$, effectively taking a weighted sum of the sequence tokens embedded in the value space, extracting the information it learns to be most important. 

It is noteworthy to observe that the query, key and value are not static but rather different representations learned throughout the course of the training process.

In sum, the attention weights matrix guides how much attention the $i$-th query token should pay to the $j$-th key token for the downstream task. This allows the model to dynamically determine which parts of the sequence are most relevant when processing each token.

## Multi-Head Attention

The original transformer paper not only presented the Scaled Dot-Product Attention, but also its implementation applied to multiple "heads", which is known as **Multi-head attention**. A *head* is an individual attention mechanism that operates in parallel with other heads. Its mathematical formulation is as follows:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
where $$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

$AW_i^A$ represents the projection of $A$ in the $i$-th head. It is important to note that if the query, key, and value lie in a $d$-dimensional space and the model has $n$ heads, the projections at each of the heads will convert the respective query, key and value to a space of $d/n$ dimensions. After the concatenation of the heads, they are projected again to a $d$-dimensional space ($W^O$ matrix). 

The effectiveness of employing multiple heads is similar to that of an ensemble, in which each head learns different aspects of the sequence. Moreover, it enables parallel computation, which makes it more computionally efficient. In language modeling, for example, one could say that one head focuses on subject-verb interactions, while another head focus on noun-subject. However, in reality, the interactions during the learning process are much more complex and not as well-defined as in this analogy. 

<div align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/a5fce677dfff6da5a76534e53a7c89ea1ae19bd9/multi-head-attention.svg" alt="Multi-Head Attention" style="width:29%;" >
</div>

## Self-attention, causal self-attention, and cross-attention

**Self-attention** occurs when a token attends to other tokens in the same sequence, meaning the query, key, and value all originate from the same source. In the original Transformer model, self-attention is used in both the encoder and decoder. However, the decoder employs **causal self-attention**, also known as 'Masked Multi-Head Attention'. This ensures, the model is prevented from attending tokens it has not seen yet, i.e., future tokens. Therefore, it is commonly present in autoregressive or *causal* language models, whose main purpose is sequence generation. The causal mask assigns large negative values (e.g., $-\infty$) to attention scores of future tokens before applying the softmax. This forces their attention weights to be zero, ensuring that no information leaks from future tokens. 

In contrast to self-attention, **cross-attention** is an attention mechanism in which the query, key and value do not come from the same source. In the original Transformer model, the output of the final encoder layer is fed as the key and value for the second multi-head attention module of the transformer. The query comes from the decoder (causal) self-attention module. In cross-attention, the query comes from the decoder (based on the current token in the target sequence), while the key (which provides the "index") and value (which carries actual encoded information) come from the encoder. Therefore, the network is trying to make sense of how the past tokens influence the token being currently generated. 

In the context of machine translation, suppose the task is to translate "The cat sat on the mat" from English to French, "Le chat était assis sur le tapis", which is the target sequence. During token generation, the query (from the decoder) attends to the entire encoded English sentence, which serves as the key and value. This allows the model to determine how different parts of the source sentence influence the next token to be generated in the target language.

The figure below illustrates where each of these attention mechanisms is present in the transformer model. 

<div align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/a5fce677dfff6da5a76534e53a7c89ea1ae19bd9/transformer-attentions.svg" alt="Highlighted attention mechanisms of a Transformer">
</div>


## Sparse Attention Mechanisms

Most transformer models use full attention in the sense that the attention matrix is square. It can be a big computational bottleneck when you have long sequences. Longformer and reformer are models that try to be more efficient and
use a sparse version of the attention matrix to speed up training.

### Locality-Sensitive Hashing Attention

[Reformer](model_doc/reformer) uses Locality-Sensitive Hashing (LSH) Attention. In the $ \text{softmax}(Q\cdot K^T)$, only the biggest elements (in the softmax
dimension) of the matrix $Q\cdot K^T$ are going to give useful contributions. So for each query q in Q, we can consider only
the keys $k$ in $K$ that are close to $q$. A hash function is used to determine if $q$ and $k$ are close. The attention mask is
modified to mask the current token (except at the first position), because it will give a query and a key equal (so
very similar to each other). Since the hash can be a bit random, several hash functions are used in practice
(determined by a n_rounds parameter) and then are averaged together.

### Local Attention for Efficient Sequence Processing

[Longformer](model_doc/longformer) uses local attention: often, the local context (e.g., what are the two tokens to the left and right?) is enough to take action for a given token. Also, by stacking attention layers that have a small window, the last layer will have a receptive field of more than just the tokens in the window, allowing them to build a representation of the whole sentence.

Some preselected input tokens are also given global attention: for those few tokens, the attention matrix can access
all tokens and this process is symmetric: all other tokens have access to those specific tokens (on top of the ones in
their local window). This is shown in Figure 2d of the paper, see below for a sample attention mask:

<div align="center">
<img scale="50 %" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png"/>
</div>

Using those attention matrices with less parameters then allows the model to have inputs having a bigger sequence
length.

#### Other tricks

##### Axial positional encodings

[Reformer](model_doc/reformer) uses axial positional encodings: in traditional transformer models, the positional encoding
$E$ is a matrix of size $l$ by $d$, where $l$ is the sequence length and $d$ is the dimension of the
hidden state. If you have very long texts, this matrix can be huge and take way too much space on the GPU. To alleviate
that, axial positional encodings consist of factorizing that big matrix $E$ into two smaller matrices $E_1$ and $E_2$, with
dimensions $l_1 \times d_1$ and $l_2 \times d_2$, such that $l_1 \times l_2 = l$ and
$d_1 + d_2 = d$ (with the product for the lengths, this ends up being way smaller). The embedding for time
step $j$ in $E$ is obtained by concatenating the embeddings for timestep $j\mod l_1$ in $E_1$ and $j // l_1$
in $E_2$.