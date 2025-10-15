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
*This model was released on 2021-12-15 and added to Hugging Face Transformers on 2022-06-13 and contributed by [stancld](https://huggingface.co/stancld).*

# LongT5

[LongT5: Efficient Text-To-Text Transformer for Long Sequences](https://huggingface.co/papers/2112.07916) integrates attention mechanisms from long-input transformers and pre-training strategies from summarization models into the T5 architecture. It introduces a new attention mechanism called Transient Global (TGlobal), which combines local and global attention without additional side-inputs. LongT5 achieves state-of-the-art results in summarization tasks and outperforms the original T5 models in question answering tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="Stancld/longt5-tglobal-large-16384-pubmed-3k_steps", dtype="auto",)
pipeline("""
The statistics of base-pair usage within known recognition sites for a particular DNA-binding protein can be used to estimate the relative protein binding affinities to these sites, as well as to sites containing any other combinations of base-pairs. As has been described elsewhere, the connection between base-pair statistics and binding free energy is made by an equal probability selection assumption; i.e. that all base-pair sequences that provide appropriate binding strength are equally likely to have been chosen as recognition sites in the course of evolution. This is analogous to a statistical-mechanical system where all configurations with the same energy are equally likely to occur. In this communication, we apply the statistical-mechanical selection theory to analyze the base-pair statistics of the known recognition sequences for the cyclic AMP receptor protein (CRP). The theoretical predictions are found to be in reasonable agreement with binding data for those sequences for which experimental binding information is available, thus lending support to the basic assumptions of the selection theory. On the basis of this agreement, we can predict the affinity for CRP binding to any base-pair sequence, albeit with a large statistical uncertainty. When the known recognition sites for CRP are ranked according to predicted binding affinities, we find that the ranking is consistent with the hypothesis that the level of function of these sites parallels their fractional saturation with CRP-cAMP under in-vivo conditions. When applied to the entire genome, the theory predicts the existence of a large number of randomly occurring "pseudosites" with strong binding affinity for CRP. It appears that most CRP molecules are engaged in non-productive binding at non-specific or pseudospecific sites under in-vivo conditions. In this sense, the specificity of the CRP binding site is very low. Relative specificity requirements for polymerases, repressors and activators are compared in light of the results of this and the first paper in this series.
"""
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
model = AutoModelForSeq2SeqLM.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps", dtype="auto",)

text = """
The statistics of base-pair usage within known recognition sites for a particular DNA-binding protein can be used to estimate the relative protein binding affinities to these sites, as well as to sites containing any other combinations of base-pairs. As has been described elsewhere, the connection between base-pair statistics and binding free energy is made by an equal probability selection assumption; i.e. that all base-pair sequences that provide appropriate binding strength are equally likely to have been chosen as recognition sites in the course of evolution. This is analogous to a statistical-mechanical system where all configurations with the same energy are equally likely to occur. In this communication, we apply the statistical-mechanical selection theory to analyze the base-pair statistics of the known recognition sequences for the cyclic AMP receptor protein (CRP). The theoretical predictions are found to be in reasonable agreement with binding data for those sequences for which experimental binding information is available, thus lending support to the basic assumptions of the selection theory. On the basis of this agreement, we can predict the affinity for CRP binding to any base-pair sequence, albeit with a large statistical uncertainty. When the known recognition sites for CRP are ranked according to predicted binding affinities, we find that the ranking is consistent with the hypothesis that the level of function of these sites parallels their fractional saturation with CRP-cAMP under in-vivo conditions. When applied to the entire genome, the theory predicts the existence of a large number of randomly occurring "pseudosites" with strong binding affinity for CRP. It appears that most CRP molecules are engaged in non-productive binding at non-specific or pseudospecific sites under in-vivo conditions. In this sense, the specificity of the CRP binding site is very low. Relative specificity requirements for polymerases, repressors and activators are compared in light of the results of this and the first paper in this series.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- [`LongT5ForConditionalGeneration`] extends [`T5ForConditionalGeneration`] by replacing the traditional encoder self-attention layer with efficient local attention or transient-global (tglobal) attention.
- Unlike T5, LongT5 doesn't use a task prefix. It uses a different pre-training objective inspired by [`PegasusForConditionalGeneration`].
- LongT5 works efficiently on long-range sequence-to-sequence tasks where input sequences exceed 512 tokens. It handles input sequences up to 16,384 tokens.
- Local attention uses a sparse sliding-window operation. A token attends only to r tokens to the left and right (r=127 by default). Local attention doesn't introduce new parameters. Complexity is linear: O(l*r).
- Transient Global Attention extends Local Attention. Each input token interacts with all other tokens in the layer. This splits input sequences into blocks of fixed length k (k=16 by default).
- A global token for each block is obtained by summing and normalizing embeddings of every token in the block. Each token attends to nearby tokens (like Local attention) and every global token (like standard global attention).
- TGlobal attention introduces new parameters: global relative position biases and layer normalization for global token embeddings. Complexity is O(l(r + l/k)).

## LongT5Config

[[autodoc]] LongT5Config

## LongT5Model

[[autodoc]] LongT5Model
    - forward

## LongT5ForConditionalGeneration

[[autodoc]] LongT5ForConditionalGeneration
    - forward

## LongT5EncoderModel

[[autodoc]] LongT5EncoderModel
    - forward

