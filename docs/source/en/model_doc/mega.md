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
*This model was released on 2022-09-21 and added to Hugging Face Transformers on 2023-06-20 and contributed by [mnaylor](https://huggingface.co/mnaylor).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code.
>
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.


# MEGA

[Mega: Moving Average Equipped Gated Attention](https://huggingface.co/papers/2209.10655) introduces Mega, a single-head gated attention mechanism enhanced with an exponential moving average to incorporate position-aware local dependencies. This design addresses the Transformer's limitations in handling long sequences and computational inefficiency. Mega achieves competitive results on benchmarks like the Long Range Arena while using fewer parameters and offering linear time and space complexity through sequence chunking. Experiments demonstrate improvements over other sequence models in various tasks, including neural machine translation, auto-regressive language modeling, and image and speech classification.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mnaylor/mega-base-wikitext", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
model = AutoModelForCausalLM.from_pretrained("mnaylor/mega-base-wikitext", dtype="auto")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
print(f"Next predicted token: {tokenizer.decode([outputs.logits[0, -1, :].argmax().item()])}")
```

</hfoption>
</hfoptions>

## Usage tips

- MEGA performs well with relatively few parameters. See Appendix D in the MEGA paper for examples of architectural specs that perform well in various settings. If using MEGA as a decoder, set `bidirectional=False` to avoid errors with default bidirectional.
- Mega-chunk is a variant of MEGA that reduces time and space complexity from quadratic to linear. Use chunking with [`MegaConfig.use_chunking`] and control chunk size with [`MegaConfig.chunk_size`].
- The original MEGA implementation had inconsistent expectations of attention masks for padding and causal self-attention between the softmax attention and Laplace/squared ReLU method. This implementation addresses that inconsistency.
- The original implementation didn't include token type embeddings. This implementation adds support for these, controlled by [`MegaConfig.add_token_type_embeddings`].

## MegaConfig

[[autodoc]] MegaConfig

## MegaModel

[[autodoc]] MegaModel
    - forward

## MegaForCausalLM

[[autodoc]] MegaForCausalLM
    - forward

## MegaForMaskedLM

[[autodoc]] MegaForMaskedLM
    - forward

## MegaForSequenceClassification

[[autodoc]] MegaForSequenceClassification
    - forward

## MegaForMultipleChoice

[[autodoc]] MegaForMultipleChoice
    - forward

## MegaForTokenClassification

[[autodoc]] MegaForTokenClassification
    - forward

## MegaForQuestionAnswering

[[autodoc]] MegaForQuestionAnswering
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mnaylor/mega-base-wikitext", dtype="auto")
pipeline("The future of artificial intelligence is")
```

