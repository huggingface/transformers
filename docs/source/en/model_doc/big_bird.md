<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-07-28 and added to Hugging Face Transformers on 2021-03-30.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# BigBird

[BigBird](https://huggingface.co/papers/2007.14062) is a transformer model built to handle sequence lengths up to 4096 compared to 512 for [BERT](./bert). Traditional transformers struggle with long inputs because attention gets really expensive as the sequence length grows. BigBird fixes this by using a sparse attention mechanism, which means it doesn’t try to look at everything at once. Instead, it mixes in local attention, random attention, and a few global tokens to process the whole input. This combination gives it the best of both worlds. It keeps the computation efficient while still capturing enough of the sequence to understand it well. Because of this, BigBird is great at tasks involving long documents, like question answering, summarization, and genomic applications.

You can find all the original BigBird checkpoints under the [Google](https://huggingface.co/google?search_models=bigbird) organization.

> [!TIP]
> Click on the BigBird models in the right sidebar for more examples of how to apply BigBird to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google/bigbird-roberta-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/bigbird-roberta-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "google/bigbird-roberta-base",
    dtype=torch.float16,
    device_map="auto",
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
!echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers-cli run --task fill-mask --model google/bigbird-roberta-base --device 0
```
</hfoption>
</hfoptions>

## Notes
- Inputs should be padded on the right because BigBird uses absolute position embeddings.
- BigBird supports `original_full` and `block_sparse` attention. If the input sequence length is less than 1024, it is recommended to use `original_full` since sparse patterns don't offer much benefit for smaller inputs.
- The current implementation uses window size of 3 blocks and 2 global blocks, only supports the ITC-implementation, and doesn't support `num_random_blocks=0`.
- The sequence length must be divisible by the block size.

## Resources

- Read the [BigBird](https://huggingface.co/blog/big-bird) blog post for more details about how its attention works.

## BigBirdConfig

[[autodoc]] BigBirdConfig

## BigBirdTokenizer

[[autodoc]] BigBirdTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BigBirdTokenizerFast

[[autodoc]] BigBirdTokenizerFast

## BigBird specific outputs

[[autodoc]] models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput

## BigBirdModel

[[autodoc]] BigBirdModel
    - forward

## BigBirdForPreTraining

[[autodoc]] BigBirdForPreTraining
    - forward

## BigBirdForCausalLM

[[autodoc]] BigBirdForCausalLM
    - forward

## BigBirdForMaskedLM

[[autodoc]] BigBirdForMaskedLM
    - forward

## BigBirdForSequenceClassification

[[autodoc]] BigBirdForSequenceClassification
    - forward

## BigBirdForMultipleChoice

[[autodoc]] BigBirdForMultipleChoice
    - forward

## BigBirdForTokenClassification

[[autodoc]] BigBirdForTokenClassification
    - forward

## BigBirdForQuestionAnswering

[[autodoc]] BigBirdForQuestionAnswering
    - forward
