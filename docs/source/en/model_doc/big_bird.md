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
*This model was released on 2020-07-28 and added to Hugging Face Transformers on 2021-03-30 and contributed by [vasudevgupta](https://huggingface.co/vasudevgupta).*

# BigBird

[BigBird: Transformers for Longer Sequences](https://huggingface.co/papers/2007.14062) introduces a sparse-attention mechanism that reduces the quadratic dependency on sequence length to linear, enabling handling of much longer sequences compared to models like BERT. BigBird combines sparse, global, and random attention to approximate full attention efficiently. This allows it to process sequences up to 8 times longer on similar hardware, improving performance on long document NLP tasks such as question answering and summarization. Additionally, the model supports novel applications in genomics.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="google/bigbird-roberta-base", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("google/bigbird-roberta-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

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

