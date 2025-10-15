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
*This model was released on 2021-05-09 and added to Hugging Face Transformers on 2021-09-20 and contributed by [gchhablani](https://huggingface.co/gchhablani).*

# FNet

[FNet: Mixing Tokens with Fourier Transforms](https://huggingface.co/papers/2105.03824) demonstrates that Transformer encoders can be significantly accelerated by replacing self-attention layers with simple linear mixers or even an unparameterized Fourier Transform. This FNet approach achieves 92–97% of BERT’s accuracy on the GLUE benchmark while training 80% faster on GPUs and 70% faster on TPUs at standard input lengths. On longer sequences, FNet maintains competitive accuracy compared to efficient Transformers while being faster across most sequence lengths. Additionally, FNet has a smaller memory footprint, making it especially efficient for smaller models, which can outperform Transformer counterparts under the same speed and accuracy constraints.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="google/fnet-base", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("google/fnet-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Usage tips

- FNet doesn't use attention masks since it's based on Fourier Transform. The model trained with maximum sequence length 512 (including pad tokens). Use the same maximum sequence length for fine-tuning and inference.

## FNetConfig

[[autodoc]] FNetConfig

## FNetTokenizer

[[autodoc]] FNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FNetTokenizerFast

[[autodoc]] FNetTokenizerFast

## FNetModel

[[autodoc]] FNetModel
    - forward

## FNetForPreTraining

[[autodoc]] FNetForPreTraining
    - forward

## FNetForMaskedLM

[[autodoc]] FNetForMaskedLM
    - forward

## FNetForNextSentencePrediction

[[autodoc]] FNetForNextSentencePrediction
    - forward

## FNetForSequenceClassification

[[autodoc]] FNetForSequenceClassification
    - forward

## FNetForMultipleChoice

[[autodoc]] FNetForMultipleChoice
    - forward

## FNetForTokenClassification

[[autodoc]] FNetForTokenClassification
    - forward

## FNetForQuestionAnswering

[[autodoc]] FNetForQuestionAnswering
    - forward
