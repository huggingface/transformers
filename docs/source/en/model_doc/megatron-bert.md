<!--Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-09-17 and added to Hugging Face Transformers on 2021-04-08 and contributed by [jdemouth](https://huggingface.co/jdemouth).*

# MegatronBERT

[MegatronBERT](https://huggingface.co/papers/1909.08053) presents techniques for training very large transformer models using intra-layer model parallelism, enabling the training of models with billions of parameters on 512 GPUs. This approach achieves 15.1 PetaFLOPs with 76% scaling efficiency. The model demonstrates state-of-the-art results on datasets like WikiText103, LAMBADA, and RACE, with careful attention to layer normalization placement being crucial for BERT-like models as size increases.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="IDEA-CCNL/Erlangshen-MegatronBert-1.3B", dtype="auto")
pipeline("植物通过<mask>合作用产生能量")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")

inputs = tokenizer("植物通过<mask>合作用产生能量", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## MegatronBertConfig

[[autodoc]] MegatronBertConfig

## MegatronBertModel

[[autodoc]] MegatronBertModel
    - forward

## MegatronBertForMaskedLM

[[autodoc]] MegatronBertForMaskedLM
    - forward

## MegatronBertForCausalLM

[[autodoc]] MegatronBertForCausalLM
    - forward

## MegatronBertForNextSentencePrediction

[[autodoc]] MegatronBertForNextSentencePrediction
    - forward

## MegatronBertForPreTraining

[[autodoc]] MegatronBertForPreTraining
    - forward

## MegatronBertForSequenceClassification

[[autodoc]] MegatronBertForSequenceClassification
    - forward

## MegatronBertForMultipleChoice

[[autodoc]] MegatronBertForMultipleChoice
    - forward

## MegatronBertForTokenClassification

[[autodoc]] MegatronBertForTokenClassification
    - forward

## MegatronBertForQuestionAnswering

[[autodoc]] MegatronBertForQuestionAnswering
    - forward

