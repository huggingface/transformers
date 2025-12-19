<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-07-26 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
        <img alt="Flax" src="https://img.shields.io/badge/Flax-FFB000?style=flat&logo=flax&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# RoBERTa

[RoBERTa](https://huggingface.co/papers/1907.11692) is like BERT's smarter cousin - it takes everything BERT does well and makes it even better! The key insight was that BERT wasn't actually trained enough, so RoBERTa uses a more robust training strategy with dynamic masking (instead of static), removes the next sentence prediction task, and trains on way more data. This makes RoBERTa particularly great for tasks like sentiment analysis, text classification, and understanding language nuances that BERT might miss.

You can find all the original RoBERTa checkpoints under the [roberta](https://huggingface.co/models?search=roberta) collection.

> [!TIP]
> This model was contributed by [Joao Gante](https://huggingface.co/joaogante). Click on the RoBERTa models in the right sidebar for more examples of how to apply RoBERTa to different language tasks.

The example below demonstrates how to analyze sentiment with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="FacebookAI/roberta-base",
    dtype=torch.float16,
    device=0
)
# Returns: [{'sequence': 'I love using RoBERTa for NLP tasks!', 'score': 0.95, 'token': 5, 'token_str': 'RoBERTa'}]
pipeline("I love using <mask> for NLP tasks!")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/roberta-base",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# Predict masked token in a sample sentence
text = "I love using <mask> for NLP tasks!"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

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
echo "I love using <mask> for NLP tasks!" | transformers run --task fill-mask --model FacebookAI/roberta-base --device 0
```

</hfoption>
</hfoptions>

## Resources

A list of official Hugging Face and community resources to help you get started with RoBERTa.

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://huggingface.co/papers/1907.11692) - The original paper
- [Official RoBERTa implementation](https://github.com/pytorch/fairseq/tree/main/examples/roberta) - Facebook AI's original code
- [Understanding RoBERTa: A Complete Guide](https://huggingface.co/blog/roberta) - Comprehensive blog post about RoBERTa
- [Fine-tuning RoBERTa for Text Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) - Official training guide
- [RoBERTa vs BERT: What's the Difference?](https://huggingface.co/blog/roberta-vs-bert) - Comparison article

## Notes

- RoBERTa doesn't have `token_type_ids` so you don't need to indicate which token belongs to which segment. Separate your segments with the separation token `tokenizer.sep_token` or `</s>`.
- Unlike BERT, RoBERTa uses dynamic masking during training, which means the model sees different masked tokens in each epoch, making it more robust.
- RoBERTa uses a byte-level BPE tokenizer, which handles out-of-vocabulary words better than BERT's WordPiece tokenizer.

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer
    - get_special_tokens_mask
    - save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast

## RobertaModel

[[autodoc]] RobertaModel
    - forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM
    - forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM
    - forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification
    - forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice
    - forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification
    - forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering
    - forward
