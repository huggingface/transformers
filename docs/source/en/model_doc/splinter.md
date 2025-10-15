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
*This model was released on 2021-01-02 and added to Hugging Face Transformers on 2021-08-17 and contributed by [yuvalkirstain](https://huggingface.co/yuvalkirstain) and [oriram](https://huggingface.co/oriram).*

# Splinter

[Splinter](https://huggingface.co/papers/2101.00438) is an encoder-only transformer pretrained using the recurring span selection task on a large corpus of Wikipedia and the Toronto Book Corpus. This pretraining scheme involves masking recurring spans in a passage and asking the model to select the correct span, with masked spans replaced by a special token. The model demonstrates strong performance in few-shot question answering scenarios, achieving 72.7 F1 on SQuAD with just 128 training examples, while also maintaining competitive results in high-resource settings.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("question-answering", model="tau/splinter-base", dtype="auto")
question = "How do plants create energy?"
context = "Plants create energy through a process known as photosynthesis, which converts sunlight into chemical energy using chlorophyll in their leaves."
pipeline(question=question, context=context)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import SplinterForQuestionAnswering, SplinterTokenizer

tokenizer = SplinterTokenizer.from_pretrained("tau/splinter-base")
model = SplinterForQuestionAnswering.from_pretrained("tau/splinter-base", dtype="auto")

question = "How do plants create energy?"
context = "Plants create energy through a process known as photosynthesis, which converts sunlight into chemical energy using chlorophyll in their leaves."
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_scores = outputs.start_logits[0]
end_scores = outputs.end_logits[0]

start_idx = start_scores.argmax().item()
end_idx = end_scores.argmax().item()

answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"Answer: {answer}")
print(f"Starting position: {start_idx}, Ending position: {end_idx}")
```

</hfoption>
</hfoptions>

## Usage tips

- Splinter was trained to predict answer spans conditioned on a special `[QUESTION]` token. These tokens contextualize to question representations for answer prediction.
- The QASS layer is the default behavior in [`SplinterForQuestionAnswering`]. It handles question-aware span selection.
- Use [`SplinterTokenizer`] instead of [`BertTokenizer`]. It contains the special token and uses it by default when two sequences are given.
- Keep the question token in mind when using Splinter outside `run_qa.py`. It's important for model success, especially in few-shot settings.
- Two checkpoint variants exist for each Splinter size:
  - `tau/splinter-base-qass` and `tau/splinter-large-qass`: Include pretrained QASS layer weights
  - `tau/splinter-base` and `tau/splinter-large`: Don't include QASS weights for random initialization during fine-tuning
- Random initialization of the QASS layer during fine-tuning yields better results in some cases.

## SplinterConfig

[[autodoc]] SplinterConfig

## SplinterTokenizer

[[autodoc]] SplinterTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SplinterTokenizerFast

[[autodoc]] SplinterTokenizerFast

## SplinterModel

[[autodoc]] SplinterModel
    - forward

## SplinterForQuestionAnswering

[[autodoc]] SplinterForQuestionAnswering
    - forward

## SplinterForPreTraining

[[autodoc]] SplinterForPreTraining
    - forward

