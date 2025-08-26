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
*This model was released on 2020-05-20 and added to Hugging Face Transformers on 2020-11-16.*

# BERTweet

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## BERTweet

[BERTweet](https://huggingface.co/papers/2005.10200) shares the same architecture as [BERT-base](./bert), but it’s pretrained like [RoBERTa](./roberta) on English Tweets. It performs really well on Tweet-related tasks like part-of-speech tagging, named entity recognition, and text classification.


You can find all the original BERTweet checkpoints under the [VinAI Research](https://huggingface.co/vinai?search_models=BERTweet) organization.

> [!TIP]
> Refer to the [BERT](./bert) docs for more examples of how to apply BERTweet to different language tasks.

The example below demonstrates how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="vinai/bertweet-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create <mask> through a process known as photosynthesis.")
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
   "vinai/bertweet-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "vinai/bertweet-base",
    dtype=torch.float16,
    device_map="auto"
)
inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt").to(model.device)

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
echo -e "Plants create <mask> through a process known as photosynthesis." | transformers-cli run --task fill-mask --model vinai/bertweet-base --device 0
```

</hfoption>
</hfoptions>

## Notes
- Use the [`AutoTokenizer`] or [`BertweetTokenizer`] because it’s preloaded with a custom vocabulary adapted to tweet-specific tokens like hashtags (#), mentions (@), emojis, and common abbreviations. Make sure to also install the [emoji](https://pypi.org/project/emoji/) library.
- Inputs should be padded on the right (`padding="max_length"`) because BERT uses absolute position embeddings.

## BertweetTokenizer

[[autodoc]] BertweetTokenizer
