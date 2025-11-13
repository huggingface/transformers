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
*This model was released on 2020-03-02 and added to Hugging Face Transformers on 2020-11-16 and contributed by [dqnguyen](https://huggingface.co/dqnguyen).*

# PhoBERT

[PhoBERT](https://huggingface.co/papers/2020.findings-emnlp.92.pdf) introduces PhoBERT-base and PhoBERT-large, the first large-scale monolingual language models pre-trained for Vietnamese. Experiments demonstrate that PhoBERT outperforms XLM-R in various Vietnamese-specific NLP tasks such as Part-of-speech tagging, Dependency parsing, Named-entity recognition, and Natural language inference.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("fill-mask", model="vinai/phobert-base", dtype="auto")
pipeline("Thực vật tự tạo ra <mask> thông qua một quá trình được gọi là quang hợp.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

inputs = tokenizer("Thực vật tự tạo ra <mask> thông qua một quá trình được gọi là quang hợp.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## PhobertTokenizer

[[autodoc]] PhobertTokenizer

