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
*This model was released on 2023-01-25 and added to Hugging Face Transformers on 2023-06-20 and contributed by [stefan-it](https://huggingface.co/stefan-it).*

# XLM-V

[XLM-V](https://huggingface.co/papers/2301.10472) is a multilingual language model featuring a one million token vocabulary, trained on 2.5TB of data from Common Crawl. It addresses the vocabulary bottleneck in multilingual models by optimizing token sharing and vocabulary capacity for individual languages, resulting in more semantically meaningful and shorter tokenizations. XLM-V outperforms XLM-R across various tasks, including natural language inference, question answering, named entity recognition, and low-resource tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="facebook/xlm-v-base", dtype="auto")
pipeline("Les plantes créent <mask> grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("facebook/xlm-v-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-v-base")

inputs = tokenizer("Les plantes créent <mask> grâce à un processus appelé photosynthèse.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Usage tips

- XLM-V is compatible with the XLM-RoBERTa model architecture. Only model weights from the fairseq library needed conversion.
- The [`XLMTokenizer`] implementation loads the vocabulary and performs tokenization.