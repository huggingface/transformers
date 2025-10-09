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
*This model was released on 2019-03-24 and added to Hugging Face Transformers on 2020-11-16 and contributed by [cl-tohoku](https://huggingface.co/cl-tohoku).*

# BertJapanese

[BERTJapanese](https://github.com/cl-tohoku/bert-japanese) is a collection of pretrained BERT models for Japanese, developed at Tohoku University and released on Hugging Face. The models follow the original BERT architecture, with base models (12 layers, 768 hidden units, 12 heads) and large models (24 layers, 1024 hidden units, 16 heads). Training was performed on large-scale Japanese corpora such as Wikipedia and the Japanese portion of Common Crawl, with different tokenization strategies including subword and character-based. Multiple versions exist (v1, v2, v3), improving coverage and accuracy for Japanese natural language processing tasks

Run the command below to install the Japanese dependencies.

```bash
!pip install transformers["ja"]
```

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="tohoku-nlp/bert-base-japanese", dtype="auto")
pipeline("植物は[MASK]を光合成と呼ばれる過程を通じて作り出します。")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("tohoku-nlp/bert-base-japanese", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese")

inputs = tokenizer("植物は[MASK]を光合成と呼ばれる過程を通じて作り出します。", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## BertJapaneseTokenizer

[[autodoc]] BertJapaneseTokenizer
