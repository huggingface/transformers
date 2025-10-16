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
*This model was released on 2021-10-15 and added to Hugging Face Transformers on 2021-12-07 and contributed by [ryo0634](https://huggingface.co/ryo0634).*

# mLUKE

[mLUKE](https://huggingface.co/papers/2110.08151) extends XLM-RoBERTa by incorporating entity embeddings, enhancing its performance on cross-lingual tasks involving entities. Trained on 24 languages, mLUKE consistently outperforms word-based models in various transfer tasks. The model's ability to extract language-agnostic features through entity representations is highlighted, and it demonstrates superior performance in a multilingual cloze prompt task using the mLAMA dataset.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="studio-ousia/mluke-base", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("studio-ousia/mluke-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("studio-ousia/mluke-base")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## MLukeTokenizer

[[autodoc]] MLukeTokenizer
    - __call__
    - save_vocabulary

