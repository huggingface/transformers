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
*This model was released on 2020-10-23 and added to Hugging Face Transformers on 2020-11-27 and contributed by [moussakam](https://huggingface.co/moussakam).*

# BARThez

[BARThez](https://huggingface.co/papers/2010.12321) is the first BART model for the French language, pretrained on a large monolingual French corpus. Unlike BERT-based models like CamemBERT and FlauBERT, BARThez includes both an encoder and a decoder pretrained, making it well-suited for generative tasks. Evaluated on the FLUE benchmark and a new summarization dataset, OrangeSum, BARThez demonstrates strong performance. Additionally, continuing the pretraining of multilingual BART on BARThez's corpus results in mBARTHez, which outperforms or matches CamemBERT and FlauBERT.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("fill-mask", model="moussaKam/barthez", dtype="auto")
pipeline("Les plantes créent <mask> grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("moussaKam/barthez", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")

inputs = tokenizer("Les plantes créent <mask> grâce à un processus appelé photosynthèse.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## BarthezTokenizer

[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast

[[autodoc]] BarthezTokenizerFast
