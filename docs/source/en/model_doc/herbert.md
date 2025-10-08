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
*This model was released on 2020-05-01 and added to Hugging Face Transformers on 2020-11-16 and contributed by [rmroczkowski](https://huggingface.co/rmroczkowski).*

# HerBERT

[HerBERT](https://huggingface.co/papers/2005.00630) is a BERT-based Language Model trained on Polish corpora using only the MLM objective with dynamic masking of whole words. It was introduced alongside a comprehensive multi-task benchmark for Polish language understanding, KLEJ, which includes a diverse set of tasks from named entity recognition, question-answering, textual entailment, and a new sentiment analysis task for e-commerce domain reviews. HerBERT achieves the best average performance and top results in three out of nine tasks in the benchmark.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", tokenizer="allegro/herbert-klej-cased-tokenizer-v1", model="allegro/herbert-klej-cased-v1", dtype="auto")
pipeline("Rośliny tworzą <mask> w procesie zwanym fotosyntezą.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("allegro/herbert-klej-cased-v1", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")

inputs = tokenizer("Rośliny tworzą <mask> w procesie zwanym fotosyntezą.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## HerbertTokenizer

[[autodoc]] HerbertTokenizer

## HerbertTokenizerFast

[[autodoc]] HerbertTokenizerFast

