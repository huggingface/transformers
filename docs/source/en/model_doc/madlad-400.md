<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-09-09 and added to Hugging Face Transformers on 2023-11-28 and contributed by [jbochi](https://huggingface.co/jbochi).*

# MADLAD-400

[MADLAD-400](https://huggingface.co/papers/MADLAD-400%3A%20A%20Multilingual%20And%20Document-Level%20Large%20Audited%20Dataset) is a manually audited, 3T token monolingual dataset spanning 419 languages. It was used to train a 10.7B-parameter multilingual machine translation model on 250 billion tokens covering over 450 languages, demonstrating competitiveness with larger models across various domains. Additionally, an 8B-parameter language model was trained and evaluated for few-shot translation. The models are available for use without fine-tuning and support many low-resource languages.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="translation_en_to_fr", model="google/madlad400-3b-mt", dtype="auto")
pipeline("<2fr> Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")

inputs = tokenizer("<2fr> Plants create energy through a process known as photosynthesis.", return_tensors="pt")
generated_tokens = model.generate(**inputs)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
```

</hfoption>
</hfoptions>