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
*This model was released on 2020-10-20 and added to Hugging Face Transformers on 2023-06-20 and contributed by [stefan-it](https://huggingface.co/stefan-it).*

> [!WARNING]
> This model is in maintenance mode only, we do not accept any new PRs changing its code.
> If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0. You can do so by running the following command: pip install -U transformers==4.30.0.

# BORT

[BORT](https://huggingface.co/papers/2010.10499) extracts an optimal subset of architectural parameters from BERT, significantly reducing its size to 5.5% of BERT-large's effective size and 16% of its net size. BORT can be pretrained in 288 GPU hours, which is 1.2% of the time required for RoBERTa-large and 33% of BERT-large. It is 7.9x faster on a CPU and outperforms other compressed and some non-compressed variants, achieving performance improvements of 0.3% to 31% on various NLU benchmarks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="amazon/bort", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("amazon/bort", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("amazon/bort")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>