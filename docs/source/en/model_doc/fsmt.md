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
*This model was released on 2019-07-15 and added to Hugging Face Transformers on 2020-11-16 and contributed by [stas](https://huggingface.co/stas).*

# FSMT
[FSMT](https://huggingface.co/papers/1907.06616) models participated in the WMT19 shared news translation task for English <-> German and English <-> Russian. The systems are large BPE-based transformer models trained with Fairseq, utilizing sampled back-translations. This year, experiments included various bitext data filtering schemes and the addition of filtered back-translated data. Models were ensembled and fine-tuned on domain-specific data, with decoding enhanced by noisy channel model reranking. The submissions achieved top rankings in all four directions, with the En->De system outperforming other systems and human translations, improving by 4.5 BLEU points from the WMT'18 submission.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="facebook/wmt19-en-de", dtype="auto")
pipeline("Plants generate energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-de")

inputs = tokenizer("Plants generate energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## FSMTConfig

[[autodoc]] FSMTConfig

## FSMTTokenizer

[[autodoc]] FSMTTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FSMTModel

[[autodoc]] FSMTModel
    - forward

## FSMTForConditionalGeneration

[[autodoc]] FSMTForConditionalGeneration
    - forward
