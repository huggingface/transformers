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
*This model was released on 2021-12-20 and added to Hugging Face Transformers on 2022-01-28 and contributed by [valhalla](https://huggingface.co/valhalla).*

# XGLM

[XGLM](https://huggingface.co/papers/2112.10668) trains multilingual autoregressive language models on a balanced corpus across diverse languages, enhancing few- and zero-shot learning capabilities. The largest model, with 7.5 billion parameters, achieves state-of-the-art results in multilingual commonsense reasoning and natural language inference, outperforming GPT-3 of similar size. It also excels in the FLORES-101 machine translation benchmark, surpassing GPT-3 in 171 out of 182 translation directions and the supervised baseline in 45 directions. The model demonstrates cross-lingual in-context learning but has limitations in surface form robustness and adaptation to non-cloze tasks. Additionally, it shows similar limitations as GPT-3 in social value tasks like hate speech detection across five languages.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="facebook/xglm-564M", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
model = AutoModelForCausalLM.from_pretrained("facebook/xglm-564M", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## XGLMConfig

[[autodoc]] XGLMConfig

## XGLMTokenizer

[[autodoc]] XGLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XGLMTokenizerFast

[[autodoc]] XGLMTokenizerFast

## XGLMModel

[[autodoc]] XGLMModel
    - forward

## XGLMForCausalLM

[[autodoc]] XGLMForCausalLM
    - forward

