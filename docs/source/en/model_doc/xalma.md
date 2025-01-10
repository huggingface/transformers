<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# X-ALMA

## Overview

[X-ALMA](https://arxiv.org/pdf/2410.03115) is a high-quality LLM-based translation model that supports 50 languages: en,da,nl,de,is,no,sv,af,ca,ro,gl,it,pt,es,bg,mk,sr,uk,ru,id,ms,th,vi,mg,fr,hu,el,cs,pl,lt,lv,ka,zh,ja,ko,fi,et,gu,hi,mr,ne,ur,az,kk,ky,tr,uz,ar,he,fa, ensuring their high-performance in translation, regardless of their resource level.

X-ALMA builds upon [ALMA-R](https://arxiv.org/pdf/2401.08417) and [ALMA](https://arxiv.org/abs/2309.11674) by expanding support from 6 to 50 languages. It utilizes a plug-and-play architecture with language-specific modules, complemented by a carefully designed training recipe. This release includes the **the complete X-ALMA model that contains the [X-ALMA pre-trained base model](https://huggingface.co/haoranxu/X-ALMA-13B-Pretrain) and all its language-specific modules**. Please find more details and other uses of X-ALMA [here](https://github.com/fe1ixxu/ALMA).


## Usage tips

Here, we show an example of translating "我爱机器翻译。" into English (X-ALMA should also able to do multilingual open-ended QA).

```
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("haoranxu/X-ALMA", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("haoranxu/X-ALMA", padding_side='left')

# Add the source sentence into the prompt template
prompt="Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"

# X-ALMA needs chat template but ALMA and ALMA-R don't need it.
chat_style_prompt = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

# Translation
with torch.no_grad():
    # Add `lang="zh"`: specify the language to instruct the model on which group to use for the third loading method during generation.
    generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9, lang="zh")
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)
```

There are two other recommended ways to load X-ALMA in a language-specific and memory-efficient manner:

*Alternative method 1*: loading the merged model where the language-specific module has been merged into the base model
```
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel

GROUP2LANG = {
1: ["da", "nl", "de", "is", "no", "sv", "af"],
2: ["ca", "ro", "gl", "it", "pt", "es"],
3: ["bg", "mk", "sr", "uk", "ru"],
4: ["id", "ms", "th", "vi", "mg", "fr"],
5: ["hu", "el", "cs", "pl", "lt", "lv"],
6: ["ka", "zh", "ja", "ko", "fi", "et"],
7: ["gu", "hi", "mr", "ne", "ur"],
8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
}
LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}
group_id = LANG2GROUP["zh"]

model = AutoModelForCausalLM.from_pretrained(f"haoranxu/X-ALMA-13B-Group{group_id}", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side='left')

# Add the source sentence into the prompt template
prompt="Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"

# X-ALMA needs chat template but ALMA and ALMA-R don't need it.
chat_style_prompt = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

# Translation
with torch.no_grad():
  generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
  outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)
```

*Alternative method 2*: loading the base model and language-specific module
```
model = AutoModelForCausalLM.from_pretrained("haoranxu/X-ALMA-13B-Pretrain", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, f"haoranxu/X-ALMA-13B-Group{group_id}")
tokenizer = AutoTokenizer.from_pretrained(f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side='left')
```

## XALMAConfig

[[autodoc]] XALMAConfig

## XALMAModel

[[autodoc]] XALMAModel
    - forward

## XALMAForCausalLM

[[autodoc]] XALMAForCausalLM
    - forward

## XALMAForSequenceClassification

[[autodoc]] XALMAForSequenceClassification
    - forward

## XALMAForQuestionAnswering

[[autodoc]] XALMAForQuestionAnswering
    - forward

## XALMAForTokenClassification

[[autodoc]] XALMAForTokenClassification
    - forward
