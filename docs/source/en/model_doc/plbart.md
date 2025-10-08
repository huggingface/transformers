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
*This model was released on 2021-03-10 and added to Hugging Face Transformers on 2022-02-18 and contributed by [gchhablani](https://huggingface.co/gchhablani).*

# PLBart

[PLBart](https://huggingface.co/papers/2103.06333) is a sequence-to-sequence model designed for both code and natural language tasks, including code summarization, generation, and translation. It is pre-trained on large datasets of Java and Python functions paired with natural language using denoising autoencoding. The model outperforms or matches state-of-the-art performance across multiple programming languages and tasks, including program repair, clone detection, and vulnerability detection. Analysis shows that PLBART captures key aspects of code syntax, style, and logical flow, enabling strong performance even with limited annotated data.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
pipeline = pipeline(task="text2text-generation", model="uclanlp/plbart-python-en_XX", dtype="auto", tokenizer=tokenizer)
pipeline("def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

AutoTokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
model = AutoModelForSeq2SeqLM.from_pretrained("uclanlp/plbart-python-en_XX", dtype="auto")

inputs = tokenizer("def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])", return_tensors="pt")
outputs = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## PLBartConfig

[[autodoc]] PLBartConfig

## PLBartTokenizer

[[autodoc]] PLBartTokenizer
    - build_inputs_with_special_tokens

## PLBartModel

[[autodoc]] PLBartModel
    - forward

## PLBartForConditionalGeneration

[[autodoc]] PLBartForConditionalGeneration
    - forward

## PLBartForSequenceClassification

[[autodoc]] PLBartForSequenceClassification
    - forward

## PLBartForCausalLM

[[autodoc]] PLBartForCausalLM
    - forward