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
*This model was released on 2022-06-25 and added to Hugging Face Transformers on 2022-12-12 and contributed by [AI-Sweden-Models](https://huggingface.co/AI-Sweden-Models).*

# GPT-Sw3

[GPT-Sw3](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf) is a 3.5 billion parameter autoregressive language model trained on a 100 GB Swedish corpus. The paper details its data collection and training process, along with challenges in evaluating performance. Quantitative evaluation using perplexity shows that GPT-SW3 performs competitively with other models of similar scale. A prompting study further demonstrates the model’s strong text generation abilities in Swedish.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="AI-Sweden-Models/gpt-sw3-356m", dtype="auto",)
pipeline("Växter skapar energi genom en process som kallas fotosyntes.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-356m")
model = AutoModelForCausalLM.from_pretrained("AI-Sweden-Models/gpt-sw3-356m", dtype="auto",)

inputs = tokenizer("Växter skapar energi genom en process som kallas fotosyntes.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GPTSw3Tokenizer

[[autodoc]] GPTSw3Tokenizer
    - save_vocabulary
