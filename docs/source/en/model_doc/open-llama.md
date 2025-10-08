<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2023-06-20 and contributed by [s-JoL](https://huggingface.co/s-JoL).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.31.0. You can do so by running the following command: pip install -U transformers==4.31.0.

# Open-Llama

[Open-Llama](https://github.com/s-JoL/Open-Llama) is an open-source framework for training large language models, covering the full pipeline from dataset preparation and tokenization to pre-training, instruction tuning, LoRA fine-tuning, and RLHF. It supports HuggingFace Transformers and achieves high performance, reaching 89% of GPT-3.5 on Chinese benchmarks, with a training speed of 3620 tokens/s, surpassing the original LLaMA. The project provides both pre-trained and instruction-tuned checkpoints, trained on 330 billion tokens with a global batch size of 4 million, and includes multi-turn dialogue, programming, and mathematical abilities. Evaluation follows FastChat methods, and the codebase is designed for high-performance experimentation and deployment.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openlm-research/open_llama_7b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## OpenLlamaConfig

[[autodoc]] OpenLlamaConfig

## OpenLlamaModel

[[autodoc]] OpenLlamaModel
    - forward

## OpenLlamaForCausalLM

[[autodoc]] OpenLlamaForCausalLM
    - forward

## OpenLlamaForSequenceClassification

[[autodoc]] OpenLlamaForSequenceClassification
    - forward
