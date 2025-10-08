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
*This model was released on 2020-04-28 and added to Hugging Face Transformers on 2020-11-16 and contributed by [sshleifer](https://huggingface.co/sshleifer).*

# Blenderbot

[Blender](https://huggingface.co/papers/2004.13637) focuses on building open-domain chatbots by emphasizing the importance of various conversational skills beyond just scaling model parameters and data size. The model variants include 90M, 2.7B, and 9.4B parameters, demonstrating that with the right training data and generation strategies, large-scale models can learn to provide engaging talking points, listen, display knowledge, empathy, and personality, while maintaining a consistent persona. Human evaluations indicate that the best models outperform existing approaches in terms of engagingness and humanness in multi-turn dialogues. The paper also analyzes failure cases to highlight the limitations of the work.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="facebook/blenderbot-400M-distill", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## BlenderbotConfig

[[autodoc]] BlenderbotConfig

## BlenderbotTokenizer

[[autodoc]] BlenderbotTokenizer
    - build_inputs_with_special_tokens

## BlenderbotTokenizerFast

[[autodoc]] BlenderbotTokenizerFast
    - build_inputs_with_special_tokens

## BlenderbotModel

See [`~transformers.BartModel`] for arguments to *forward* and *generate*

[[autodoc]] BlenderbotModel
    - forward

## BlenderbotForConditionalGeneration

See [`~transformers.BartForConditionalGeneration`] for arguments to *forward* and *generate*

[[autodoc]] BlenderbotForConditionalGeneration
    - forward

## BlenderbotForCausalLM

[[autodoc]] BlenderbotForCausalLM
    - forward

