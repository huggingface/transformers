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
*This model was released on 2020-04-10 and added to Hugging Face Transformers on 2021-01-05 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# LED

[LED](https://huggingface.co/papers/2004.05150) introduces an attention mechanism that scales linearly with sequence length, enabling the processing of very long documents. This mechanism combines local windowed attention with task-specific global attention, making it a drop-in replacement for standard self-attention. The model achieves state-of-the-art results in character-level language modeling on text8 and enwik8. Pretrained Longformer outperforms RoBERTa on long document tasks, setting new benchmarks on WikiHop and TriviaQA. Additionally, Longformer-Encoder-Decoder (LED) is introduced for generative sequence-to-sequence tasks, demonstrating effectiveness on the arXiv summarization dataset.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="summarization", model="allenai/led-base-16384", dtype="auto")
pipeline("Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems. These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

text="""
Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
"""
global_attention_mask = torch.zeros_like(input_ids.input_ids)
global_attention_mask[:, 0] = 1

output = model.generate(**input_ids, global_attention_mask=global_attention_mask)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## LEDConfig

[[autodoc]] LEDConfig

## LEDTokenizer

[[autodoc]] LEDTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LEDTokenizerFast

[[autodoc]] LEDTokenizerFast

## LED specific outputs

[[autodoc]] models.led.modeling_led.LEDEncoderBaseModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqModelOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqLMOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput

[[autodoc]] models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput

## LEDModel

[[autodoc]] LEDModel
    - forward

## LEDForConditionalGeneration

[[autodoc]] LEDForConditionalGeneration
    - forward

## LEDForSequenceClassification

[[autodoc]] LEDForSequenceClassification
    - forward

## LEDForQuestionAnswering

[[autodoc]] LEDForQuestionAnswering
    - forward

