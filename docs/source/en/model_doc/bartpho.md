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

# BARTpho

<div style="float: right;">
<div class="flex flex-wrap space-x-1">
<img alt="Hugging Face Model" src="https://img.shields.io/badge/Model%20Hub-BARTpho-blue">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green">
<img alt="Language" src="https://img.shields.io/badge/Language-Vietnamese-orange">
</div>
</div>

# BARTpho

[BARTpho](https://arxiv.org/abs/2109.09701) is the first large-scale, monolingual sequence-to-sequence model pre-trained exclusively for Vietnamese, developed by [VinAI Research](https://huggingface.co/vinai).
It’s based on the **BART** denoising autoencoder architecture, with adaptations from **mBART**, and comes in two variants — **word** and **syllable** — to handle the unique way Vietnamese uses whitespace.
Think of it like a supercharged summarizer and text generator that really “gets” Vietnamese — both at the word and syllable level.

You can find all official checkpoints in the [BARTpho collection](https://huggingface.co/collections/vinai/bartpho-66f8a74775316eaa77d59969).

> \[!TIP]
> This model was contributed by [VinAI Research](https://huggingface.co/vinai).
> Check out the `bartpho-word` and `bartpho-syllable` variants in the right sidebar for examples of summarization, punctuation restoration, and capitalization restoration.

The example below demonstrates how to run summarization with \[`pipeline`] or load the model via \[`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Ensure tone normalization + word segmentation for bartpho-word
summarizer = pipeline(
"summarization",
model="vinai/bartpho-word",
tokenizer="vinai/bartpho-word"
)

ARTICLE = "BARTpho là mô_hình Xử_lý ngôn_ngữ tự_nhiên đơn_ngữ cho tiếng Việt..."
summary = summarizer(ARTICLE, max_length=25, min_length=10, do_sample=False)
print(summary[0]['summary_text'])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_checkpoint = "vinai/bartpho-word"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)

TXT = "Chúng_tôi là những nghiên_cứu_viên."
inputs = tokenizer(TXT, return_tensors="pt")
features = model(**inputs)
print("Shape:", features.last_hidden_state.shape)
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli download vinai/bartpho-word
```

</hfoption>
</hfoptions>

Quantization reduces the memory footprint of large models by storing weights in lower precision. See the [BitsAndBytes Quantization guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes) for details.

Example: 4-bit quantization with `bitsandbytes`:

```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
model = AutoModel.from_pretrained("vinai/bartpho-word", quantization_config=quant_config, device_map="auto")
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/attention_visualizer.py) to see what tokens the model attends to.

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("vinai/bartpho-word")
visualizer("Chúng_tôi là những nghiên_cứu_viên.")
```


## Notes

* **Preprocessing is non-negotiable**:

* All inputs must undergo Vietnamese tone normalization.
* For `bartpho-word` variants, text must also be segmented with [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP).
* The model is trained on a 20GB corpus (\~145M sentences), so domain-specific performance may vary.
* `bartpho-word` consistently outperforms `bartpho-syllable` on Vietnamese generative tasks.

```python
# Example: Masked language modeling with bartpho-syllable
from transformers import MBartForConditionalGeneration, AutoTokenizer
import torch

model = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

TXT = "Chúng tôi là <mask> nghiên cứu viên."
input_ids = tokenizer(TXT, return_tensors="pt")["input_ids"]
logits = model(input_ids).logits
masked_index = (input_ids == tokenizer.mask_token_id).nonzero().item()
predicted_ids = torch.topk(logits[0, masked_index], 5).indices
print(tokenizer.decode(predicted_ids, skip_special_tokens=True))
```

## Resources

* [BARTpho GitHub](https://github.com/VinAIResearch/BARTpho)
* [BARTpho Paper (INTERSPEECH 2022)](https://arxiv.org/abs/2109.09701)
* [VinAI Hugging Face Organization](https://huggingface.co/vinai)
* [BitsAndBytes Quantization Guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
