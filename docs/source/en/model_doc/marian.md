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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
          <img alt="Model downloads" src="https://img.shields.io/huggingface/model-downloads/Helsinki-NLP/opus-mt-en-de?logo=huggingface" />
           <img alt="License" src="https://img.shields.io/github/license/huggingface/transformers?logo=open-source-initiative" />
           <img alt="Task Translation" src="https://img.shields.io/badge/task-translation-blue?logo=google-translate" />
           <img alt="Model size" src="https://img.shields.io/badge/model-size-298MB-green" />
    </div>
</div>

# MarianMT

[MarianMT: Fast Neural Machine Translation in C++](https://arxiv.org/abs/2001.08210)

## Overview

MarianMT is a machine translation model developed by the Microsoft Translator team and trained originally by Jörg Tiedemann using the Marian C++ library. MarianMT models are designed to be fast, efficient, and lightweight for translation tasks. Unlike very large general models, MarianMT provides compact, language-specific models that are small enough to run on CPUs or low-resource environments, making it ideal for production and offline usage.

All MarianMT models are Transformer encoder-decoder architectures with 6 layers each in both encoder and decoder, similar in design to BART but with important modifications for translation tasks:

- Static (sinusoidal) positional embeddings (`MarianConfig.static_position_embeddings=True`)
- No layer normalization on embeddings (`MarianConfig.normalize_embedding=False`)
- Starts decoding with `pad_token_id` instead of special `<s>` tokens as BART does

There are over **1,000 MarianMT models**, covering a wide variety of language pairs. Each model is around **298 MB** on disk.

You can find all the original MarianMT checkpoints under the [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) collection.

The MarianMT code and framework are open source and available on [Marian GitHub](https://github.com/marian-nmt/marian).

> [!TIP]
> Click on the MarianMT models in the right sidebar to see more examples of how to apply MarianMT to different translation tasks.

---

The example below demonstrates how to translate text using [`Pipeline`] or the [`AutoModelForSeq2SeqLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python

from transformers import pipeline

translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
result = translator("Hello, how are you?")
print(result)

```

</hfoption>

<hfoption id="AutoModel">

```python

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

</hfoption>
<hfoption id="transformers-cli">

Not supported for this model.

</hfoption>
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [dynamic quantization](https://docs.pytorch.org/docs/stable/quantization.html#dynamic-quantization) to only quantize the weights to INT8.

```python

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = quantized_model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Attention Mask Visualizer Support

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("Helsinki-NLP/opus-mt-en-de")
visualizer("Hello, how are you?")
```


## Supported Languages
All models follow the naming convention:
Helsinki-NLP/opus-mt-{src}-{tgt}, where src is the source language code and tgt is the target language code.

The list of supported languages and codes is available in each model card.

Some models are multilingual; for example, opus-mt-en-ROMANCE translates English to multiple Romance languages (French, Spanish, Portuguese, etc.).

Newer models use 3-character language codes, e.g., >>fra<< for French, >>por<< for Portuguese.

Older models use 2-character or region-specific codes like es_AR (Spanish from Argentina).

Example of translating English to multiple Romance languages:
```python
from transformers import MarianMTModel, MarianTokenizer

src_text = [
    ">>fra<< This is a sentence in English to translate to French.",
    ">>por<< This should go to Portuguese.",
    ">>spa<< And this to Spanish."
]

model_name = "Helsinki-NLP/opus-mt-en-roa"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

inputs = tokenizer(src_text, return_tensors="pt", padding=True)
outputs = model.generate(**inputs)
result = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
print(result)

```


## Notes

- MarianMT models are smaller than many other translation models, enabling faster inference, low memory usage, and suitability for CPU environments.

- Based on Transformer encoder-decoder architecture with 6 layers each.

- Originally trained with the Marian C++ framework for efficiency.

- Does not support the older OPUS models that require BPE preprocessing (80 models not supported).

- When using quantization, expect a small trade-off in accuracy for a significant gain in speed and memory.

- The modeling code is based on BartForConditionalGeneration with adjustments for translation.


## Resources

- **Marian Research Paper:** [Marian: Fast Neural Machine Translation in C++](https://arxiv.org/abs/2001.08210)  
- **MarianMT Model Collection:** [Helsinki-NLP on Hugging Face](https://huggingface.co/Helsinki-NLP)  
- **Marian Official Framework:** [Marian-NMT GitHub](https://github.com/marian-nmt/marian)  
- **Language Codes Reference:** [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)  
- **Translation Task Guide:** [Hugging Face Translation Guide](https://huggingface.co/tasks/translation)  
- **Quantization Overview:** [Transformers Quantization Docs](https://huggingface.co/docs/transformers/main/en/perf_optimization#model-quantization)  
- **Tokenizer Guide:** [Hugging Face Tokenizer Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer)  
- **Model Conversion Tool:** [convert_marian_to_pytorch.py (GitHub)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/marian/convert_marian_to_pytorch.py)  
- **Supported Language Pairs:** Refer to individual model cards under [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for supported languages.  




</jax>
</frameworkcontent>
