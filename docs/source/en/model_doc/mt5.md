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
*This model was released on 2020-10-22 and added to Hugging Face Transformers on 2020-11-17 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# mT5

[mT5](https://huggingface.co/papers/2010.11934) leverages a unified text-to-text format and scale to achieve state-of-the-art results across various multilingual NLP tasks. It was pre-trained on a Common Crawl-based dataset covering 101 languages. The model demonstrates superior performance on multilingual benchmarks and includes a technique to mitigate accidental translation in zero-shot scenarios. mT5 requires fine-tuning for specific tasks and does not benefit from task prefixes during single-task fine-tuning, unlike the original T5. Google provides several variants of mT5, ranging from small to XXL sizes.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="csebuetnlp/mT5_multilingual_XLSum", dtype="auto")
pipeline("""
Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

text="""
Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfopton>
</hfoptions>

## MT5Config

[[autodoc]] MT5Config

## MT5Tokenizer

[[autodoc]] MT5Tokenizer

See [`T5Tokenizer`] for all details.

## MT5TokenizerFast

[[autodoc]] MT5TokenizerFast

See [`T5TokenizerFast`] for all details.

## MT5Model

[[autodoc]] MT5Model

## MT5ForConditionalGeneration

[[autodoc]] MT5ForConditionalGeneration

## MT5EncoderModel

[[autodoc]] MT5EncoderModel

## MT5ForSequenceClassification

[[autodoc]] MT5ForSequenceClassification

## MT5ForTokenClassification

[[autodoc]] MT5ForTokenClassification

## MT5ForQuestionAnswering

[[autodoc]] MT5ForQuestionAnswering

