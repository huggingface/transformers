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
*This model was released on 2021-04-18 and added to Hugging Face Transformers on 2021-11-03 and contributed by [nielsr](https://huggingface.co/nielsr).*

# LayoutXLM

[LayoutXLM](https://huggingface.co/papers/2104.08836) is a multimodal pre-trained model for multilingual document understanding, extending LayoutLMv2 to support 53 languages. It integrates text, layout, and image data to achieve state-of-the-art results in visually-rich document tasks. The model's performance is evaluated using XFUN, a multilingual form understanding benchmark dataset with samples in seven languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese). LayoutXLM significantly outperforms existing cross-lingual pre-trained models on this dataset.

<hfoptions id="usage">
<hfoption id="LayoutLMv2ForQuestionAnswering">

```py
import torch
from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering
from PIL import Image
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/layoutxlm-base")
model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutxlm-base", dtype="auto")

dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
image = dataset["test"][0]["image"]
question = "When is coffee break?"
encoding = processor(image, question, return_tensors="pt")

outputs = model(**encoding)
predicted_start_idx = outputs.start_logits.argmax(-1).item()
predicted_end_idx = outputs.end_logits.argmax(-1).item()
predicted_start_idx, predicted_end_idx

predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
print(processor.tokenizer.decode(predicted_answer_tokens))
```

</hfoption>
</hfoptions>

## LayoutXLMTokenizer

[[autodoc]] LayoutXLMTokenizer
    - __call__
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LayoutXLMTokenizerFast

[[autodoc]] LayoutXLMTokenizerFast
    - __call__

## LayoutXLMProcessor

[[autodoc]] LayoutXLMProcessor
    - __call__

