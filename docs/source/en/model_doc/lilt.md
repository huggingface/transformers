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
*This model was released on 2022-02-28 and added to Hugging Face Transformers on 2022-10-12 and contributed by [nielsr](https://huggingface.co/nielsr).*

# LiLT

[LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding](https://huggingface.co/papers/2202.13669) combines a pre-trained RoBERTa text encoder with a lightweight Layout Transformer to enable document understanding across multiple languages. LiLT can be pre-trained on structured documents in one language and fine-tuned on others using off-the-shelf monolingual or multilingual pre-trained textual models. Experiments across eight languages demonstrate competitive or superior performance on various benchmarks, highlighting its language-independent benefits in document layout structure pre-training.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
model = AutoModelForQuestionAnswering.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base", dtype="auto")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
words = example["tokens"]
boxes = example["bboxes"]

encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

outputs = model(**encoding)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))
```

</hfoption>
</hfoptions>

## Usage tips

- Combine the Language-Independent Layout Transformer with a new RoBERTa checkpoint from the Hub using this [guide](https://github.com/jpwang/lilt). The script stores `config.json` and `pytorch_model.bin` files locally.
- Use the token vocabulary that corresponds to the RoBERTa checkpoint you combined with the Layout Transformer when preparing data.
- The `lilt-roberta-en-base` model uses the same vocabulary as LayoutLMv3. Use [`LayoutLMv3TokenizerFast`] to prepare data for this model.
- Use [`LayoutXLMTokenizerFast`] for models that share vocabulary with LayoutXLM.

## LiltConfig

[[autodoc]] LiltConfig

## LiltModel

[[autodoc]] LiltModel
    - forward

## LiltForSequenceClassification

[[autodoc]] LiltForSequenceClassification
    - forward

## LiltForTokenClassification

[[autodoc]] LiltForTokenClassification
    - forward

## LiltForQuestionAnswering

[[autodoc]] LiltForQuestionAnswering
    - forward

