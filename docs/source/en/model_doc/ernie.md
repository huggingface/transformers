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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# ERNIE

[ERNIE1.0](https://arxiv.org/abs/1904.09223), [ERNIE2.0](https://ojs.aaai.org/index.php/AAAI/article/view/6428),
[ERNIE3.0](https://arxiv.org/abs/2107.02137), [ERNIE-Gram](https://arxiv.org/abs/2010.12148), [ERNIE-health](https://arxiv.org/abs/2110.07244) are a series of powerful models proposed by baidu, especially in Chinese tasks.

ERNIE (Enhanced Representation through kNowledge IntEgration) is designed to learn language representation enhanced by knowledge masking strategies, which includes entity-level masking and phrase-level masking.

Other ERNIE models released by baidu can be found at [Ernie 4.5](./ernie4_5), and [Ernie 4.5 MoE](./ernie4_5_moe).

> [!TIP]
> This model was contributed by [nghuyong](https://huggingface.co/nghuyong), and the official code can be found in [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) (in PaddlePaddle).
>
> Click on the ERNIE models in the right sidebar for more examples of how to apply ERNIE to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="nghuyong/ernie-3.0-xbase-zh"
)

pipeline("巴黎是[MASK]国的首都。")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "nghuyong/ernie-3.0-xbase-zh",
)
model = AutoModelForMaskedLM.from_pretrained(
    "nghuyong/ernie-3.0-xbase-zh",
    dtype=torch.float16,
    device_map="auto"
)
inputs = tokenizer("巴黎是[MASK]国的首都。", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "巴黎是[MASK]国的首都。" | transformers run --task fill-mask --model nghuyong/ernie-3.0-xbase-zh --device 0
```

</hfoption>
</hfoptions>

## Notes

Model variants are available in different sizes and languages.

|     Model Name      | Language |           Description           |
|:-------------------:|:--------:|:-------------------------------:|
|  ernie-1.0-base-zh  | Chinese  | Layer:12, Heads:12, Hidden:768  |
|  ernie-2.0-base-en  | English  | Layer:12, Heads:12, Hidden:768  |
| ernie-2.0-large-en  | English  | Layer:24, Heads:16, Hidden:1024 |
|  ernie-3.0-base-zh  | Chinese  | Layer:12, Heads:12, Hidden:768  |
| ernie-3.0-medium-zh | Chinese  |  Layer:6, Heads:12, Hidden:768  |
|  ernie-3.0-mini-zh  | Chinese  |  Layer:6, Heads:12, Hidden:384  |
| ernie-3.0-micro-zh  | Chinese  |  Layer:4, Heads:12, Hidden:384  |
|  ernie-3.0-nano-zh  | Chinese  |  Layer:4, Heads:12, Hidden:312  |
|   ernie-health-zh   | Chinese  | Layer:12, Heads:12, Hidden:768  |
|    ernie-gram-zh    | Chinese  | Layer:12, Heads:12, Hidden:768  |

## Resources

You can find all the supported models from huggingface's model hub: [huggingface.co/nghuyong](https://huggingface.co/nghuyong), and model details from paddle's official
repo: [PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html)
and [ERNIE's legacy branch](https://github.com/PaddlePaddle/ERNIE/tree/legacy/develop).

## ErnieConfig

[[autodoc]] ErnieConfig
    - all

## Ernie specific outputs

[[autodoc]] models.ernie.modeling_ernie.ErnieForPreTrainingOutput

## ErnieModel

[[autodoc]] ErnieModel
    - forward

## ErnieForPreTraining

[[autodoc]] ErnieForPreTraining
    - forward

## ErnieForCausalLM

[[autodoc]] ErnieForCausalLM
    - forward

## ErnieForMaskedLM

[[autodoc]] ErnieForMaskedLM
    - forward

## ErnieForNextSentencePrediction

[[autodoc]] ErnieForNextSentencePrediction
    - forward

## ErnieForSequenceClassification

[[autodoc]] ErnieForSequenceClassification
    - forward

## ErnieForMultipleChoice

[[autodoc]] ErnieForMultipleChoice
    - forward

## ErnieForTokenClassification

[[autodoc]] ErnieForTokenClassification
    - forward

## ErnieForQuestionAnswering

[[autodoc]] ErnieForQuestionAnswering
    - forward
