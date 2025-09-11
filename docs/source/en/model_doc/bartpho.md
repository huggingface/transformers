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
*This model was released on 2021-09-20 and added to Hugging Face Transformers on 2021-10-18.*

<div style="float: right;">
   <div class="flex flex-wrap space-x-1">
      <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
   </div>
</div>

# BARTpho

[BARTpho](https://huggingface.co/papers/2109.09701) is a large-scale Vietnamese sequence-to-sequence model. It offers a word-based and syllable-based version. This model is built on the [BART](./bart) large architecture with its denoising pretraining.

You can find all the original checkpoints under the [VinAI](https://huggingface.co/vinai/models?search=bartpho) organization.

> [!TIP]
> This model was contributed by [dqnguyen](https://huggingface.co/dqnguyen).
> Check out the right sidebar for examples of how to apply BARTpho to different language tasks.

The example below demonstrates how to summarize text with [`Pipeline`] or the [`AutoModel`] class.


<hfoptions id="usage">
<hfoption id="Pipeline">



```python
import torch
from transformers import pipeline

pipeline = pipeline(
   task="summarization",
   model="vinai/bartpho-word",
   dtype=torch.float16,
   device=0
)

text = """
Quang tổng hợp hay gọi tắt là quang hợp là quá trình thu nhận và chuyển hóa năng lượng ánh sáng Mặt trời của thực vật,
tảo và một số vi khuẩn để tạo ra hợp chất hữu cơ phục vụ bản thân cũng như làm nguồn thức ăn cho hầu hết các sinh vật
trên Trái Đất. Quang hợp trong thực vật thường liên quan đến chất tố diệp lục màu xanh lá cây và tạo ra oxy như một sản phẩm phụ
"""
pipeline(text)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "vinai/bartpho-word",
)
model = BartForConditionalGeneration.from_pretrained(
    "vinai/bartpho-word",
    dtype=torch.float16,
    device_map="auto",
)

text = """
Quang tổng hợp hay gọi tắt là quang hợp là quá trình thu nhận và chuyển hóa năng lượng ánh sáng Mặt trời của thực vật,
tảo và một số vi khuẩn để tạo ra hợp chất hữu cơ phục vụ bản thân cũng như làm nguồn thức ăn cho hầu hết các sinh vật
trên Trái Đất. Quang hợp trong thực vật thường liên quan đến chất tố diệp lục màu xanh lá cây và tạo ra oxy như một sản phẩm phụ
"""
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Quang tổng hợp hay gọi tắt là quang hợp là quá trình thu nhận và chuyển hóa năng lượng ánh sáng Mặt trời của thực vật,
tảo và một số vi khuẩn để tạo ra hợp chất hữu cơ phục vụ bản thân cũng như làm nguồn thức ăn cho hầu hết các sinh vật
trên Trái Đất. Quang hợp trong thực vật thường liên quan đến chất tố diệp lục màu xanh lá cây và tạo ra oxy như một sản phẩm phụ" | \
transformers run --task summarization --model vinai/bartpho-word --device 0
```

</hfoption>
</hfoptions>



## Notes

- BARTpho uses the large architecture of BART with an additional layer-normalization layer on top of the encoder and decoder. The BART-specific classes should be replaced with the mBART-specific classes.
- This implementation only handles tokenization through the `monolingual_vocab_file` file. This is a Vietnamese-specific subset of token types taken from that multilingual vocabulary. If you want to use this tokenizer for another language, replace the `monolingual_vocab_file` with one specialized for your target language.

## BartphoTokenizer

[[autodoc]] BartphoTokenizer
