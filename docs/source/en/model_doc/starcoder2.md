<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-02-29 and added to Hugging Face Transformers on 2024-02-28.*

# Starcoder2

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
  </div>
</div>

[StarCoder2](https://huggingface.co/papers/2402.19173) is an open family of Large Language Models for code, trained on **The Stack v2** (600+ programming languages, 3.3–4.3T tokens). It comes in 3B, 7B, and 15B variants, with the 15B flagship model supporting a **16,384-token context window** and trained using **Fill-in-the-Middle (FIM)**. StarCoder2 is optimized for code generation and reasoning, and matches or outperforms much larger models on many benchmarks.

You can find all original [StarCoder2 checkpoints](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a) under the BigCode collection.

> [!TIP]
> This model was contributed by [BigCode](https://huggingface.co/bigcode).
> Click on the StarCoder2 variants in the right sidebar for task-specific examples like text generation and code completion.

The example below demonstrates how to generate Python code with [`pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="bigcode/starcoder2-7b",
    torch_dtype="auto",
    device_map="auto",
)

prompt = "# Write a function to check if a number is prime\n"
out = pipe(prompt, max_new_tokens=64, do_sample=True, temperature=0.2)
print(out[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoder2-7b",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
    device_map="auto",
)

prompt = "# Python function to reverse a string\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)
gen = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.3)
print(tok.decode(gen[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers chat bigcode/starcoder2-7b
```

</hfoption>
</hfoptions>

## Quantization

You can load StarCoder2 in 8-bit or 4-bit precision using [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes), which reduces memory usage significantly with minimal performance loss.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True

checkpoint = "bigcode/starcoder2-7b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=quant_config,
    device_map="auto"
)

inputs = tokenizer("def print_hello_world():", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

StarCoder2 has been trained on source code from 17 programming languages. While it can generate high-quality code snippets, the outputs are not guaranteed to be correct, safe, or efficient. Generated code may contain bugs, security vulnerabilities, or license-sensitive snippets. See the [paper](https://huggingface.co/papers/2402.19173) for a detailed analysis of limitations.

## Attribution

StarCoder2 may generate code seen during training. Users are responsible for checking if generated code requires attribution or license compliance. You can search the [training dataset](https://huggingface.co/spaces/bigcode/the-stack-dedup) to identify potential sources and apply the appropriate licenses.

## License

StarCoder2 models are released under the **BigCode OpenRAIL‑M v1 License Agreement**, which enables free and responsible use, modification, and redistribution. See the full terms [here](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).

## Citation

```bibtex
@misc{lozhkov2024starcoder,
  title     = {StarCoder 2 and The Stack v2: The Next Generation},
  author    = {Anton Lozhkov and Raymond Li and Loubna Ben Allal and Federico Cassano and Joel Lamy-Poirier and Nouamane Tazi and Ao Tang and Dmytro Pykhtar and Jiawei Liu and Yuxiang Wei and Tianyang Liu and Max Tian and Denis Kocetkov and Arthur Zucker and Younes Belkada and Zijian Wang and Qian Liu and Dmitry Abulkhanov and Indraneil Paul and Zhuang Li and Wen-Ding Li and Megan Risdal and Jia Li and Jian Zhu and Terry Yue Zhuo and Evgenii Zheltonozhskii and Nii Osae Osae Dade and Wenhao Yu and Lucas Krauß and Naman Jain and Yixuan Su and Xuanli He and Manan Dey and Edoardo Abati and Yekun Chai and Niklas Muennighoff and Xiangru Tang and Muhtasham Oblokulov and Christopher Akiki and Marc Marone and Chenghao Mou and Mayank Mishra and Alex Gu and Binyuan Hui and Tri Dao and Armel Zebaze and Olivier Dehaene and Nicolas Patry and Canwen Xu and Julian McAuley and Han Hu and Torsten Scholak and Sebastien Paquet and Jennifer Robinson and Carolyn Jane Anderson and Nicolas Chapados and Mostofa Patwary and Nima Tajbakhsh and Yacine Jernite and Carlos Muñoz Ferrandis and Lingming Zhang and Sean Hughes and Thomas Wolf and Arjun Guha and Leandro von Werra and Harm de Vries},
  year      = {2024},
  eprint    = {2402.19173},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE}
}
```

## Starcoder2Config

[[autodoc]] Starcoder2Config

## Starcoder2Model

[[autodoc]] Starcoder2Model
    - forward

## Starcoder2ForCausalLM

[[autodoc]] Starcoder2ForCausalLM
    - forward

## Starcoder2ForSequenceClassification

[[autodoc]] Starcoder2ForSequenceClassification
    - forward

## Starcoder2ForTokenClassification

[[autodoc]] Starcoder2ForTokenClassification
    - forward

## Training Details

- **Architecture**: Transformer decoder with grouped-query attention, sliding window attention, and Fill-in-the-Middle (FIM) objective
- **Pretraining Tokens**: 3.5T+
- **Precision**: bfloat16
- **Steps**: 1 million
- **Compute**: 432 × H100 GPUs
- **Framework**: nanotron (PyTorch backend)