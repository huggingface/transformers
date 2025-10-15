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
*This model was released on 2024-04-18 and added to Hugging Face Transformers on 2024-04-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Llama 3

[Llama 3](https://huggingface.co/papers/2407.21783) is a new family of foundation models designed for multilinguality, coding, reasoning, and tool use. The largest model is a 405B-parameter dense Transformer with a context window of 128K tokens, achieving performance on par with GPT-4 across a wide range of tasks. The release includes both pre-trained and post-trained models, as well as Llama Guard 3 for safety. Preliminary experiments also extend Llama 3 with image, video, and speech capabilities using a compositional approach, showing competitive results though these multimodal models are not yet fully released.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="meta-llama/Llama-3.1-8B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- LLaMA 3 models were trained using `bfloat16`, but original inference uses `float16`. Hub checkpoints use `dtype='float16'`. The [`AutoModel`] API casts checkpoints from `torch.float32` to `torch.float16`.
- Online weight dtype matters only when using `dtype="auto"`. The model downloads first (using checkpoint dtype), then casts to torch's default dtype (`torch.float32`). Specify your desired dtype or it defaults to `torch.float32`.
- Don't train in `float16`. It produces NaN values. Train in `bfloat16` instead.
- The tokenizer is a BPE model based on tiktoken (vs SentencePiece for LLaMA 2). It ignores BPE merge rules when an input token is part of the vocab. If "hugging" is in the vocab, it returns as one token instead of splitting into `["hug","ging"]`.
- The original model uses `pad_id = -1` (no padding token). Add a padding token with `tokenizer.add_special_tokens({"pad_token":"<pad>"})` and resize token embeddings. Set `model.config.pad_token_id`. Initialize `embed_tokens` with `padding_idx` to ensure padding tokens output zeros.
- Convert original checkpoints using the conversion script. The script requires enough CPU RAM to host the whole model in `float16` precision. For the 75B model, you need 145GB of RAM.
- When using Flash Attention 2 via `attn_implementation="flash_attention_2"`, don't pass `dtype` to [`~AutoModel.from_pretrained`]. Use [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) training. With [`Trainer`], set `fp16` or `bf16` to `True`. Otherwise, use [torch.autocast](https://pytorch.org/docs/stable/amp.html#torch.autocast). Flash Attention only supports `fp16` and `bf16` data types.