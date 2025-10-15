<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-04-27 and added to Hugging Face Transformers on 2023-08-25 and contributed by [ArthurZ](https://huggingface.co/ArthurZ).*

# CodeLlama

[CodeLlama](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) is a family of large language models for code, built on Llama 2, offering state-of-the-art performance among open models. It includes foundation models, Python specializations, and instruction-following models in 7B, 13B, and 34B parameter sizes. These models support infilling, handle large input contexts, and perform zero-shot instruction following for programming tasks. Trained on sequences of 16k tokens, they show improvements with inputs up to 100k tokens. The 7B and 13B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama achieves top scores on HumanEval and MBPP benchmarks, with Code Llama - Python 7B outperforming Llama 2 70B on these tasks. All models outperform other publicly available models on MultiPL-E. Code Llama is released under a permissive license for both research and commercial use.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="meta-llama/CodeLlama-7b-hf", dtype="auto")
pipeline("def fibonacci(n):")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")

inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Infilling works only in 7B and 13B base models. It doesn't work in Python, Instruct, 34B, or 70B models.
- Use the `<FILL_ME>` token where you want input filled. The tokenizer splits this token to create a formatted input string that follows the original training pattern. This beats preparing the pattern yourself.
- Use `bfloat16` for training or fine-tuning and `float16` for inference.
- The `BOS` character isn't used for infilling when encoding the prefix or suffix. It only appears at the beginning of each prompt.
- The tokenizer is a byte-pair encoding model based on SentencePiece. During decoding, if the first token starts a word (like "Banana"), the tokenizer doesn't prepend the prefix space.

## CodeLlamaTokenizer

[[autodoc]] CodeLlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeLlamaTokenizerFast

[[autodoc]] CodeLlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

