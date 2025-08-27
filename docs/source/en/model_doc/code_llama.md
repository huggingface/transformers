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
*This model was released on 2023-08-24 and added to Hugging Face Transformers on 2023-08-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# CodeLlama

[Code Llama](https://huggingface.co/papers/2308.12950) is a specialized family of large language models based on [Llama 2](./llama2) for coding tasks.  It comes in different flavors - general code, Python-specific, and instruction-following variant - all available in 7B, 13B, 34B, and 70B parameters. Code Llama models can generate, explain, and even fill in missing parts of your code (called "infilling"). It can also handle very long contexts with stable generation up to 100k tokens, even though it was trained on sequences of 16K tokens.

You can find all the original Code Llama checkpoints under the [Code Llama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933) collection.

> [!TIP]
> Click on the Code Llama models in the right sidebar for more examples of how to apply Code Llama to different coding tasks.

The example below demonstrates how to generate code with [`Pipeline`], or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/CodeLlama-7b-hf",
    dtype=torch.float16,
    device_map=0
)

# basic code generation
result = pipe("# Function to calculate the factorial of a number\ndef factorial(n):", max_new_tokens=256)
print(result[0]['generated_text'])

# infilling
infill_result = pipe("def remove_non_ascii(s: str) -> str:\n    \"\"\" <FILL_ME>\n    return result", max_new_tokens=200)
print(infill_result[0]['generated_text'])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/CodeLlama-7b-hf",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# basic code generation
prompt = "# Function to calculate the factorial of a number\ndef factorial(n):"
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **input_ids,
    max_new_tokens=256,
    cache_implementation="static"
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# infilling
infill_prompt = "def remove_non_ascii(s: str) -> str:\n    \"\"\" <FILL_ME>\n    return result"
input_ids = tokenizer(infill_prompt, return_tensors="pt").to(model.device)

filled_output = model.generate(**input_ids, max_new_tokens=200)
filled_text = tokenizer.decode(filled_output[0], skip_special_tokens=True)
print(filled_text)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "# Function to calculate the factorial of a number\ndef factorial(n):" | transformers run --task text-generation --model meta-llama/CodeLlama-7b-hf --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
# pip install bitsandbytes
import torch
from transformers import AutoModelForCausalLM, CodeLlamaTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-34b-hf")
model = AutoModelForCausalLM.from_pretrained(
   "meta-llama/CodeLlama-34b-hf",
   dtype=torch.bfloat16,
   device_map="auto",
   quantization_config=bnb_config
)

prompt = "# Write a Python function to check if a string is a palindrome\ndef is_palindrome(s):"
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=200, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("meta-llama/CodeLlama-7b-hf")
visualizer("""def func(a, b):
  return a + b""")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/codellama-attn-mask.png"/>
</div>

## Notes

- Infilling is only available in the 7B and 13B base models, and not in the Python, Instruct, 34B, or 70B models.
- Use the `<FILL_ME>` token where you want your input to be filled. The tokenizer splits this token to create a formatted input string that follows the [original training pattern](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402). This is more robust than preparing the pattern yourself.
    ```py
    from transformers import LlamaForCausalLM, CodeLlamaTokenizer

    tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
    PROMPT = '''def remove_non_ascii(s: str) -> str:
        """ <FILL_ME>
        return result
    '''
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids, max_new_tokens=128)

    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print(PROMPT.replace("<FILL_ME>", filling))
    ```
- Use `bfloat16` for further training or fine-tuning and `float16` for inference.
- The `BOS` character is not used for infilling when encoding the prefix or suffix, but only at the beginning of each prompt.
- The tokenizer is a byte-pair encoding model based on [SentencePiece](https://github.com/google/sentencepiece). During decoding, if the first token is the start of the word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.

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
