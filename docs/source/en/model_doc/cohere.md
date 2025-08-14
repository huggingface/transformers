<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>


# Cohere

Cohere Command-R is a 35B parameter multilingual large language model designed for long context tasks like retrieval-augmented generation (RAG) and calling external APIs and tools. The model is specifically trained for grounded generation and supports both single-step and multi-step tool use. It supports a context length of 128K tokens.

You can find all the original Command-R checkpoints under the [Command Models](https://huggingface.co/collections/CohereForAI/command-models-67652b401665205e17b192ad) collection.


> [!TIP]
> Click on the Cohere models in the right sidebar for more examples of how to apply Cohere to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="CohereForAI/c4ai-command-r-v01",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-v01", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")

# format message with the Command-R chat template
messages = [{"role": "user", "content": "How do plants make energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    cache_implementation="static",
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
# pip install -U flash-attn --no-build-isolation
transformers chat CohereForAI/c4ai-command-r-v01 --dtype auto --attn_implementation flash_attention_2
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.

```python
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-v01", dtype=torch.float16, device_map="auto", quantization_config=bnb_config, attn_implementation="sdpa")

# format message with the Command-R chat template
messages = [{"role": "user", "content": "How do plants make energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    cache_implementation="static",
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("CohereForAI/c4ai-command-r-v01")
visualizer("Plants create energy through a process known as")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/cohere-attn-mask.png"/>
</div>


## Notes
- Don’t use the dtype parameter in [`~AutoModel.from_pretrained`] if you’re using FlashAttention-2 because it only supports fp16 or bf16. You should use [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html), set fp16 or bf16 to True if using [`Trainer`], or use [torch.autocast](https://pytorch.org/docs/stable/amp.html#torch.autocast).

## CohereConfig

[[autodoc]] CohereConfig

## CohereTokenizerFast

[[autodoc]] CohereTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## CohereModel

[[autodoc]] CohereModel
    - forward


## CohereForCausalLM

[[autodoc]] CohereForCausalLM
    - forward
