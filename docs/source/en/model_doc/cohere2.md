<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>


# Cohere2

[Cohere Command R7B](https://cohere.com/blog/command-r7b) is an open weights research release of a 7B billion parameter model developed by Cohere and Cohere For AI. It has advanced capabilities optimized for various use cases, including RAG, tool use, agentic capabilities and tasks requiring complex reasoning and multiple steps,. C4AI Command R7B is a multilingual model trained on 23 languages and has a context window of 128k.

You can find all the original Command-R checkpoints under the [Command Models](https://huggingface.co/collections/CohereForAI/command-models-67652b401665205e17b192ad) collection.


> [!TIP]
> Click on the Cohere models in the right sidebar for more examples of how to apply Cohere to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation", 
    model="CohereLabs/c4ai-command-r7b-12-2024",
    torch_dtype=torch.float16,
    device_map=0
)

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipeline(messages)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r7b-12-2024")
model = AutoModelForCausalLM.from_pretrained(
    "CohereForAI/c4ai-command-r7b-12-2024", torch_dtype=torch.float16, 
    device_map="auto"
)

# Format message with the command-r chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
)

output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

print(tokenizer.decode(output[0],skip_special_tokens=True))
```

</hfoption>
</hfoptions>


## Notes
- For quantized version of Cohere R7B, you can refer to this [collection](https://huggingface.co/models?other=base_model:quantized:CohereLabs/c4ai-command-r7b-12-2024).

## Cohere2Config

[[autodoc]] Cohere2Config

## Cohere2Model

[[autodoc]] Cohere2Model
    - forward


## Cohere2ForCausalLM

[[autodoc]] Cohere2ForCausalLM
    - forward


