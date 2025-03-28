<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# Cohere

The **Cohere Command-R** model was proposed in the blog post: [Command-R: Retrieval Augmented Generation at Production Scale](https://cohere.com/blog/command-r) by the Cohere Team.

Command-R is a **scalable generative model** optimized for long-context tasks such as retrieval-augmented generation (RAG) and external tool/API use. It is designed to work alongside Cohere’s Embed and Rerank models to provide best-in-class performance for RAG pipelines, especially in enterprise use cases.

Key highlights:
- Strong accuracy on RAG and Tool Use  
- Low latency and high throughput  
- Longer 128k token context length  
- Multilingual support across 10 key languages  
- Model weights available on Hugging Face for research and evaluation  

You can find all the original Command-R checkpoints under the [Cohere Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01) collection.

This model was contributed by [Saurabh Dash](https://huggingface.co/saurabhdash) and [Ahmet Üstün](https://huggingface.co/ahmetustun). The code of the implementation in Hugging Face is based on [GPT-NeoX](https://github.com/EleutherAI/gpt-neox).

> [!TIP]
> Click on the Cohere models in the right sidebar for more examples of how to apply Cohere to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] and[`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="CohereForAI/c4ai-command-r-v01",
    torch_dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```python
# pip install transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Format message with the command-r chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below demonstrates loading a 4bit quantized model using [bitsandbytes](../quantization/bitsandbytes).

```python
# pip install transformers bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
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
<Tip warning={true}>
The checkpoints uploaded on the Hub use `torch_dtype = 'float16'`, which will be
used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`. 

The `dtype` of the online weights is mostly irrelevant unless you are using `torch_dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online), then it will be casted to the default `dtype` of `torch` (becomes `torch.float32`), and finally, if there is a `torch_dtype` provided in the config, it will be used. 

Training the model in `float16` is not recommended and is known to produce `nan`; as such, the model should be trained in `bfloat16`.
</Tip>

When using Flash Attention 2 via `attn_implementation="flash_attention_2"`, don't pass `torch_dtype` to the `from_pretrained` class method and use Automatic Mixed-Precision training. When using `Trainer`, it is simply specifying either `fp16` or `bf16` to `True`. Otherwise, make sure you are using `torch.autocast`. This is required because the Flash Attention only support `fp16` and `bf16` data type.


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