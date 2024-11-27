# Cohere

## Overview

The Cohere Command-R model was proposed in the blogpost [Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/) by the Cohere Team.

The abstract from the paper is the following:

*Command-R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise. Today, we are introducing Command-R, a new LLM aimed at large-scale production workloads. Command-R targets the emerging “scalable” category of models that balance high efficiency with strong accuracy, enabling companies to move beyond proof of concept, and into production.*

*Command-R is a generative model optimized for long context tasks such as retrieval augmented generation (RAG) and using external APIs and tools. It is designed to work in concert with our industry-leading Embed and Rerank models to provide best-in-class integration for RAG applications and excel at enterprise use cases. As a model built for companies to implement at scale, Command-R boasts:
- Strong accuracy on RAG and Tool Use
- Low latency, and high throughput
- Longer 128k context and lower pricing
- Strong capabilities across 10 key languages
- Model weights available on HuggingFace for research and evaluation

Checkout model checkpoints [here](https://huggingface.co/CohereForAI/c4ai-command-r-v01).
This model was contributed by [Saurabh Dash](https://huggingface.co/saurabhdash) and [Ahmet Üstün](https://huggingface.co/ahmetustun). The code of the implementation in Hugging Face is based on GPT-NeoX [here](https://github.com/EleutherAI/gpt-neox).

## Usage tips

<Tip warning={true}>

The checkpoints uploaded on the Hub use `torch_dtype = 'float16'`, which will be
used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`. 

The `dtype` of the online weights is mostly irrelevant unless you are using `torch_dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online), then it will be casted to the default `dtype` of `torch` (becomes `torch.float32`), and finally, if there is a `torch_dtype` provided in the config, it will be used. 

Training the model in `float16` is not recommended and is known to produce `nan`; as such, the model should be trained in `bfloat16`.

</Tip>
The model and tokenizer can be loaded via:

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

- When using Flash Attention 2 via `attn_implementation="flash_attention_2"`, don't pass `torch_dtype` to the `from_pretrained` class method and use Automatic Mixed-Precision training. When using `Trainer`, it is simply specifying either `fp16` or `bf16` to `True`. Otherwise, make sure you are using `torch.autocast`. This is required because the Flash Attention only support `fp16` and `bf16` data type.


## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Command-R. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.


<PipelineTag pipeline="text-generation"/>

Loading FP16 model
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

Loading bitsnbytes 4bit quantized model
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


