<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# OLMoE

<<<<<<< Updated upstream
The OLMoE model was proposed in [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060) by Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, Hannaneh Hajishirzi.
=======
[OLMoE](https://huggingface.co/papers/2409.02060) stands for **O**pen **L**anguage **Mo**dels using sparse **M**ixture-**o**f-**E**xperts, created by AllenAI. Released with open data and training code, the model is a [Mixture of Experts](https://huggingface.co/blog/moe) (MoE) : a neural network architecture that replaces dense layers with multiple specialized "expert" networks and uses a gating mechanism to route different parts of the input to only the most relevant experts, allowing the model to be much larger while using the same computational resources during inference. 
>>>>>>> Stashed changes

You can find all the original OLMoE checkpoints under the [OLMoE](https://huggingface.co/collections/allenai/olmoe-january-2025-67992134f9ebea0a941706ca) collection.

> [!TIP]
> This model was contributed by [Muennighoff](https://hf.co/Muennighoff).
>
> Click on the OLMoE models in the right sidebar for more examples of how to apply OLMoE to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`OlmoeForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline>

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/OLMoE-1B-7B-0125",
    torch_dtype=torch.float16,
    device=0,
)

result = pipe("Dionysus is the god of")
print(result)
```

</hfoption>
<hfoption id="OlmoeForCausalLM">

```py
from transformers import OlmoeForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`
model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
inputs = tokenizer("The 26th letter of the alphabet is", return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
out = model.generate(**inputs, max_length=64)
print(tokenizer.decode(out[0]))
```

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization overview](https://huggingface.co/docs/transformers/en/quantization/overview) for more available quantization backends.

The example below uses [BitsAndBytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) to quantize the weights to 4-bit precision.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4"
)

# Load quantized OLMoE model
model = AutoModelForCausalLM.from_pretrained(
   "allenai/OLMoE-1B-7B-0125",
   quantization_config=quantization_config,
   device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")

# Use the quantized model
inputs = tokenizer("How many arms does an octopus have?", return_tensors="pt")
with torch.no_grad():
   outputs = model.generate(**inputs, max_length=64, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```