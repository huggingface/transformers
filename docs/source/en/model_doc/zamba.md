# Zamba

Zamba is a large language model (LLM) trained by Zyphra, and made available under an Apache 2.0 license. Please see the [Zyphra Hugging Face](https://huggingface.co/collections/zyphra/) repository for model weights.


## Model details

Zamba-7B-v1 is a hybrid between state-space models (Specifically [Mamba](https://github.com/state-spaces/mamba)) and transformer, and was trained using next-token prediction. Zamba uses a shared transformer layer after every 6 mamba blocks. It uses the [Mistral v0.1 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1). We came to this architecture after a series of ablations at small scales. Zamba-7B-v1 was pre-trained on 1T tokens of text and code data.

<img src="zamba-arch.png" width=40% height=40% />


## Quick start

### Presequities

Jamba requires you use `transformers` version 4.39.0 or higher:
```bash
pip install transformers>=4.39.0
```

In order to run optimized Mamba implementations, you first need to install `mamba-ssm` and `causal-conv1d`:
```bash
pip install mamba-ssm causal-conv1d>=1.2.0
```
You also have to have the model on a CUDA device.

You can run the model not using the optimized Mamba kernels, but it is **not** recommended as it will result in significantly lower latencies. In order to do that, you'll need to specify `use_mamba_kernels=False` when loading the model.

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-v1")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba-v1", device_map="auto", torch_dtype=torch.bfloat16)

input_text = "A funny prompt would be "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Model card

The model cards can be found at:
* [Zamba-7B](MODEL_CARD_ZAMBA-7B-v1.md)

## Issues
For issues with model output, or community discussion, please use the Hugging Face community [forum](https://huggingface.co/zyphra/zamba-7b)

## License

The model weights are open-sourced via an Apache 2.0 license.