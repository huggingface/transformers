# Zamba

Zamba is a large language model (LLM) trained by Zyphra, and made available under an Apache 2.0 license. Please see the [Zyphra Hugging Face](https://huggingface.co/Zyphra) repository for model weights.


## Model details

Zamba-7B-v1 is a hybrid between state-space models (Specifically [Mamba](https://github.com/state-spaces/mamba)) and transformer. Zamba consists of a Mamba backbone with a shared transformer block every 6 Mamba layers. We came to this architecture after a series of ablations at small scales, where we found it proved highly effective at combining the inference efficiency of Mamba with the minimal amount of attention required to maintain performance and expressivity. Zambaand was trained using next-token prediction and uses the [Mistral v0.1 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1).  Zamba-7B-v1 was pre-trained on 1T tokens of text and code data. We then performed an annealing phase over 50B high quality tokens.

<img src="https://github.com/Zyphra/HF-zamba/blob/main/zamba-arch.png" width=40% height=40% />


## Quick start

### Presequities

Zamba requires you use `transformers` version 4.39.0 or higher:
```bash
pip install transformers>=4.39.0
```

In order to run optimized Mamba implementations on a CUDA device, you first need to install `mamba-ssm` and `causal-conv1d`:
```bash
pip install mamba-ssm causal-conv1d>=1.2.0
```

You can run the model not using the optimized Mamba kernels, but it is **not** recommended as it will result in significantly higher latency. 

To run on CPU, please specify `use_mamba_kernels=False` when loading the model using ``AutoModelForCausalLM.from_pretrained``.

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba-7B-v1", device_map="auto", torch_dtype=torch.bfloat16)

input_text = "A funny prompt would be "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```


## Issues
For issues with model output, or community discussion, please use the Hugging Face community [forum](https://huggingface.co/zyphra/zamba-7b)

## License

The model weights are open-sourced via an Apache 2.0 license.
