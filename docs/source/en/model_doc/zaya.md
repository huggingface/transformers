---
license: apache-2.0
library_name: transformers
---

# Model Card for ZAYA1-base

ZAYA1 is an 800m active/8.3B total parameter MoE model, and the first trained entirely end-to-end on AMD’s hardware, software, and networking stack.

Our ZAYA1 base model benchmark performance is extremely competitive with the SoTA Qwen3 series of models of comparable scale, and outperforms comparable western open-source models such as SmolLM3, and Phi4. ZAYA1-base excels especially at complex and challenging mathematical and STEM reasoning tasks, nearly matching the performance of SoTA Qwen3 thinking models under high pass@k settings even prior to explicit post-training for reasoning, and exceeds other strong reasoning models such as Phi4-reasoning, and Deepseek-R1-Distill. 

Details of our pretraining efforts, hardware specific optimizations, and ZAYA1 base model benchmarks are described in the [accompanying technical report](https://arxiv.org/abs/2511.17127).


## Model Details

ZAYA1's architecture includes several innovations developed at Zyphra. These include:

- **Compressed Convolutional Attention (CCA)**: [This novel attention](https://arxiv.org/abs/2510.04476) mechanism performs attention entirely in the latent space enabling significant reductions in parameter count, prefill compute, and KV cache size compared to alternative attention mechanisms, while also being more performant in loss/flop. 
- **ZAYA1 Router**: The ZAYA1 router makes fundamental improvements to the linear router used in almost all existing large-scale MoE models. The ZAYA1 router replaces the linear with a downprojection followed by a depth-mixing EDA layer then a three-layer MLP per expert to add significant nonlinear expressivity to the router.
- **Residual Scaling**: We add learnable scalar gates and biases to the residual stream and the outputs of each block. This provides a lightweight method to allow the model to carefully control its own norm and degree of forgetting across depth.


![zaya_arch](https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/Ih8RnOPNbtRzaVcH16ar-.png)

ZAYA1-base uses the [Gemma3](https://ai.google.dev/gemma/terms) tokenizer.


## Performance

ZAYA1-base performs extremely competitively against other base models of a similar and even greater scale.

![mmlu_pro_vs_ttft](https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/nyWieuzXks9H4GM71XAzn.png)

![Screenshot 2025-11-20 at 00.44.44](https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/tsdgc4KWWs4SXfo4orOp4.png)

## Quick start

### Prerequisites

To use ZAYA1, install `zaya` branch from our fork of `transformers` library, which is based on the v4.57.1 of `transformers`:  
```bash
pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"
```

The command above relies on requirements for `transformers v4.57.1` being installed in your environment. If you're installing in a fresh Python environment, you might want to specify a specific extra, like `[dev-torch]`, to install all the dependencies:  
```bash
pip install "transformers[dev-torch] @ git+https://github.com/Zyphra/transformers.git@zaya"
```


### Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/ZAYA1-base")
model = AutoModelForCausalLM.from_pretrained("Zyphra/ZAYA1-base", device_map="cuda", dtype=torch.bfloat16)

input_text = "What factors contributed to the fall of the Roman Empire?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## ZayaConfig

[[autodoc]] ZayaConfig

## ZayaModel

[[autodoc]] ZayaModel
    - forward

## ZayaForCausalLM

[[autodoc]] ZayaForCausalLM
    - forward
