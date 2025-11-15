*This model was released by NVIDIA and added to Hugging Face Transformers in 2021.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
        <img alt="Mixed Precision" src="https://img.shields.io/badge/Mixed%20Precision-eae0c8?style=flat">
    </div>
</div>

# Megatron-BERT

[Megatron-BERT](https://arxiv.org/abs/2104.02096) is a family of large-scale BERT models trained with NVIDIA's Megatron-LM framework. These models are designed for high performance and scalability, supporting distributed training, tensor parallelism, and mixed precision. Megatron-BERT is ideal for tasks requiring powerful language understanding and is available in various sizes.

You can find all the original Megatron-BERT checkpoints under the [Megatron-BERT collection](https://huggingface.co/models?search=megatron-bert).

> [!TIP]
> Click on the Megatron-BERT models in the right sidebar for more examples of how to apply Megatron-BERT to different NLP tasks.

## Usage Examples

### Pipeline
```python
from transformers import pipeline
pipe = pipeline('fill-mask', model='nvidia/megatron-bert-uncased-345m')
result = pipe('The capital of France is [MASK].')
print(result)
```

### AutoModel
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('nvidia/megatron-bert-uncased-345m')
model = AutoModelForMaskedLM.from_pretrained('nvidia/megatron-bert-uncased-345m')
inputs = tokenizer('The capital of France is [MASK].', return_tensors='pt')
outputs = model(**inputs)
```

### transformers CLI
```
transformers-cli download nvidia/megatron-bert-uncased-345m
```

## Quantization
Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses bitsandbytes to quantize the weights to int8:
```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained('nvidia/megatron-bert-uncased-345m', load_in_8bit=True)
```

## Attention Mask Visualizer
Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/attention_visualizer.py) to better understand what tokens the model can and cannot attend to.

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer
visualizer = AttentionMaskVisualizer('nvidia/megatron-bert-uncased-345m')
visualizer('The capital of France is [MASK].')
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/megatron-bert-attn-mask.png"/>
</div>

## Notes

- Megatron-BERT is optimized for large-scale training and inference.
- Supports distributed training, tensor parallelism, and mixed precision.
- For best performance, use with NVIDIA GPUs and distributed setups.

## Resources
- [Megatron-BERT Paper](https://arxiv.org/abs/2104.02096)
- [NVIDIA Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Hugging Face Model Page](https://huggingface.co/models?search=megatron-bert)

## MegatronBertConfig

[[autodoc]] MegatronBertConfig

## MegatronBertModel

[[autodoc]] MegatronBertModel
    - forward

## MegatronBertForMaskedLM

[[autodoc]] MegatronBertForMaskedLM
    - forward

## MegatronBertForSequenceClassification

[[autodoc]] MegatronBertForSequenceClassification
    - forward

## MegatronBertForTokenClassification

[[autodoc]] MegatronBertForTokenClassification
    - forward

