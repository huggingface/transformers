# CamemBERT Base
<div style="float: right;">
	<div class="flex flex-wrap space-x-1">
		<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
		<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
	</div>
</div>

[CamemBERT](https://huggingface.co/docs/transformers/model_doc/camembert) is a language model based on RoBERTa, but it was trained specifically on French text from the OSCAR dataset.

What sets CamemBERT apart is that it learned from a huge, high quality collection of French data, as opposed to mixing lots of languages. This helps it really understand French better than many multilingual models.

Common applications of CamemBERT include masked language modeling (Fill-mask prediction), text classification (sentiment analysis), token classification (entity recognition) and sentence pair classification (entailment tasks).

You can find all the original CamemBERT checkpoints under the [CamemBERT](https://huggingface.co/models?search=camembert) collection.

> [!TIP]
> This model was contributed by the [Facebook AI](https://huggingface.co/facebook) team.
>
> Click on the CamemBERT models in the right sidebar for more examples of how to apply CamemBERT to different NLP tasks.

The examples below demonstrate how to perform masked language modeling with `pipeline` or the `AutoModel` class.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="camembert-base")
result = fill_mask("Le camembert est un délicieux fromage <mask>.")
print(result)
```

</hfoption>

<hfoption id="AutoModel">

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")

inputs = tokenizer("Le camembert est un délicieux fromage <mask>.", return_tensors="pt")
outputs = model(**inputs)
```

</hfoption>

</hfoptions>

Quantization reduces the memory burden of large models by representing weights in lower precision. Refer to the [Quantization](https://huggingface.co/docs/transformers/main/en/quantization) overview for available options.
The example below uses [BitsAndBytes](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#load-in-8bit-or-4bit-using-bitsandbytes) quantization to load the model in 8-bit precision.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForMaskedLM.from_pretrained(
    "camembert-base",
    quantization_config=quant_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
```
	
## Notes

- CamemBERT uses RoBERTa pretraining objectives.
- It makes use of a SentencePiece tokenizer.
- It does not support token type IDs (segment embeddings).
- Special pre-processing/post-processing is not needed.

## Resources

- [Original Paper](https://arxiv.org/abs/1911.03894)
- [Hugging Face Model Card](https://huggingface.co/camembert-base)
