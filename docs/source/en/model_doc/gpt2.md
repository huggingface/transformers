<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>


# GPT-2

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) is a scaled up version of GPT, a causal transformer language model, with 10x more parameters and training data. The model was pretrained on a 40GB dataset to predict the next word in a sequence based on all the previous words. This approach enabled the model to perform many downstream tasks in a zero-shot setting.

The model architecture uses a unidirectional (causal) attention mechanism where each token can only attend to previous tokens, making it particularly effective for text generation tasks.

You can find all the original GPT-2 checkpoints under the [OpenAI community](https://huggingface.co/openai-community?search_models=gpt) organization.

> [!TIP]
> Click on the GPT-2 models in the right sidebar for more examples of how to apply GPT-2 to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2", dtype=torch.float16, device=0)
pipeline("Hello, I'm a language model")
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

input_ids = tokenizer("Hello, I'm a language model", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Hello, I'm a language model" | transformers run --task text-generation --model openai-community/gpt2 --device 0
```

</hfoption>
</hfoptions>

One can also serve the model using vLLM with the `transformers backend`.

```
vllm serve openai-community/gpt2 --model-imp transformers
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2-xl",
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
inputs = tokenizer("Once upon a time, there was a magical forest", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- Pad inputs on the right because GPT-2 uses absolute position embeddings.
- GPT-2 can reuse previously computed key-value attention pairs. Access this feature with the [past_key_values](https://huggingface.co/docs/transformers//en/model_doc/gpt2#transformers.GPT2Model.forward.past_key_values) parameter in [`GPT2Model.forward`].
- Enable the [scale_attn_by_inverse_layer_idx](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.scale_attn_by_inverse_layer_idx) and [reorder_and_upcast_attn](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.reorder_and_upcast_attn) parameters to apply the training stability improvements from [Mistral](./mistral).

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer
    - save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## GPT2 specific outputs

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput

<frameworkcontent>
<pt>

## GPT2Model

[[autodoc]] GPT2Model
    - forward

## GPT2LMHeadModel

[[autodoc]] GPT2LMHeadModel
    - forward

## GPT2DoubleHeadsModel

[[autodoc]] GPT2DoubleHeadsModel
    - forward

## GPT2ForQuestionAnswering

[[autodoc]] GPT2ForQuestionAnswering
    - forward

## GPT2ForSequenceClassification

[[autodoc]] GPT2ForSequenceClassification
    - forward

## GPT2ForTokenClassification

[[autodoc]] GPT2ForTokenClassification
    - forward

</pt>
<tf>

## TFGPT2Model

[[autodoc]] TFGPT2Model
    - call

## TFGPT2LMHeadModel

[[autodoc]] TFGPT2LMHeadModel
    - call

## TFGPT2DoubleHeadsModel

[[autodoc]] TFGPT2DoubleHeadsModel
    - call

## TFGPT2ForSequenceClassification

[[autodoc]] TFGPT2ForSequenceClassification
    - call

## TFSequenceClassifierOutputWithPast

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutputWithPast

## TFGPT2Tokenizer

[[autodoc]] TFGPT2Tokenizer

</tf>
<jax>

## FlaxGPT2Model

[[autodoc]] FlaxGPT2Model
    - __call__

## FlaxGPT2LMHeadModel

[[autodoc]] FlaxGPT2LMHeadModel
    - __call__

</jax>
</frameworkcontent>
