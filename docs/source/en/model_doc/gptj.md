<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-06-04 and added to Hugging Face Transformers on 2021-08-31.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# GPT-J

[GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) is a GPT-like model trained on [the Pile](https://pile.eleuther.ai/) dataset. It was trained with the Mesh Transformer JAX framework, a model parallelism scheme. This model efficiently computes attention and feedforward neural networks in parallel and uses rotary position embeddings to better inject positional information.

You can find all the original [GPT-J] checkpoints under the [EleutherAI](https://huggingface.co/EleutherAI/models?search=gpt-j) organization.

> [!TIP]
> This model was contributed by [Stella Biderman](https://huggingface.co/stellaathena).
> Click on the GPT-J models in the right sidebar for more examples of how to apply GPT-J to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptoins id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="EleutherAI/gpt-j-6B",
    dtype=torch.float16,
    device=0
)
pipeline(
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains."
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, infer_device

device = infer_device()
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
tokenizer.batch_decode(output)[0]
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains." | transformers run --task text-generation --model EleutherAI/gpt-j-6B --device 0
```

</hfoption>
</hfoptions>

## Notes

- Training requires at least 4x model size GPU memory even with mixed precision. Explore options such as DeepSpeed or use the original codebase to train and fine-tune the model on TPU and convert to Transformers for inference (see [here](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md)).

- Although the embedding matrix is 50400, only 50257 entries are used. The extra tokens are added for TPU efficiency. To avoid a mismatch between embedding matrix size and vocab size, the GPT-J tokenizer contains 143 extra tokens (`<|extratoken_1|>... <|extratoken_143|>`.

## Resources

- Blog on how to [Accelerate GPT-J inference with DeepSpeed-Inference on GPUs](https://www.philschmid.de/gptj-deepspeed-inference).
- Blog post introducing [GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/).
- Notebook for [GPT-J-6B Inference Demo](https://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb).
- Notebook demonstrating [Inference with GPT-J-6B](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GPT-J-6B/Inference_with_GPT_J_6B.ipynb).
- [`GPTJForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling), [text generation example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation), and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
- [`TFGPTJForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)
- [`FlaxGPTJForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb)


## GPTJConfig

[[autodoc]] GPTJConfig
    - all

## GPTJModel

[[autodoc]] GPTJModel
    - forward

## GPTJForCausalLM

[[autodoc]] GPTJForCausalLM
    - forward

## GPTJForSequenceClassification

[[autodoc]] GPTJForSequenceClassification
    - forward

## GPTJForQuestionAnswering

[[autodoc]] GPTJForQuestionAnswering
    - forward
