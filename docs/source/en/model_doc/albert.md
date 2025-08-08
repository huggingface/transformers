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
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
        <img alt= "TensorFlow" src= "https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" >
        <img alt= "Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style…Nu+W0m6K/I9gGPd/dfx/EN/wN62AhsBWuAAAAAElFTkSuQmCC">
        <img alt="SDPA" src= "https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white" > 
    </div>
</div>

# ALBERT

[ALBERT](https://huggingface.co/papers/1909.11942) is designed to address memory limitations of scaling and training of [BERT](./bert). It adds two parameter reduction techniques. The first, factorized embedding parametrization, splits the larger vocabulary embedding matrix into two smaller matrices so you can grow the hidden size without adding a lot more parameters. The second, cross-layer parameter sharing, allows layer to share parameters which keeps the number of learnable parameters lower.

ALBERT was created to address problems like -- GPU/TPU memory limitations, longer training times, and unexpected model degradation in BERT. ALBERT uses two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

- **Factorized embedding parameterization:** The large vocabulary embedding matrix is decomposed into two smaller matrices, reducing memory consumption.
- **Cross-layer parameter sharing:** Instead of learning separate parameters for each transformer layer, ALBERT shares parameters across layers, further reducing the number of learnable weights.

ALBERT uses absolute position embeddings (like BERT) so padding is applied at right. Size of embeddings is 128 While BERT uses 768. ALBERT can processes maximum 512 token at a time.

You can find all the original ALBERT checkpoints under the [ALBERT community](https://huggingface.co/albert) organization.

> [!TIP]
> Click on the ALBERT models in the right sidebar for more examples of how to apply ALBERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="albert-base-v2",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.", top_k=5)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForMaskedLM.from_pretrained(
    "albert/albert-base-v2",
    dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto"
)

prompt = "Plants create energy through a process known as [MASK]."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    predictions = outputs.logits[0, mask_token_index]

top_k = torch.topk(predictions, k=5).indices.tolist()
for token_id in top_k[0]:
    print(f"Prediction: {tokenizer.decode([token_id])}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers run --task fill-mask --model albert-base-v2 --device 0
```

</hfoption>

</hfoptions>

## Notes

- Inputs should be padded on the right because BERT uses absolute position embeddings.
- The embedding size `E` is different from the hidden size `H` because the embeddings are context independent (one embedding vector represents one token) and the hidden states are context dependent (one hidden state represents a sequence of tokens). The embedding matrix is also larger because `V x E` where `V` is the vocabulary size. As a result, it's more logical if `H >> E`. If `E < H`, the model has less parameters.

## Resources

The resources provided in the following sections consist of a list of official Hugging Face and community (indicated by 🌎) resources to help you get started with AlBERT. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- [`AlbertForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).

- [`TFAlbertForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification).

- [`FlaxAlbertForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).
- Check the [Text classification task guide](../tasks/sequence_classification) on how to use the model.

<PipelineTag pipeline="token-classification"/>

- [`AlbertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification).

- [`TFAlbertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

- [`FlaxAlbertForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- [Token classification](https://huggingface.co/course/chapter7/2?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Token classification task guide](../tasks/token_classification) on how to use the model.

<PipelineTag pipeline="fill-mask"/>

- [`AlbertForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFAlbertForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxAlbertForMaskedLM`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Masked language modeling task guide](../tasks/masked_language_modeling) on how to use the model.

<PipelineTag pipeline="question-answering"/>

- [`AlbertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [`TFAlbertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).
- [`FlaxAlbertForQuestionAnswering`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter of the 🤗 Hugging Face Course.
- Check the [Question answering task guide](../tasks/question_answering) on how to use the model.

**Multiple choice**

- [`AlbertForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- [`TFAlbertForMultipleChoice`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

- Check the [Multiple choice task guide](../tasks/multiple_choice) on how to use the model.

## AlbertConfig

[[autodoc]] AlbertConfig

## AlbertTokenizer

[[autodoc]] AlbertTokenizer - build_inputs_with_special_tokens - get_special_tokens_mask - create_token_type_ids_from_sequences - save_vocabulary

## AlbertTokenizerFast

[[autodoc]] AlbertTokenizerFast

## Albert specific outputs

[[autodoc]] models.albert.modeling_albert.AlbertForPreTrainingOutput

[[autodoc]] models.albert.modeling_tf_albert.TFAlbertForPreTrainingOutput

<frameworkcontent>
<pt>

## AlbertModel

[[autodoc]] AlbertModel - forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining - forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM - forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification - forward

## AlbertForMultipleChoice

[[autodoc]] AlbertForMultipleChoice

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification - forward

## AlbertForQuestionAnswering

[[autodoc]] AlbertForQuestionAnswering - forward

</pt>

<tf>

## TFAlbertModel

[[autodoc]] TFAlbertModel - call

## TFAlbertForPreTraining

[[autodoc]] TFAlbertForPreTraining - call

## TFAlbertForMaskedLM

[[autodoc]] TFAlbertForMaskedLM - call

## TFAlbertForSequenceClassification

[[autodoc]] TFAlbertForSequenceClassification - call

## TFAlbertForMultipleChoice

[[autodoc]] TFAlbertForMultipleChoice - call

## TFAlbertForTokenClassification

[[autodoc]] TFAlbertForTokenClassification - call

## TFAlbertForQuestionAnswering

[[autodoc]] TFAlbertForQuestionAnswering - call

</tf>
<jax>

## FlaxAlbertModel

[[autodoc]] FlaxAlbertModel - **call**

## FlaxAlbertForPreTraining

[[autodoc]] FlaxAlbertForPreTraining - **call**

## FlaxAlbertForMaskedLM

[[autodoc]] FlaxAlbertForMaskedLM - **call**

## FlaxAlbertForSequenceClassification

[[autodoc]] FlaxAlbertForSequenceClassification - **call**

## FlaxAlbertForMultipleChoice

[[autodoc]] FlaxAlbertForMultipleChoice - **call**

## FlaxAlbertForTokenClassification

[[autodoc]] FlaxAlbertForTokenClassification - **call**

## FlaxAlbertForQuestionAnswering

[[autodoc]] FlaxAlbertForQuestionAnswering - **call**

</jax>
</frameworkcontent>
