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
        <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ELECTRA

[ELECTRA](https://huggingface.co/papers/2003.10555) modifies the pretraining objective of traditional masked language models like BERT. Instead of just masking tokens and asking the model to predict them, ELECTRA trains two models, a generator and a discriminator. The generator replaces some tokens with plausible alternatives and the discriminator (the model you'll actually use) learns to detect which tokens are original and which were replaced. This training approach is very efficient and scales to larger models while using considerably less compute.

This approach is super efficient because ELECTRA learns from every single token in the input, not just the masked ones. That's why even the small ELECTRA models can match or outperform much larger models while using way less computing resources.

You can find all the original ELECTRA checkpoints under the [ELECTRA](https://huggingface.co/collections/google/electra-release-64ff6e8b18830fabea30a1ab) release.

> [!TIP]
> Click on the right sidebar for more examples of how to use ELECTRA for different language tasks like sequence classification, token classification, and question answering.

The example below demonstrates how to classify text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="bhadresh-savani/electra-base-emotion",
    dtype=torch.float16,
    device=0
)
classifier("This restaurant has amazing food!")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "bhadresh-savani/electra-base-emotion",
)
model = AutoModelForSequenceClassification.from_pretrained(
    "bhadresh-savani/electra-base-emotion",
    dtype=torch.float16
)
inputs = tokenizer("ELECTRA is more efficient than BERT", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {predicted_label}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "This restaurant has amazing food." | transformers run --task text-classification --model bhadresh-savani/electra-base-emotion --device 0
```

</hfoption>
</hfoptions>

## Notes

- ELECTRA consists of two transformer models, a generator (G) and a discriminator (D). For most downstream tasks, use the discriminator model (as indicated by `*-discriminator` in the name) rather than the generator.
- ELECTRA comes in three sizes: small (14M parameters), base (110M parameters), and large (335M parameters).
- ELECTRA can use a smaller embedding size than the hidden size for efficiency. When `embedding_size` is smaller than `hidden_size` in the configuration, a projection layer connects them.
- When using batched inputs with padding, make sure to use attention masks to prevent the model from attending to padding tokens.

    ```py
    # Example of properly handling padding with attention masks
    inputs = tokenizer(["Short text", "This is a much longer text that needs padding"],
                    padding=True,
                    return_tensors="pt")
    outputs = model(**inputs)  # automatically uses the attention_mask
    ```

- When using the discriminator for a downstream task, you can load it into any of the ELECTRA model classes ([`ElectraForSequenceClassification`], [`ElectraForTokenClassification`], etc.).

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## Electra specific outputs

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

[[autodoc]] models.electra.modeling_tf_electra.TFElectraForPreTrainingOutput

<frameworkcontent>
<pt>

## ElectraModel

[[autodoc]] ElectraModel
    - forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining
    - forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM
    - forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM
    - forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification
    - forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice
    - forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification
    - forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering
    - forward

</pt>
<tf>

## TFElectraModel

[[autodoc]] TFElectraModel
    - call

## TFElectraForPreTraining

[[autodoc]] TFElectraForPreTraining
    - call

## TFElectraForMaskedLM

[[autodoc]] TFElectraForMaskedLM
    - call

## TFElectraForSequenceClassification

[[autodoc]] TFElectraForSequenceClassification
    - call

## TFElectraForMultipleChoice

[[autodoc]] TFElectraForMultipleChoice
    - call

## TFElectraForTokenClassification

[[autodoc]] TFElectraForTokenClassification
    - call

## TFElectraForQuestionAnswering

[[autodoc]] TFElectraForQuestionAnswering
    - call

</tf>
<jax>

## FlaxElectraModel

[[autodoc]] FlaxElectraModel
    - __call__

## FlaxElectraForPreTraining

[[autodoc]] FlaxElectraForPreTraining
    - __call__

## FlaxElectraForCausalLM

[[autodoc]] FlaxElectraForCausalLM
    - __call__

## FlaxElectraForMaskedLM

[[autodoc]] FlaxElectraForMaskedLM
    - __call__

## FlaxElectraForSequenceClassification

[[autodoc]] FlaxElectraForSequenceClassification
    - __call__

## FlaxElectraForMultipleChoice

[[autodoc]] FlaxElectraForMultipleChoice
    - __call__

## FlaxElectraForTokenClassification

[[autodoc]] FlaxElectraForTokenClassification
    - __call__

## FlaxElectraForQuestionAnswering

[[autodoc]] FlaxElectraForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>
