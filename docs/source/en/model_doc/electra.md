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
*This model was released on 2020-03-23 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
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
