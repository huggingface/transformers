<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-24.*

# Jeva

Jeva was introduced in [Jeva: System One Models from First Principles](https://huggingface.co/papers/2610.18773). It encodes a state and its options independently with a shared text model. A matrix decision stack predicts a distribution over the options and a separate coverage probability: whether any supplied option applies. It supports choice, binary (noul), and score decisions without producing vocabulary logits.

Convert the [Jeva-4B checkpoint](https://huggingface.co/LIQIIIII/Jeva-4B) before loading it with `AutoModelForDecision`:

```bash
python src/transformers/models/jeva/convert_jeva_weights_to_hf.py --checkpoint LIQIIIII/Jeva-4B --output_dir jeva-4b
```

To train a new head, initialize it on a pretrained text encoder:

```python
from transformers import AutoModel, AutoTokenizer, JevaConfig, JevaForDecision, pipeline

repo = "Qwen/Qwen3-0.6B"
encoder = AutoModel.from_pretrained(repo)
model = JevaForDecision(JevaConfig(text_config=encoder.config, projection_dim=128), model=encoder)
tokenizer = AutoTokenizer.from_pretrained(repo)
```

## Training

Jeva combines candidate cross-entropy with binary coverage loss for nullable decisions, following the
[official implementation](https://github.com/LiQiiiii/Jeva/blob/e2104e0720100f979f2cdd456a1cbbee96da0abd/src/jeva/finetune.py#L218-L231).
With the model and tokenizer above, give the first option 80% target probability and null 20%.

```python
import torch

model.train()
state = tokenizer(["My card was charged twice."], return_tensors="pt")
options = tokenizer(["Refund", "Delivery"], padding=True, return_tensors="pt")
output = model(
    **state,
    option_input_ids=options.input_ids,
    option_attention_mask=options.attention_mask,
    num_options=torch.tensor([2]),
    labels=torch.tensor([[0.2, 0.8, 0.0]]),
    allow_null=torch.tensor([True]),
)
output.loss.backward()
```

Use right padding and put null probability first in `labels`. Training updates the encoder and head by default.
For exact weighting across uneven microbatches in a custom loop, pass the total decision count as
`num_items_in_batch`. The standard `Trainer` averages the microbatch losses.

After training, save and reload with `save_pretrained` and `AutoModelForDecision.from_pretrained`. The [decision pipeline](../main_classes/pipelines#transformers.DecisionPipeline) accepts the shared `state` / `questions` schema. A choice with `allow_null=True` also returns `null_probability` and can return `choice=None`. Score questions can supply numeric `values` for their rubric levels.

```python
decide = pipeline("decision", model=model.eval(), tokenizer=tokenizer)
decide("My card was charged twice.", questions={
    "intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"refund": None, "delivery": None}},
})
```

## JevaConfig

[[autodoc]] JevaConfig

## JevaForDecision

[[autodoc]] JevaForDecision
    - forward
