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
*This model was contributed to Hugging Face Transformers on 2026-11-11.*

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

Pass right-padded state inputs and flattened option inputs to `JevaForDecision`. Soft `labels` have shape
`(batch_size, max_num_options + 1)`: null probability first, then option probabilities, with zero on padding.
The loss follows the [official implementation](https://github.com/LiQiiiii/Jeva/blob/e2104e0720100f979f2cdd456a1cbbee96da0abd/src/jeva/finetune.py#L218-L231):
candidate cross-entropy plus binary coverage loss when `allow_null=True`. It returns standard `loss` and `logits`
for `Trainer` and TRL. Pass `num_items_in_batch` to normalize by decisions across accumulation steps.

Training updates the encoder and decision head by default. MoE router auxiliary losses are not added automatically.

`outputs.logits` contains conditional candidate log probabilities and `outputs.coverage_logits` contains coverage logits. For a nullable choice, candidate probabilities are `sigmoid(coverage_logits) * exp(logits)` and null probability is `sigmoid(-coverage_logits)`. For a closed choice, use `exp(logits)` directly.

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
