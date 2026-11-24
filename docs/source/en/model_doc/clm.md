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

# CLM

[Contrastive Language Models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) encode states and candidates
independently with a shared language encoder. The released CLM-v0.1-8B uses Qwen3-8B with normalized last-token
pooling and separate state and action projection heads. Scaled cosine similarities produce candidate scores.
The publisher checkpoint contains projection heads only; its encoder is loaded separately.

CLM uses the shared [decision API](jev): `model.decide` and `pipeline("decision")` answer Choice, Noul and Score
questions. Structured inputs use the publisher's prose formatting. A candidate's description is encoded when supplied;
otherwise its key is used. Candidates do not see the current state or other candidates. Null decisions are unsupported. Answer confidence follows the shared decision API.

## Training

CLM uses bidirectional InfoNCE over the candidates in each forward pass. After converting the checkpoint below,
pack each state followed by its matching candidate and mark the end of each segment.

```python
import torch
from transformers import AutoModelForDecision, AutoTokenizer

model = AutoModelForDecision.from_pretrained("clm-8b")
tokenizer = AutoTokenizer.from_pretrained("clm-8b")
segments = tokenizer(["I was charged twice.", "Billing", "Where is my parcel?", "Delivery"]).input_ids
model.train()
output = model(
    input_ids=torch.tensor([[token for segment in segments for token in segment]]),
    marks=torch.tensor([[len(segment) for segment in segments]]).cumsum(dim=-1),
    num_options=torch.tensor([[1, 1]]),
    labels=torch.tensor([0, 1]),
)
output.loss.backward()
```

Use `group_ids` in paired batches to exclude same-group false negatives. Each microbatch has its own contrastive pool.
Gradient accumulation averages losses across those pools. To train only the projection heads and scale, freeze
`model.base_model` before training.

## Convert a checkpoint

The published heads use a separate Qwen3-8B encoder. Convert both into a standard Transformers checkpoint:

```bash
python src/transformers/models/clm/convert_clm_weights_to_hf.py --output_dir clm-8b
```

```python
from transformers import pipeline

decide = pipeline("decision", model="clm-8b", dtype="bfloat16", device="cuda")
answers = decide({"state": "My invoice was charged twice.", "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"}
}})
```

## ClmConfig

[[autodoc]] ClmConfig

## ClmForDecision

[[autodoc]] ClmForDecision
    - forward
    - decide
