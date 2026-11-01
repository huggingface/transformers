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

# CLM

[Contrastive Language Models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) encode states and candidates
independently with a shared language encoder. The released CLM-v0.1-8B uses Qwen3-8B with normalized last-token
pooling and separate state and action projection heads. Scaled cosine similarities produce candidate scores.
The publisher checkpoint contains projection heads only; its encoder is loaded separately.

CLM uses the shared [decision API](jev): `model.decide` and `pipeline("decision")` answer Choice, Noul and Score
questions. Structured inputs use the publisher's prose formatting. A candidate's description is encoded when supplied;
otherwise its key is used. Candidates do not see the current state or other candidates. Null decisions are unsupported.

For training, `labels` supplies matching candidate indices or target distributions over the in-batch candidate pool.
The model returns a standard output with `loss` and `logits`. The loss averages forward and reverse InfoNCE; reverse
supervision includes only candidates with positive target mass. Paired examples may supply `group_ids` to exclude
off-diagonal examples from the same group as false negatives, matching the official CLM trainer.
For the released heads' training setup, freeze `model.base_model` before constructing the Trainer.

## ClmConfig

[[autodoc]] ClmConfig

## ClmForDecision

[[autodoc]] ClmForDecision
    - forward
    - decide
