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
*This model was contributed to Hugging Face Transformers on 2026-10-28.*

# Clef

Clef is a decision head that scores all the options of all the questions of a request jointly. It was introduced by [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) and is also used by [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash). The template marks the span of each instruction and of each of its options. The options of all the questions are packed in one sequence: pooled from the hidden states and from the output embeddings of their tokens, they attend to the prompt and are summarized into the field of their question. The fields then attend to each other and to the prompt, and each option is scored against the field of its question. The original `joint_head.safetensors` loads as is.

Set `head_type="cosine"` in [`ClefConfig`] for a much smaller head, without the language modeling head, the encoder or the scorer: the hidden states of the backbone are averaged over the span of each instruction and of each option, projected (`cosine_width`) and normalized, and each option is scored by the cosine with its question times a learned temperature (initially `cosine_initial_scale`, capped at 100), all in float32. The output is the same as the joint head's.

Decision heads are applied by [`DecisionPipeline`] to the hidden states of the original checkpoint. The chat template,
the settings and the head live in their own repo next to it.

```python
from transformers import pipeline

decide = pipeline("decision", model="Cloudflare/clef-flash")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is a refund requested?"},
            "topic": {"type": "choice", "instructions": "What is it about?", "criteria": {"billing": None, "shipping": None}},
        },
    }
)
```

## Training

[`ClefForDecision`] computes a loss when it is given `labels`, one per question of the prompt: the index of the right option, `-100` to ignore the question, or a distribution over its options (shape `(num_questions, max_options)`). The output is a [`~models.clef.modeling_clef.ClefDecisionOutput`] whose float32 `logits` have the shape `(num_questions, max_options)`, padded with the minimum of their dtype where a question has fewer options, and whose `option_mask` marks the options that exist. The loss is the sum of three terms, whose weights are fields of [`ClefConfig`]:

- the cross entropy of the softmax over the options of each question, averaged over the questions. `label_smoothing` (default 0.0) mixes the uniform distribution over the options into its target.
- `brier_weight` (default 0.0) times the squared distance of the probabilities to the target, summed over the options and averaged over the questions.
- `ordinal_weight` (default 0.0) times, on score questions only, the expected distance `|i - j|` between the options and the right one, divided by the number of options minus one and averaged over the score questions.

`forward` takes a padded batch of prompts: `input_ids` and `attention_mask` of shape `(batch_size, length)` (left or right padding), `marks` of shape `(batch_size, num_marks)` padded with `-1`, and `num_options` and `question_type_ids` of shape `(batch_size, max_questions)` padded with `0`. A question is real when it has options; the rows of `logits` and of `labels` are the real questions of every prompt in turn, so the loss of a batch is the mean over all its questions. Padding is masked out of every attention of the head and of the routing of the options, and the spans are averaged from float32 prefix sums, in memory linear in the length of the prompt.

Hard and soft labels give the same loss for the same target. Pass `num_items_in_batch` to the loss to divide the summed cross entropy and Brier terms by it instead of by the number of questions.

Cloudflare has not published Clef's training objective; these terms and their defaults are a community recipe.

## ClefConfig

[[autodoc]] ClefConfig

## ClefForDecision

[[autodoc]] ClefForDecision
    - forward

## ClefDecisionOutput

[[autodoc]] models.clef.modeling_clef.ClefDecisionOutput
