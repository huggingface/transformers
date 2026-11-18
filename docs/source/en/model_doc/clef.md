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

The backbone runs attention over the prompt, and `attn_implementation` selects its implementation, for example `ClefForDecision.from_pretrained(repo, attn_implementation="kernels-community/flash-attn2")` for a Flash Attention kernel (or `attn_implementation={"text_config": ...}` for the backbone alone); the attention layers of the head run sdpa (or eager) whatever the setting.

Clef's decision configs set `labels` to `None` for every question type: the head pools the span of each option and reads no label token, so a question can have any number of options.

## Encoder backbones

The backbone can be any model of the Hub, including bidirectional encoders such as BERT or ModernBERT. Two fields of [`ClefConfig`] adapt the head to them:

- `global_pooling="first"` makes the first attended token of each prompt (the `[CLS]` token) its global state, instead of the last one, which only a causal backbone has read the whole prompt at.
- `prompt_style="plain"` gives a new model the decision config of `CLEF_PLAIN_TEMPLATE`: the same schema text and marks as the ChatML prompt of the checkpoints, without its turns and system prompt, between the `[CLS]` and `[SEP]` tokens of the tokenizer (or its beginning and end of sequence tokens).

When the backbone has no causal language modeling head class (ModernBERT), the model has no `lm_head`: the lexical prior of the options reads the input embeddings of the backbone. Positions skip the padding, so left and right padding give the same scores with absolute position embeddings too. Backbones of the RoBERTa family (RoBERTa, XLM-RoBERTa, CamemBERT, MPNet) number their positions themselves, from their padding token: pad their batches with that token (the tokenizer's `pad_token`), and a prompt scores the same with an attention mask (in training) and without one (in the pipeline).

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer, ClefConfig, ClefForDecision, pipeline

text_config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
config = ClefConfig(
    text_config=text_config,
    backbone_hidden_size=text_config.hidden_size,
    global_pooling="first",
    prompt_style="plain",
)
model = ClefForDecision(config)
# The weights of the pretrained encoder; the head is new and is to be trained
model.model.load_state_dict(AutoModel.from_pretrained("answerdotai/ModernBERT-base").state_dict())
decide = pipeline("decision", model=model, tokenizer=AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"))
```

A new head can also be built around a backbone that is already loaded, given as `model`: it is used as it is, with its weights, precision and attention implementation, and the new head is in float32 (move the model to the device of the backbone, and run a backbone in half precision under autocast, see [Converting checkpoints](#converting-checkpoints)).

```python
from transformers import AutoModel, ClefConfig, ClefForDecision

backbone = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa")
config = ClefConfig(
    text_config=backbone.config,
    backbone_hidden_size=backbone.config.hidden_size,
    global_pooling="first",
    prompt_style="plain",
)
model = ClefForDecision(config, model=backbone)  # `model.model` is `backbone`
```

UltraBERT ([`UltrabertModel`], for example `dmis-lab/UltraBERT-150M`) works the same way, with `global_pooling="first"` and `prompt_style="plain"`. Its rotary positions and its alternating local and global attention layers keep the scores of a prompt the same alone, in a batch, and with left or right padding, including prompts longer than its local attention window. Its tokenizer has no `[CLS]` token, and its padding, beginning and end of sequence tokens are the same `<|endoftext|>`: the plain prompt starts with that token and ends with its `<|sep|>` separator. It has no causal language modeling head class, so the model has no `lm_head`. A checkpoint that wraps an UltraBERT in remote code (`model_type: "ultrabert"` with another architecture and an `auto_map`, such as a reranker whose weights are stored as `backbone.<name>`) does not load its weights into a native [`UltrabertModel`] with `AutoModel` (they would all be newly initialized, with only a warning): convert it with `convert_clef_weights_to_hf.py` (see [Converting checkpoints](#converting-checkpoints)), which strips the prefix, loads the backbone strictly and adds a new decision head.

## Backbones defined by their checkpoint's code

A backbone whose architecture is not in transformers, but in the Python files of its checkpoint (an `auto_map` in its `config.json`), loads only with `trust_remote_code=True`, and so does a Clef model built on it. Loading such a model runs that code only when the caller passes `trust_remote_code=True` itself (not another true value): without it, `ClefForDecision.from_pretrained`, [`AutoModelForDecision`] and `pipeline("decision")` raise a `ValueError` on a saved Clef model (or an adapter folder of one) that names the classes and where their code would come from, before anything of that code is downloaded or imported, whatever classes are already registered in the process, and without asking on stdin. Backbones of transformers load as before, with or without the flag, and the `auto_map` of a backbone of a model type of transformers is never followed (one without an `AutoModel` class is refused). A folder that is not a saved Clef model, such as the folder of the backbone itself, is read first by transformers' `AutoConfig` in [`AutoModelForDecision`] and `pipeline`, which asks on an interactive stdin when the flag is unset: pass `trust_remote_code=False` to refuse without asking.

Build the model around the backbone loaded with the flag. The backbone can be a base model (`AutoModel`) or, when the package only has a causal language model (`AutoModelForCausalLM`, and no `AutoModel` in its `auto_map`), that causal language model itself: then the lexical prior of the joint head reads its output embeddings, and nothing is added to it. Give `backbone_hidden_size` yourself, since such a config may have no `hidden_size`. The backbone is called with only the arguments its `forward` accepts. Its final hidden states are the output of the submodule named by `backbone_hidden_module` when that is set (for a model that returns no hidden states), else the `last_hidden_state` of its output, else the last of its `hidden_states`. The submodule is the one that [`~ClefForDecision.find_backbone_hidden_module`] finds by checking, in one small forward pass, which output gives the logits of the model through its output embeddings (it raises when none does). The field is saved, so loading never guesses.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, ClefConfig, ClefForDecision

backbone = AutoModelForCausalLM.from_pretrained("org/custom-model", trust_remote_code=True)
config = ClefConfig(
    text_config=backbone.config,
    backbone_hidden_size=backbone.get_output_embeddings().weight.shape[-1],
    backbone_hidden_module=ClefForDecision.find_backbone_hidden_module(backbone),
    head_type="cosine",
)
model = ClefForDecision(config, model=backbone)
# ... train, then
model.save_pretrained("path/to/clef")
AutoTokenizer.from_pretrained("org/custom-model").save_pretrained("path/to/clef")
```

`save_pretrained` keeps the `auto_map` of the backbone in the `text_config` of the saved config. For a backbone loaded from the Hub (its classes imported from the repository of its `_name_or_path`), its references name the repository (`org/custom-model--modeling_custom.CustomForCausalLM`): loading downloads the code of that repository again, from its default branch unless `code_revision` is given to `ClefForDecision.from_pretrained`. For a backbone loaded from a local folder, the code files are saved next to the model, which is then self-contained (upload the folder rather than `push_to_hub=True`, which is refused in that case). Load the saved model with the flag; the backbone is rebuilt with the class of its `auto_map` (`AutoModel` when there is one, else `AutoModelForCausalLM`), so build the model around a backbone of that class:

```python
from transformers import AutoModelForDecision, pipeline

model = AutoModelForDecision.from_pretrained("path/to/clef", trust_remote_code=True)
decide = pipeline("decision", model="path/to/clef", trust_remote_code=True)
```

A clef checkpoint in its original layout (`joint_head_config.json` next to the weights) on such a backbone is refused: build and save the model as above.

## Training

Cloudflare has not published Clef's training objective; the loss below and its defaults are a community recipe, not an official Cloudflare objective.

[`ClefForDecision`] computes a loss when it is given `labels`, one per question of the prompt: the index of the right option, `-100` to ignore the question, or a distribution over its options (shape `(num_questions, max_options)`). The output is a [`~models.clef.modeling_clef.ClefDecisionOutput`] whose float32 `logits` have the shape `(num_questions, max_options)`, padded with the minimum of their dtype where a question has fewer options, and whose `option_mask` marks the options that exist. The loss is the sum of three terms and of the RLCD term below, whose weights are fields of [`ClefConfig`]:

- the cross entropy of the softmax over the options of each question, averaged over the questions. `label_smoothing` (default 0.0) mixes the uniform distribution over the options into its target.
- `brier_weight` (default 0.0) times the squared distance of the probabilities to the target, summed over the options and averaged over the questions.
- `ordinal_weight` (default 0.0) times, on score questions only, the expected distance `|i - j|` between the options and the right one, divided by the number of options minus one and averaged over the score questions.

`forward` takes a padded batch of prompts: `input_ids` and `attention_mask` of shape `(batch_size, length)` (left or right padding), `marks` of shape `(batch_size, num_marks)` padded with `-1`, and `num_options` and `question_type_ids` of shape `(batch_size, max_questions)` padded with `0`. A question is real when it has options; the rows of `logits` and of `labels` are the real questions of every prompt in turn, so the loss of a batch is the mean over all its questions. Padding is masked out of every attention of the head and of the routing of the options, and each span is averaged in float32 over its own tokens only, in memory linear in the number of tokens of the spans: a prompt scores the same whatever the padding side and the other prompts of its batch, up to the rounding of the backbone.

Hard and soft labels give the same loss for the same target. A hard label must be `-100` or the index of an option of its question, and soft labels must have the shape of the logits: inconsistent labels, `option_mask` or `marks` raise a `ValueError` (checked on the device, at the cost of one synchronisation per call). Soft labels cannot ignore a question: a row of zeros still counts in the average, and feeds the Brier and RLCD terms. `forward` divides the loss by the number of questions that are not ignored; the loss function itself, `ClefForDecisionLoss` in `transformers.loss.loss_clef`, takes a `num_items_in_batch` that divides the summed cross entropy and Brier terms instead (`forward` does not take it).

Train with float32 parameters under autocast (`dtype=torch.float32` with `bf16=True` in [`TrainingArguments`]): a model loaded in bfloat16 keeps the joint head in bfloat16 too (the cosine head is always in float32).

Under the stock [`Trainer`], evaluation concatenates the outputs of its batches by padding the narrower ones: the `logits` with `-100` and the boolean `option_mask` with `True`, so metrics must read the options of each question from their counts (`num_options`), not from the concatenated `option_mask`. TRL's `DecisionTrainer` handles this.

The pooler of a BERT-style backbone gets no gradient, since the head reads the hidden states. Under DDP with activation checkpointing, where the [`Trainer`] sets `find_unused_parameters=False`, training then raises: freeze the pooler before training (`model.model.pooler.requires_grad_(False)`), which DDP then leaves out.

### RLCD

`rlcd_weight` (default 0.0) adds a policy term adapted from RLCD, the "Reinforcement Learning for Calibrated Decisions" objective of [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0). Zero-mean Gaussian noise of standard deviation `rlcd_sigma` (default 0.4) over the options of each question is added to the detached logits, `rlcd_samples` (default 32) times. Each noisy sample is rewarded by a strictly proper scoring rule against the target: the log score, plus `rlcd_spherical_weight` (default 0.75) times the spherical score, minus, on score questions, `rlcd_rps_weight` (default 1.0) times the ranked probability score. The advantage of each sample over the mean reward of the samples of its question weights the Gaussian log density of the sample, through which alone the gradient reaches the logits, and the term is averaged over the questions like the others. Ignored questions and padded options contribute nothing.

The term is random, so [`ClefForDecision`] applies it in training mode only: in eval mode the loss is the supervised loss, the same at every call. The defaults are a revised variant of Laya's objective, with more samples and advantages that are not divided by their standard deviation; `rlcd_weight=1.0, rlcd_normalize_advantage=True, rlcd_samples=4` reproduces the objective of Laya's notebook.

## Converting checkpoints

`convert_clef_weights_to_hf.py` converts a folder (or repo of the Hub) that ships a backbone with a head in its own format to a native Clef checkpoint: a `config.json` of `model_type: clef`, the weights with the head stored in float32 and the backbone (with its language modeling head) stored in its own precision, the tokenizer, `decision.json` and its template. Its decision config takes any number of options of every question type.

- A clef checkpoint, or a folder trained from one, with `joint_head.safetensors` and `joint_head_config.json` next to the weights of its backbone: the joint head, or the cosine head when its `head_type` is `cosine_span_v1`, with their sizes. The temperatures (noul, choice, score) of an `unsloth_decision_config.json`, times its `head_temperature`, become those of the decision config, and its `max_len` the `model_max_length` of the tokenizer. A folder with only one of the two head files is refused.
- A remote-code package around a native model, such as a reranker, whose weights are those of that model under a single prefix next to a task head: the prefix is stripped, the task head dropped (and named in the log), and a new head of `head_type` (`"joint"` by default, sizes from the arguments) initialized. The conversion is refused when no prefix, or more than one, holds exactly the weights of the native model. The backbone keeps the values and the precision of the source: one in mixed precision is refused.

The backbone is the bare model of the `model_type` of the `config.json` of the folder, loaded strictly, whatever classes the folder registers; no code of the folder is imported. A new head on a bidirectional backbone gets `global_pooling="first"` and `prompt_style="plain"`, on a causal one `"last"` and `"chatml"`. A folder that ships a trained head on a bidirectional backbone does not record how the head was trained, so its conversion requires both `--global_pooling` and `--prompt_style`, as they were in training; such a folder, and one with a `cosine_span_v1` head, does not load directly with `from_pretrained` and must be converted first. Backbones in float32, bfloat16 or float16 are converted (float64 is refused). The converted weights are staged next to the output folder, and the conversion needs up to about three times the size of a half-precision source in RAM.

```bash
python src/transformers/models/clef/convert_clef_weights_to_hf.py --source path/to/checkpoint --output path/to/clef
python src/transformers/models/clef/convert_clef_weights_to_hf.py --source ericlewis/ultrabert-150m-reranker \
    --output path/to/clef --head_type cosine --cosine_width 512
```

```py
from transformers import AutoModelForDecision
from transformers.models.clef.convert_clef_weights_to_hf import convert_clef_checkpoint

convert_clef_checkpoint("path/to/checkpoint", "path/to/clef")
model = AutoModelForDecision.from_pretrained("path/to/clef")
```

The `dtype` of the `config.json` of the converted checkpoint is the precision of its backbone. For a backbone in bfloat16:

- the file stores the backbone in bfloat16, exactly as the source, and the head in float32;
- a default `from_pretrained`, as above, loads the backbone in bfloat16 and casts the joint head to bfloat16 too: the float32 precision of the stored joint head is not used. The cosine head is always loaded and run in float32;
- to keep the stored float32 joint head over the bfloat16 backbone, load in float32, cast the backbone (and its language modeling head, when the model has one) back to bfloat16, which gives back its stored values exactly, and run the model under autocast, since the joint head does not cast the bfloat16 hidden states of the backbone itself. Loading holds the backbone in float32 (twice its bfloat16 memory) until it is cast back; then the model takes the bfloat16 backbone plus the float32 head. Under autocast, the matrix products of the head run in bfloat16 on its float32 weights. To compute the head in float32 too, keep the whole model in float32, at twice the memory of the backbone.

```py
import torch

model = AutoModelForDecision.from_pretrained("path/to/clef", dtype=torch.float32)
model.model.to(torch.bfloat16)
if hasattr(model, "lm_head"):
    model.lm_head.to(torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(**inputs)
```

## ClefConfig

[[autodoc]] ClefConfig

## ClefForDecision

[[autodoc]] ClefForDecision
    - forward
    - find_backbone_hidden_module

## ClefDecisionOutput

[[autodoc]] models.clef.modeling_clef.ClefDecisionOutput
