"""Text-only classes loaded directly from the joint composite checkpoint.

Covers all three public text-only classes:
  1) `Apertus1p5TextConfig.from_pretrained(composite)` extracts the nested `text_config` via
     `base_config_key` (including the pruned-head marker `output_vocab_size`);
  2) `Apertus1p5TextForCausalLM` loads straight from the composite: the `model.language_model.*` weight keys
     are remapped by the `PrefixChange` entry in the conversion mapping, and the vision/audio tokenizer
     weights surface as ignorable "unexpected" keys. Checks: clean load, pruned head size, raw-text and
     chat generation (greedy and beam search), and that generated ids always stay below the pruned cutoff,
     since the multimodal rows are removed from the output layer;
  3) the bare `Apertus1p5TextModel` backbone (no head) loads the same way and returns hidden states.
"""

import os
import warnings

import torch

from transformers import Apertus1p5TextConfig, Apertus1p5TextForCausalLM, Apertus1p5TextModel, AutoTokenizer

CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "/Users/rkre/swissai_repos/material/Apertus-1.5-8B-composite-hf")

# --- 1) config extraction from the composite ---------------------------------------------------------
config = Apertus1p5TextConfig.from_pretrained(CHECKPOINT)
assert type(config).__name__ == "Apertus1p5TextConfig"
print(f"[OK] config extracted from the composite: vocab_size {config.vocab_size}, "
      f"output_vocab_size {config.output_vocab_size}, hidden_size {config.hidden_size}")

print("loading text backbone from the composite (bf16, CPU) ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the load report lists the composite's tokenizer weights as unexpected
    model, info = Apertus1p5TextForCausalLM.from_pretrained(
        CHECKPOINT, dtype=torch.bfloat16, output_loading_info=True
    )
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# --- clean load: nothing missing, only the vision/audio tokenizer weights are unexpected ------------
assert not info["missing_keys"] and not info["mismatched_keys"]
assert all(key.startswith(("model.vision_tokenizer.", "model.audio_tokenizer.")) for key in info["unexpected_keys"])
print(f"[OK] clean load: 0 missing, {len(info['unexpected_keys'])} unexpected (all vision/audio tokenizer keys)")

# --- pruned output layer -----------------------------------------------------------------------------
config = model.config
head_rows = model.lm_head.out_features
assert head_rows == (config.output_vocab_size or config.vocab_size)
print(f"[OK] pruned head: {head_rows} rows (vocab_size {config.vocab_size}, "
      f"output_vocab_size {config.output_vocab_size}); the model cannot emit multimodal ids")

# --- raw-text continuation (base-model style) --------------------------------------------------------
inputs = tokenizer("The capital of Switzerland is", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
completion = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
assert "Bern" in completion, completion
print(f"[OK] raw-text continuation: {completion!r}")

# --- chat generation, greedy and beam search ---------------------------------------------------------
messages = [{"role": "user", "content": "Name three Swiss cities, comma separated."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
for label, generate_kwargs in (("greedy", {"do_sample": False}), ("beam-2", {"num_beams": 2, "do_sample": False})):
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, **generate_kwargs)
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    assert int(new_ids.max()) < head_rows, "generated ids must stay below the pruned cutoff"
    print(f"[OK] chat {label} (all ids < {head_rows}): {completion!r}")

# --- 3) bare backbone: Apertus1p5TextModel (no head) returns hidden states ---------------------------
del model  # free the CausalLM before loading the backbone again
print("loading bare Apertus1p5TextModel from the composite ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # lm_head and the tokenizer weights are unexpected for the bare model
    backbone = Apertus1p5TextModel.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).eval()
token_ids = tokenizer("The Alps are", return_tensors="pt")
with torch.no_grad():
    hidden_states = backbone(**token_ids).last_hidden_state
assert hidden_states.shape == (1, token_ids["input_ids"].shape[1], config.hidden_size)
assert bool(torch.isfinite(hidden_states.float()).all())
print(f"[OK] bare backbone forward: last_hidden_state {tuple(hidden_states.shape)}, finite")

print("\nALL TEXT-FROM-COMPOSITE CHECKS PASSED")
