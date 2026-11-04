# Copyright 2026 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSpark speculative decoding with Qwen3-8B and the `deepseek-ai/dspark_qwen3_8b_block7` drafter.

DSpark (https://huggingface.co/papers/2607.05147) drafts a whole block of tokens in one forward of a DFlash backbone,
then adds a sequential Markov bias token by token. The drafter is not a Transformers model: it is built in this script
and passed to `generate` with `speculation_type="dspark"`. It must follow the `DSparkTokenCandidateGenerator` contract:

- `drafter.backbone`: a DFlash backbone (here `MuseGlimmerAssistantModel`, which has the same layout), whose forward
  returns `block_size` rows, row j drafting the token right after position `anchor_position + j`
- `drafter.sequential_block(prev_token_ids: [batch]) -> [batch, vocab]`: the Markov bias added to the logits of the
  next drafted token
- `drafter.config`: the backbone config, with `block_size`, `mask_token_id`, `target_layer_ids` and `head_type`

The drafter reuses the target model's embeddings and LM head, so the copies shipped in the checkpoint are dropped.
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.muse_glimmer_assistant import MuseGlimmerAssistantConfig, MuseGlimmerAssistantModel
from transformers.models.muse_glimmer_assistant.modeling_muse_glimmer_assistant import (
    MuseGlimmerAssistantPreTrainedModel,
)


TARGET_ID = "Qwen/Qwen3-8B"
DRAFTER_ID = "deepseek-ai/dspark_qwen3_8b_block7"


class DSparkMarkovHead(nn.Module):
    """Low-rank learned bigram bias: `markov_w2(markov_w1[prev_token])`."""

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def forward(self, prev_token_ids: torch.LongTensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev_token_ids))


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance predictor on `[hidden_state ; markov_w1[prev_token]]` (loaded, but not used yet)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)


class DSparkDrafter(MuseGlimmerAssistantPreTrainedModel):
    def __init__(self, config: MuseGlimmerAssistantConfig):
        super().__init__(config)
        self.backbone = MuseGlimmerAssistantModel(config)
        self.sequential_block = DSparkMarkovHead(config.vocab_size, config.markov_rank)
        self.confidence_head = DSparkConfidenceHead(config.hidden_size + config.markov_rank)
        self.post_init()


def load_drafter(drafter_id: str, device: torch.device, dtype: torch.dtype) -> DSparkDrafter:
    with open(hf_hub_download(drafter_id, "config.json"), encoding="utf-8") as f:
        ckpt_config = json.load(f)

    config = MuseGlimmerAssistantConfig(
        hidden_size=ckpt_config["hidden_size"],
        intermediate_size=ckpt_config["intermediate_size"],
        num_hidden_layers=ckpt_config["num_hidden_layers"],
        num_attention_heads=ckpt_config["num_attention_heads"],
        num_key_value_heads=ckpt_config["num_key_value_heads"],
        head_dim=ckpt_config["head_dim"],
        rms_norm_eps=ckpt_config["rms_norm_eps"],
        rope_parameters=ckpt_config["rope_parameters"],
        max_position_embeddings=ckpt_config["max_position_embeddings"],
        layer_types=ckpt_config["layer_types"],
        hidden_act=ckpt_config["hidden_act"],
        bos_token_id=ckpt_config["bos_token_id"],
        eos_token_id=ckpt_config["eos_token_id"],
        pad_token_id=ckpt_config["pad_token_id"],
        block_size=ckpt_config["block_size"],
        mask_token_id=ckpt_config["mask_token_id"],
        target_layer_ids=ckpt_config["target_layer_ids"],
    )
    # DSpark-only fields
    config.vocab_size = ckpt_config["vocab_size"]
    config.markov_rank = ckpt_config["markov_rank"]
    config.head_type = ckpt_config["markov_head_type"]

    state_dict = load_file(hf_hub_download(drafter_id, "model.safetensors"))
    # Frozen copies of the target's embeddings and LM head: the candidate generator uses the target's own
    state_dict.pop("embed_tokens.weight")
    state_dict.pop("lm_head.weight")
    renamed = {}
    for key, value in state_dict.items():
        if key.startswith("markov_head."):
            key = key.replace("markov_head.", "sequential_block.", 1)
        elif key.startswith("fc."):
            key = key.replace("fc.", "backbone.encoder.fc.", 1)
        elif key.startswith("hidden_norm."):
            key = key.replace("hidden_norm.", "backbone.encoder.output_norm_enc.", 1)
        elif not key.startswith("confidence_head."):
            key = "backbone." + key
        renamed[key] = value

    drafter = DSparkDrafter(config)
    drafter.load_state_dict(renamed, strict=True)
    return drafter.to(device=device, dtype=dtype).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Write a bash one-liner that counts lines of Python in a repo.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    model = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.bfloat16, device_map="auto")
    drafter = load_drafter(DRAFTER_ID, device=model.device, dtype=torch.bfloat16)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Assisted decoding runs exactly one target forward per step, so counting them gives the acceptance
    target_forwards = []
    hook = model.register_forward_hook(lambda module, args, output: target_forwards.append(1))

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        assistant_model=drafter,
        speculation_type="dspark",
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.perf_counter() - start
    hook.remove()

    num_new_tokens = outputs.shape[1] - input_len
    print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True))
    print(f"\n{num_new_tokens} tokens in {elapsed:.2f}s ({num_new_tokens / elapsed:.1f} tokens/s)")

    # The prefill yields 1 token and verifies no drafts, every other step yields its accepted drafts + 1 bonus token.
    # A step cut short by EOS or `max_new_tokens` makes this slightly pessimistic.
    num_steps = len(target_forwards) - 1
    if num_steps > 0:
        mean_accepted = (num_new_tokens - 1) / num_steps - 1
        print(f"mean accepted drafts per step: {mean_accepted:.2f} / {drafter.config.block_size} ({num_steps} steps)")
