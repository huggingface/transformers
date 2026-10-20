# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Converts a JEV checkpoint (autotrust/JEV-9B, JEV-27B, JEV-27B-VL, JEV-35B, JEV-Gemma4-26B-A4B) to a JEV head, which reads
the checkpoint from its own repo. System 1 reads the log-probabilities of the slot tokens at the last prompt token, adds
the bias of each slot and divides by the temperature of the question type: the log-softmax normalizer is the same for
every slot, so the head is the `lm_head` rows of the slot tokens (with their LoRA) and their bias, or the trained head
of `head.safetensors`. The LoRA of the backbone (`adapter_vllm/` or `adapter/`) is saved in `backbone/`, where
`from_pretrained` reads it for the backbone.

    python convert_jev_to_hf.py --repo_id autotrust/JEV-27B --output_dir JEV-27B
"""

import argparse
import json
import os

import torch
from huggingface_hub import file_exists, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from transformers import AutoConfig, AutoProcessor, JevConfig
from transformers.decision import DecisionConfig


# The `bare-v1` prompt of System 1 (`serve_decide.py`), read before `<|mark|>`
TEMPLATE = """
{%- macro s(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{%- macro st(x) -%}
{%- if x is string %}{{ x }}{% elif x is mapping %}{{ x|tojson }}{% else -%}
{%- for p in x -%}
{%- if p is string %}{{ p }}
{%- elif p is mapping and ("image" in p or p.get("type") == "image_url") %}{{ raise_exception("bare-v1 text template: images are not supported") }}
{%- elif p is mapping and p.get("type") == "text" %}{{ p["text"] }}
{%- else %}{{ p|tojson }}{% endif -%}
{%- endfor -%}
{%- endif -%}
{%- endmacro -%}
{{- "[kind] " ~ question.type ~ "\\n[state] " ~ st(state) ~ "\\n[question] " ~ s(question.instructions) ~ "\\n[options]\\n" -}}
{%- for o in question.options -%}
{%- set d = question.descriptions[loop.index0] -%}
{%- if question.type == "choice" %}{{ question.labels[loop.index0] ~ ") " ~ o ~ (": " ~ s(d) if d is not none and d != "" else "") }}
{%- else %}{{ o }}{% endif -%}
{{- "\\n" if not loop.last else "" -}}
{%- endfor -%}
{{- "\\n[decision]:<|mark|>" -}}
"""


def load_json(repo_id: str, filename: str) -> dict:
    with open(hf_hub_download(repo_id, filename), encoding="utf-8") as file:
        return json.load(file)


def convert(repo_id: str, output_dir: str):
    backbone_config = AutoConfig.from_pretrained(repo_id)
    text_config = backbone_config.get_text_config()
    if file_exists(repo_id, "head.safetensors"):  # a trained float32 head, on the backbone with the LoRA of `adapter/`
        spec, adapter = load_json(repo_id, "judge_config.json"), "adapter"
        state_dict = load_file(hf_hub_download(repo_id, "head.safetensors"))
    else:  # the `lm_head` rows of the slot tokens, merged or with the LoRA of `adapter_vllm/`
        spec = load_json(repo_id, "adapter_vllm/decision_head.json")
        adapter = None if spec.get("merged") else "adapter_vllm"
        weight_map = load_json(repo_id, "model.safetensors.index.json")["weight_map"]
        name = next(key for key in weight_map if key.endswith("lm_head.weight"))
        with safe_open(hf_hub_download(repo_id, weight_map[name]), "pt") as file:
            weight = file.get_tensor(name)[spec["verbalizer_ids"]].float()
        state_dict = {"proj.weight": weight, "proj.bias": torch.tensor(spec["bias"], dtype=torch.float32)}
    if adapter is not None:
        lora_config = load_json(repo_id, f"{adapter}/adapter_config.json")
        lora = load_file(hf_hub_download(repo_id, f"{adapter}/adapter_model.safetensors"))
        if (lora_a := lora.pop("base_model.model.lm_head.lora_A.weight", None)) is not None:
            lora_b = lora.pop("base_model.model.lm_head.lora_B.weight")[spec["verbalizer_ids"]]
            scaling = lora_config["lora_alpha"] / lora_config["r"]
            state_dict["proj.weight"] += scaling * lora_b.float() @ lora_a.float()
        # The LoRA of the `ForXxx` model, on the backbone of the head: its base model
        lora = {key.replace("base_model.model.model.", "base_model.model.", 1): value for key, value in lora.items()}
        targets = lora_config["target_modules"]
        lora_config["target_modules"] = (
            targets.removeprefix("model\\.") if isinstance(targets, str) else [t for t in targets if t != "lm_head"]
        )
        lora_config["base_model_name_or_path"] = repo_id
        os.makedirs(f"{output_dir}/backbone", exist_ok=True)
        with open(f"{output_dir}/backbone/adapter_config.json", "w", encoding="utf-8") as file:
            json.dump(lora_config, file, indent=2)
        save_file(lora, f"{output_dir}/backbone/adapter_model.safetensors", metadata={"format": "pt"})

    slots, ranges = spec["slots"], spec["slots"]["ranges"]
    JevConfig(
        hidden_size=text_config.hidden_size,
        num_verbalizers=slots["num_slots"],
        offsets={question_type: start for question_type, (start, _) in ranges.items()},
        softcap=spec.get("softcap"),
        backbone_config=backbone_config,
        # System 1 adds the float32 bias to float32 log-probabilities
        keep_in_fp32_modules=["^proj"],
        # The dtype of the backbone, which some checkpoints only give in their text config: without it the head would
        # load in the dtype of its float32 weights
        dtype=text_config.dtype,
    ).save_pretrained(output_dir)
    save_file(state_dict, f"{output_dir}/model.safetensors", metadata={"format": "pt"})

    AutoProcessor.from_pretrained(repo_id).save_pretrained(output_dir)
    verbalizers = slots["verbalizers"]
    DecisionConfig(
        labels={question_type: verbalizers[start:end] for question_type, (start, end) in ranges.items()},
        temperature=load_json(repo_id, "calibration.json")["per_kind"],
        # Gemma reads the prompt after `<bos>`
        chat_template=("<bos>" if "gemma" in backbone_config.model_type else "") + TEMPLATE,
        noul_options=["false", "true"],
    ).save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="autotrust/JEV-27B")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert(args.repo_id, args.output_dir)
