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
Converts a JEV checkpoint whose System 1 is merged into its weights (autotrust/JEV-35B) to a JEV head, which reads the
checkpoint from its own repo. System 1 reads the log-probabilities of the slot tokens at the last prompt token, adds the
bias of each slot and divides by the temperature of the question type: the log-softmax normalizer is the same for every
slot, so the head is the `lm_head` rows of the slot tokens and their bias.

    python convert_jev_to_hf.py --repo_id autotrust/JEV-35B --output_dir JEV-35B
"""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

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


def convert(repo_id: str, output_dir: str):
    with open(hf_hub_download(repo_id, "adapter_vllm/decision_head.json"), encoding="utf-8") as file:
        head = json.load(file)
    if not head.get("merged"):
        raise ValueError(f"System 1 of {repo_id} is a LoRA of its backbone (`adapter_vllm/`): merge it first.")
    with open(hf_hub_download(repo_id, "calibration.json"), encoding="utf-8") as file:
        temperature = json.load(file)["per_kind"]
    slots, ranges = head["slots"], head["slots"]["ranges"]

    backbone_config = AutoConfig.from_pretrained(repo_id)
    JevConfig(
        hidden_size=backbone_config.get_text_config().hidden_size,
        num_verbalizers=slots["num_slots"],
        offsets={question_type: start for question_type, (start, _) in ranges.items()},
        softcap=head.get("softcap"),
        backbone_config=backbone_config,
        # System 1 adds the float32 bias to float32 log-probabilities
        keep_in_fp32_modules=["^proj"],
        dtype=backbone_config.dtype,
    ).save_pretrained(output_dir)

    with open(hf_hub_download(repo_id, "model.safetensors.index.json"), encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]
    name = next(key for key in weight_map if key.endswith("lm_head.weight"))
    with safe_open(hf_hub_download(repo_id, weight_map[name]), "pt") as file:
        weight = file.get_tensor(name)[head["verbalizer_ids"]].float()
    state_dict = {"proj.weight": weight, "proj.bias": torch.tensor(head["bias"], dtype=torch.float32)}
    save_file(state_dict, f"{output_dir}/model.safetensors", metadata={"format": "pt"})

    AutoProcessor.from_pretrained(repo_id).save_pretrained(output_dir)
    verbalizers = slots["verbalizers"]
    DecisionConfig(
        labels={question_type: verbalizers[start:end] for question_type, (start, end) in ranges.items()},
        temperature=temperature,
        chat_template=TEMPLATE,
        noul_options=["false", "true"],
    ).save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="autotrust/JEV-35B")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert(args.repo_id, args.output_dir)
