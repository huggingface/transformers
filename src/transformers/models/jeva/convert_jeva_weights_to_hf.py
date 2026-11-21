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

"""Convert a Jeva checkpoint to the Transformers format."""

import argparse
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AutoModel, AutoTokenizer, JevaConfig, JevaForDecision
from transformers.utils import is_peft_available


if is_peft_available():
    from peft import PeftModel


@torch.no_grad()
def convert_checkpoint(checkpoint: str, output_dir: str):
    path = Path(checkpoint)
    if not path.is_dir():
        path = Path(snapshot_download(checkpoint))
    settings = yaml.safe_load((path / "config.yaml").read_text())["model"]
    encoder = AutoModel.from_pretrained(path / "backbone", dtype=torch.float32)
    adapter_path = path / "backbone_adapter"
    if adapter_path.is_dir():
        if not is_peft_available():
            raise ImportError("Install `peft` to convert a checkpoint with an encoder adapter")
        with safe_open(adapter_path / "adapter_model.safetensors", framework="pt") as weights:
            wraps_language_model = any(key.startswith("base_model.model.language_model.") for key in weights.keys())
        if encoder.config.model_type in {"qwen3_5", "qwen3_5_moe"} and not wraps_language_model:
            encoder = encoder.language_model
        encoder = PeftModel.from_pretrained(encoder, adapter_path).merge_and_unload()
    if encoder.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        encoder = encoder.language_model
    config = JevaConfig(
        text_config=encoder.config,
        **{key: value for key, value in settings.items() if key in JevaConfig.__dataclass_fields__},
    )
    model = JevaForDecision(config, model=encoder)
    head = torch.load(path / "head.pt", map_location="cpu", weights_only=True)
    for name in ("context_proj", "option_proj", "core"):
        state = {key: value.reshape(1) if value.ndim == 0 else value for key, value in head[name].items()}
        model.get_submodule(name).load_state_dict(state)
    model.save_pretrained(output_dir)
    tokenizer_path = path / "tokenizer" if (path / "tokenizer").is_dir() else path / "backbone"
    AutoTokenizer.from_pretrained(tokenizer_path).save_pretrained(output_dir)
    model.decision_config.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="LIQIIIII/Jeva-4B")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output_dir)
