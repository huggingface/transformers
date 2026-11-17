# Copyright 2026 The KLM authors and The HuggingFace Team. All rights reserved.
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
Convert a KLM checkpoint from its original layout (a Hub repository with its own modeling code) to the Transformers one.

Sample usage:

```bash
python src/transformers/models/klm/convert_klm_weights_to_hf.py \
    --input_dir MinimaLabs/KLM-1.6 \
    --output_dir /output/path
```

Thereafter, models can be loaded via:

```python
from transformers import KlmForCausalLM, AutoTokenizer

model = KlmForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import AutoTokenizer, GenerationConfig, KlmConfig, KlmForCausalLM
from transformers.models.klm.conversion_klm import (
    ORIGINAL_TO_CONVERTED_KEY_MAPPING as ORIGINAL_TO_CONVERTED_KEY_MAPPING,
)
from transformers.models.klm.conversion_klm import (
    convert_state_dict,
)


# Original config names that differ from the Transformers ones
ORIGINAL_TO_CONVERTED_CONFIG_MAPPING = {
    "gdn_heads": "linear_num_heads",
    "gdn_head_dim": "linear_head_dim",
    "gdn_norm_eps": "linear_norm_eps",
    "decay_floor": "linear_decay_floor",
    "n_bigrams": "num_bigrams",
    "n_trigrams": "num_trigrams",
    "ngram_dim": "ngram_embedding_size",
}
SHARED_CONFIG_KEYS = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rms_norm_eps",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "tie_word_embeddings",
]


def convert_config(original_config: dict) -> KlmConfig:
    config = {key: original_config[key] for key in SHARED_CONFIG_KEYS if key in original_config}
    for original_key, key in ORIGINAL_TO_CONVERTED_CONFIG_MAPPING.items():
        if original_key in original_config:
            config[key] = original_config[original_key]
    if "rope_theta" in original_config:
        config["rope_parameters"] = {"rope_type": "default", "rope_theta": original_config["rope_theta"]}
    if "num_hidden_layers" in original_config:
        attention_every = original_config.get("attention_every", 4)
        config["layer_types"] = [
            "full_attention" if layer_idx % attention_every == attention_every - 1 else "linear_attention"
            for layer_idx in range(original_config["num_hidden_layers"])
        ]
    return KlmConfig(**config)


def convert(input_dir: str, output_dir: str):
    if not Path(input_dir).is_dir():
        input_dir = snapshot_download(input_dir, allow_patterns=["*.json", "*.safetensors"])
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    config = convert_config(json.loads((input_dir / "config.json").read_text()))
    state_dict = convert_state_dict(load_file(input_dir / "model.safetensors"), config)

    model = KlmForCausalLM(config)
    # `lm_head.weight` is tied to the token embeddings, every other weight has to be there
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys != ["lm_head.weight"] or unexpected_keys:
        raise ValueError(f"The conversion left missing keys {missing_keys} and unexpected keys {unexpected_keys}.")
    model.tie_weights()
    model.save_pretrained(output_dir, save_original_format=False)

    AutoTokenizer.from_pretrained(input_dir).save_pretrained(output_dir)
    if (input_dir / "generation_config.json").exists():
        GenerationConfig.from_pretrained(input_dir).save_pretrained(output_dir)
    print(f"Saved the converted model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KLM weights to the Transformers format.")
    parser.add_argument(
        "--input_dir", required=True, help="Local folder or Hub repository of the original checkpoint."
    )
    parser.add_argument("--output_dir", required=True, help="Where to write the converted model and tokenizer.")
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)
