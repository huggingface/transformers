# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import argparse
import os
import shutil
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from transformers import OLMoConfig, OLMoForCausalLM, OLMoTokenizerFast
from transformers.models.olmo.configuration_olmo import ActivationType, BlockType, InitFnType, LayerNormType


USAGE = """
Sample usage:

```
python src/transformers/models/olmo/convert_olmo_weights_to_hf.py \
    --input_dir /path/to/downloaded/olmo/weights --output_dir /output/path --tokenizer_json_path /path/to/downloaded/olmo/tokenizer/file.json
```

Thereafter, models can be loaded via:

```py
from transformers import OLMoForCausalLM, OLMoTokenizerFast

model = OLMoForCausalLM.from_pretrained("/output/path")
tokenizer = OLMoTokenizerFast.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""


def _get_olmo_config(config_path: Path) -> OLMoConfig:
    # Default OLMo checkpoint settings. These may not match OLMoConfig, which defaults specifically to OLMo-7B.
    olmo_config = {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "mlp_ratio": 4,
        "mlp_hidden_size": None,
        "activation_type": ActivationType.swiglu,
        "block_type": BlockType.sequential,
        "alibi": False,
        "alibi_bias_max": 8.0,
        "rope": False,
        "rope_full_precision": True,
        "flash_attention": False,
        "attention_dropout": 0.1,
        "multi_query_attention": False,
        "attention_layer_norm": False,
        "residual_dropout": 0.1,
        "embedding_dropout": 0.1,
        "layer_norm_type": LayerNormType.default,
        "layer_norm_with_affine": True,
        "attention_layer_norm_with_affine": True,
        "max_sequence_length": 1024,
        "include_bias": True,
        "bias_for_layer_norm": None,
        "scale_logits": False,
        "vocab_size": 50304,
        "weight_tying": True,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "init_device": None,
        "init_fn": InitFnType.normal,
        "init_std": 0.02,
        "init_cutoff_factor": None,
        "precision": None,
    }

    olmo_config.update(yaml.safe_load(config_path.read_text())["model"])

    # OLMo considers 'vocab size' (the number of different tokens the tokenizer can produce)
    # and 'embedding size' (the number of embeddings) to be different and separately configurable.
    # HF does not. We set `vocab_size` to `embedding_size` and set `embedding_size` to `None`
    # so that HF OLMo models always have the same vocab and embedding sizes, thus avoiding this problem.
    if "embedding_size" in olmo_config:
        olmo_config["vocab_size"] = olmo_config["embedding_size"]
        del olmo_config["embedding_size"]

    return OLMoConfig(**olmo_config)


def _write_tokenizer(output_path: Path, config: OLMoConfig, input_tokenizer_path: Path) -> None:
    print(f"Saving a {OLMoTokenizerFast.__name__} to {output_path}.")

    base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))

    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id

    tokenizer = OLMoTokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
    )

    tokenizer.save_pretrained(output_path)


def write_model(
    output_path: str,
    input_base_path: str,
    safe_serialization: bool = True,
    input_tokenizer_path: str | None = None,
):
    os.makedirs(output_path, exist_ok=True)
    tmp_output_path = os.path.join(output_path, "tmp")
    os.makedirs(tmp_output_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.yaml"
    hf_config = _get_olmo_config(config_path)

    # `PreTrainedModel.from_pretrained` does not automatically initialize meta device tensors, so use cpu instead
    hf_config.init_device = "cpu" if hf_config.init_device == "meta" else hf_config.init_device

    hf_config.save_pretrained(tmp_output_path)

    model_path = Path(input_base_path) / "model.pt"
    shutil.copy(model_path, Path(tmp_output_path) / "pytorch_model.bin")

    print("Loading the checkpoint in a OLMo model.")
    model = OLMoForCausalLM.from_pretrained(tmp_output_path, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(output_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_output_path)

    if input_tokenizer_path is not None:
        _write_tokenizer(Path(output_path), hf_config, Path(input_tokenizer_path))


def main():
    parser = argparse.ArgumentParser(usage=USAGE)
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo config weights (config.yaml and model.pt)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        help="Location of the tokenizer json file used by OLMo's original tokenizer. When supplied, this script also creates HF tokenizer files.",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    write_model(
        output_path=args.output_dir,
        input_base_path=args.input_dir,
        input_tokenizer_path=args.tokenizer_json_path,
        safe_serialization=args.safe_serialization if args.safe_serialization is not None else True,
    )


if __name__ == "__main__":
    main()
