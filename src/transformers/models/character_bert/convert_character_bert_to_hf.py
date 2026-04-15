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
"""Convert CharacterBERT checkpoints to a canonical Hugging Face format.

The script accepts either:
- a local directory containing CharacterBERT files, or
- a Hub repo id (for example: ``helboukkouri/character-bert-base-uncased``).

It rewrites legacy config fields to the current schema via model loading/saving,
exports a normalized checkpoint, and saves tokenizer assets.

Example:

```bash
python src/transformers/models/character_bert/convert_character_bert_to_hf.py \
  --source /path/to/character-bert-base-uncased \
  --output_dir /tmp/character-bert-hf \
  --verify
```
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from transformers import CharacterBertForMaskedLM, CharacterBertModel, CharacterBertTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def _resolve_source_path(source: str) -> Path:
    source_path = Path(source)
    if source_path.exists():
        return source_path

    logger.info("`%s` does not exist locally. Downloading from the Hub.", source)
    local_path = snapshot_download(
        repo_id=source,
        allow_patterns=[
            "*.bin",
            "*.safetensors",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "mlm_vocab.txt",
            "vocab.txt",
        ],
    )
    return Path(local_path)


def _copy_optional_mlm_vocab(source_path: Path, output_path: Path) -> None:
    source_vocab = source_path / "mlm_vocab.txt"
    if source_vocab.is_file():
        shutil.copy2(source_vocab, output_path / "mlm_vocab.txt")


def _verify_masked_lm(output_path: Path) -> None:
    import torch

    model = CharacterBertForMaskedLM.from_pretrained(output_path)
    tokenizer = CharacterBertTokenizer.from_pretrained(output_path)
    model.eval()

    inputs = tokenizer("paris is the capital of [MASK].", return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    mask_index = tokens.index(tokenizer.mask_token)

    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_index]

    top_index = int(logits.argmax())
    logger.info("Verification passed. Top MLM index for sample sentence: %d", top_index)


def convert_character_bert_to_hf(
    source: str,
    output_dir: str,
    model_type: str = "masked_lm",
    safe_serialization: bool = True,
    verify: bool = False,
) -> None:
    source_path = _resolve_source_path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CharacterBERT checkpoint from %s", source_path)
    if model_type == "masked_lm":
        model = CharacterBertForMaskedLM.from_pretrained(source_path)
    elif model_type == "base":
        model = CharacterBertModel.from_pretrained(source_path)
    else:
        raise ValueError(f"Unsupported `model_type={model_type}`. Expected `masked_lm` or `base`.")

    tokenizer = CharacterBertTokenizer.from_pretrained(source_path)

    logger.info("Saving converted model to %s", output_path)
    model.save_pretrained(output_path, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_path)
    _copy_optional_mlm_vocab(source_path, output_path)

    if verify and model_type == "masked_lm":
        _verify_masked_lm(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Local checkpoint directory or Hub repo id containing CharacterBERT files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="masked_lm",
        choices=["masked_lm", "base"],
        help="Architecture to export.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If set, saves PyTorch `.bin` files instead of `safetensors`.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a small masked-LM smoke test after conversion.",
    )
    args = parser.parse_args()

    convert_character_bert_to_hf(
        source=args.source,
        output_dir=args.output_dir,
        model_type=args.model_type,
        safe_serialization=not args.no_safe_serialization,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
