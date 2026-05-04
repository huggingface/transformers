# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import gc
import logging
import re
import tarfile
import tempfile
from collections import UserString
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE, WordLevel

from transformers import (
    ConformerCTCConfig,
    ConformerEncoder,
    ConformerEncoderConfig,
    ConformerFeatureExtractor,
    ConformerForCTC,
    ConformerProcessor,
    ConformerTokenizer,
)
from transformers.convert_slow_tokenizer import SentencePieceExtractor
from transformers.utils import cached_file


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class SetIfExists(UserString):
    pass


def map_config(
    config: dict[str, Any],
    mapping: dict[str, str | Callable[[dict[str, Any]], Any] | SetIfExists],
) -> dict[str, Any]:
    mapped_config = {}
    for destination, source in mapping.items():
        if callable(source):
            mapped_config[destination] = source(config)
        elif isinstance(source, SetIfExists):
            source = str(source)
            if source in config:
                mapped_config[destination] = config[source]
        else:
            mapped_config[destination] = config[source]

    return mapped_config


def map_state_dict(state_dict: dict[str, torch.Tensor], mapping: dict[str, str | None]) -> dict[str, torch.Tensor]:
    def mapped_key_of(key: str) -> str | None:
        for pattern, replacement in mapping.items():
            if re.search(pattern, key) is None:
                continue

            if replacement is None:
                return None

            key = re.sub(pattern, replacement, key)

        return key

    mapped_state_dict = {}
    for key, value in state_dict.items():
        mapped_key = mapped_key_of(key)
        if mapped_key is None:
            continue

        mapped_state_dict[mapped_key] = value

    return mapped_state_dict


@dataclass
class NeMoCheckpoint:
    root_path: Path
    config_path: Path
    weight_path: Path
    config: dict[str, Any]

    def path_to(self, component: str) -> Path:
        component = component.removeprefix("nemo:").strip()
        return self.root_path / component

    def as_feature_extractor(self) -> ConformerFeatureExtractor:
        return ConformerFeatureExtractor(
            **map_config(
                self.config["preprocessor"],
                {
                    "feature_size": "features",
                    "sampling_rate": "sample_rate",
                    "hop_length": lambda config: int(config["window_stride"] * config["sample_rate"]),
                    "n_fft": "n_fft",
                    "win_length": lambda config: int(config["window_size"] * config["sample_rate"]),
                    "preemphasis": SetIfExists("preemph"),
                    "padding_value": "pad_value",
                },
            )
        )

    def as_tokenizer(self) -> ConformerTokenizer:
        def create_bpe_tokenizer() -> ConformerTokenizer:
            config = self.config["tokenizer"]

            assert "model_path" in config
            extractor = SentencePieceExtractor(str(self.path_to(config["model_path"])))
            extracted = cast(dict[str, Any], extractor.extract(BPE))

            tokenizer = Tokenizer(
                BPE(vocab=extracted["vocab"], merges=extracted["merges"], unk_token=UNK_TOKEN, fuse_unk=True)
            )
            tokenizer.add_special_tokens([AddedToken(UNK_TOKEN), AddedToken(PAD_TOKEN)])

            return ConformerTokenizer(tokenizer_object=tokenizer, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)

        def create_word_level_tokenizer() -> ConformerTokenizer:
            config = self.config["decoder"]

            assert "vocabulary" in config
            vocabulary = {word: index for index, word in enumerate(config["vocabulary"])}

            tokenizer = Tokenizer(WordLevel(vocabulary))
            tokenizer.add_special_tokens([AddedToken("<pad>")])

            return ConformerTokenizer(tokenizer_object=tokenizer, pad_token=PAD_TOKEN)

        config = self.config.get("tokenizer", {})

        if not config:
            return create_word_level_tokenizer()
        elif config.get("type") == "bpe":
            return create_bpe_tokenizer()
        else:
            raise ValueError("Unknown tokenizer")

    def as_processor(self) -> ConformerProcessor:
        feature_extractor = self.as_feature_extractor()
        tokenizer = self.as_tokenizer()
        return ConformerProcessor(feature_extractor, tokenizer)

    def as_encoder_config(self) -> ConformerEncoderConfig:
        return ConformerEncoderConfig(
            **map_config(
                self.config["encoder"],
                {
                    "hidden_size": "d_model",
                    "num_hidden_layers": "n_layers",
                    "num_attention_heads": "n_heads",
                    "intermediate_size": lambda config: int(config["d_model"] * config["ff_expansion_factor"]),
                    "attention_bias": SetIfExists("use_bias"),
                    "convolution_bias": SetIfExists("use_bias"),
                    "conv_kernel_size": "conv_kernel_size",
                    "subsampling_factor": "subsampling_factor",
                    "subsampling_conv_channels": lambda config: config["subsampling_conv_channels"]
                    if config["subsampling_conv_channels"] != -1
                    else config["d_model"],
                    "num_mel_bins": "feat_in",
                    "dropout": "dropout",
                    "dropout_positions": "dropout_emb",
                    "activation_dropout": "dropout",
                    "attention_dropout": "dropout_att",
                    "max_position_embeddings": "pos_emb_max_len",
                    "scale_input": "xscaling",
                },
            )
        )

    def as_ctc_config(self) -> ConformerCTCConfig:
        tokenizer = self.as_tokenizer()
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define a pad token for CTC decoding.")

        assert isinstance(tokenizer.pad_token_id, int)

        return ConformerCTCConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            encoder_config=self.as_encoder_config(),
        )

    def as_encoder_model(self) -> ConformerEncoder:
        model = ConformerEncoder(self.as_encoder_config())
        model.load_state_dict(
            map_state_dict(
                torch.load(self.weight_path, map_location="cpu", weights_only=True),
                {
                    r"^preprocessor\.": None,
                    r"^encoder\.pre_encode\.conv\.": "subsampling.layers.",
                    r"^encoder\.pre_encode\.out\.": "subsampling.linear.",
                    r"^encoder\.layers\.(\d+)\.": r"layers.\1.",
                    r"^decoder\.": None,
                    r"\.conv\.batch_norm\.": ".conv.norm.",
                    r"\.linear_([qkv])\.": r".\1_proj.",
                    r"\.linear_out\.": ".o_proj.",
                    r"\.linear_pos\.": ".relative_k_proj.",
                    r"\.pos_bias_([uv])$": r".bias_\1",
                },
            ),
            strict=True,
            assign=True,
        )

        return model

    def as_ctc_model(self) -> ConformerForCTC:
        model = ConformerForCTC(self.as_ctc_config())
        model.load_state_dict(
            map_state_dict(
                torch.load(self.weight_path, map_location="cpu", weights_only=True),
                {
                    r"^preprocessor\.": None,
                    r"^encoder\.pre_encode\.conv\.": "encoder.subsampling.layers.",
                    r"^encoder\.pre_encode\.out\.": "encoder.subsampling.linear.",
                    r"^encoder\.layers\.(\d+)\.": r"encoder.layers.\1.",
                    r"^decoder\.decoder_layers\.0\.": "ctc_head.",
                    r"\.conv\.batch_norm\.": ".conv.norm.",
                    r"\.linear_([qkv])\.": r".\1_proj.",
                    r"\.linear_out\.": ".o_proj.",
                    r"\.linear_pos\.": ".relative_k_proj.",
                    r"\.pos_bias_([uv])$": r".bias_\1",
                },
            ),
            strict=True,
            assign=True,
        )

        return model


def resolve_checkpoint_path(path_or_repository_id: str) -> Path:
    checkpoint_path = Path(path_or_repository_id)

    if checkpoint_path.exists():
        if checkpoint_path.is_dir():
            for path in sorted(checkpoint_path.glob("*.nemo")):
                return path
        elif checkpoint_path.suffix == ".nemo":
            return checkpoint_path
    else:
        repository = path_or_repository_id
        filename = f"{checkpoint_path.name}.nemo"

        path = cached_file(repository, filename)
        if path is not None:
            return Path(path)

    raise FileNotFoundError(f"Could not resolve checkpoint path from {path_or_repository_id}")


def extract_nemo_checkpoint(checkpoint_path: Path, output_path: Path) -> NeMoCheckpoint:
    with tarfile.open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
        try:
            checkpoint_file.extractall(output_path, filter="data")
        except TypeError:
            checkpoint_file.extractall(output_path)

    checkpoint = {}
    for path in sorted(output_path.rglob("*")):
        if path.name == "model_config.yaml":
            checkpoint["config_path"] = path
            checkpoint["config"] = yaml.safe_load(path.read_bytes())
        elif path.suffix in (".pt", ".pth", ".ckpt", ".bin"):
            checkpoint["weight_path"] = path

    if "config_path" not in checkpoint:
        raise FileNotFoundError(f"Could not find model config file in {checkpoint_path}.")

    if "weight_path" not in checkpoint:
        raise FileNotFoundError(f"Could not find model weights file in {checkpoint_path}.")

    return NeMoCheckpoint(root_path=output_path, **checkpoint)


def convert_checkpoint(
    checkpoint_path_or_repository_id: str,
    output_path: Path,
    model_type: Literal["encoder", "ctc"] = "ctc",
    push_to_hub: str | None = None,
):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path_or_repository_id)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        checkpoint = extract_nemo_checkpoint(checkpoint_path, temporary_path)

        logger.info("Creating processor...")
        processor = checkpoint.as_processor()
        processor.save_pretrained(output_path)
        logger.info(f"Saved processor to {output_path}")

        match model_type:
            case "encoder":
                logger.info("Loading ConformerEncoder model...")
                model_cls = ConformerEncoder
                model = checkpoint.as_encoder_model()
            case "ctc":
                logger.info("Loading ConformerForCTC model...")
                model_cls = ConformerForCTC
                model = checkpoint.as_ctc_model()

        model.save_pretrained(output_path)
        logger.info(f"Saved model to {output_path}")

    logger.info("Verifying checkpoint...")
    del processor
    del model
    gc.collect()

    processor = ConformerProcessor.from_pretrained(output_path)
    model = model_cls.from_pretrained(output_path)
    logger.info("Checkpoint reloaded successfully!")

    if push_to_hub is not None:
        logger.info("Pushing to Hub...")
        processor.push_to_hub(push_to_hub)
        model.push_to_hub(push_to_hub)  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path or HF repository id to the original NeMo checkpoint."
    )
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the converted checkpoint.")
    parser.add_argument(
        "--model_type",
        default="ctc",
        choices=["encoder", "ctc"],
        help="Model type (`encoder` or `ctc`). Defaults to ctc.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (e.g., 'username/repository'). If not provided, only saves locally.",
    )
    arguments = parser.parse_args()

    convert_checkpoint(
        arguments.checkpoint,
        Path(arguments.output_dir),
        arguments.model_type,
        arguments.push_to_hub,
    )
