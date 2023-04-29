import re
import argparse
import os

import torch
import torch.nn as nn
from ...models import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

from modeling_flamingo import (
    FlamingoConfig,
    FlamingoPreTrainedModel,
    FlamingoLMMixin,
    extend_instance,
    _infer_decoder_layers_attr_name,
    FlamingoPerceiverResampler,
)

from configuration_flamingo import FlamingoConfig


class FlamingoModel(FlamingoPreTrainedModel):
    config_class = FlamingoConfig

    def __init__(
        self,
        config: FlamingoConfig,
    ):
        super().__init__(config)
        text_tokenizer = LlamaTokenizer.from_pretrained(config.text_config._name_or_path)
        lang_encoder = LlamaForCausalLM.from_pretrained(config.text_config._name_or_path)
        vision_encoder = CLIPVisionModel.from_pretrained(config.vision_config._name_or_path)

        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, FlamingoLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        self.use_media_placement_augmentation = config.use_media_placement_augmentation

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = FlamingoPerceiverResampler(dim=self.vis_dim)

        self.lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)


def rename_flamingo_checkpoint(old_ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename some keys in the public Flamingo checkpoint"""
    perceiver_pattern1 = re.compile(r"perceiver\.layers\.[0-9]\.0")
    perceiver_pattern2 = re.compile(r"perceiver\.layers\.[0-9]\.1")
    new_ckpt = old_ckpt.copy()
    for key, value in old_ckpt.items():
        if re.match(perceiver_pattern1, key):
            new_key = re.sub(r"([0-9])\.0", r"\1", key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif re.match(perceiver_pattern2, key):
            new_key = re.sub(r"([0-9])\.1", r"\1.feed_forward", key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif key.startswith("lang_encoder.gated_cross_attn_layers."):
            new_ckpt.pop(key)
        elif key.startswith("lang_encoder.") and "ff_gate" not in key:
            new_key = key.replace("ff", "feed_forward")
            new_ckpt.pop(key)
            new_ckpt[new_key] = value

    return new_ckpt


@torch.no_grad()
def dump_hf_model(old_ckpt_path: str, new_folder_path: str) -> None:
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
    config = FlamingoConfig.from_json_file("flamingo_hf/config.json")
    model = FlamingoModel(config)
    new_ckpt = rename_flamingo_checkpoint(old_ckpt)
    model.load_state_dict(new_ckpt, strict=False)
    model.save_pretrained(new_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_ckpt_path",
        "-old",
        type=str,
        required=True,
        help="Path to the OpenFlamingo checkpoint",
    )
    parser.add_argument(
        "--new_hf_path",
        "-new",
        type=str,
        required=True,
        help="Path to the HF folder",
    )
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.new_hf_path)):
        os.makedirs(os.path.dirname(args.new_hf_path))
    dump_hf_model(args.old_ckpt_path, args.new_hf_path)
