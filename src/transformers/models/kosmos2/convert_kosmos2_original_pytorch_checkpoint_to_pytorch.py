import argparse
import re

from fairseq.checkpoint_utils import load_checkpoint_to_cpu

from transformers import Kosmos2Config, Kosmos2ForConditionalGeneration


KEYS_TO_IGNORE = [
    # this buffer in the original code is only used to send weights to the desired device
    "gpt_model.decoder.embed_positions._float_tensor",
    # this weight is never used in the forward in the original KOSMOS-2)
    "gpt_model.decoder.self_attn_sope.scale",
]


def rename_vision_key(key):
    key = re.sub(r"img_model.visual\.", "vision_model.model.", key)
    key = re.sub(r"\.class_embedding$", ".embeddings.class_embedding", key)
    key = re.sub(r"\.positional_embedding$", ".embeddings.position_embedding.weight", key)
    key = re.sub(r"\.conv1.weight$", ".embeddings.patch_embedding.weight", key)
    key = re.sub(r"\.ln_pre\.", ".pre_layrnorm.", key)
    key = re.sub(r"\.transformer.resblocks\.", ".encoder.layers.", key)
    key = re.sub(r"\.ts_attn\.", ".self_attn.", key)
    key = re.sub(r"\.ln_1\.", ".layer_norm1.", key)
    key = re.sub(r"\.ln_2\.", ".layer_norm2.", key)
    key = re.sub(r"\.c_fc\.", ".fc1.", key)
    key = re.sub(r"\.c_proj\.", ".fc2.", key)
    key = re.sub(r"\.ln_post\.", ".post_layernorm.", key)

    return key


def rename_key(key):
    # text decoder
    key = re.sub(r"gpt_model.decoder\.", "text_model.", key)
    # text decode: `embed_tokens`
    key = re.sub(r"\.embed_tokens\.", ".model.embed_tokens.", key)
    key = re.sub(r"\.layers\.", ".model.layers.", key)
    key = re.sub(r"^text_model.layer_norm\.", "text_model.model.layer_norm.", key)
    key = re.sub(r"^text_model.output_projection\.", "text_model.lm_head.", key)
    key = re.sub(r"^img_connector\.", "image_to_text_projection.", key)
    key = rename_vision_key(key)

    return key


def convert_kosmos2_checkpoint_to_pytorch(checkpoint_path, pytorch_dump_folder_path):
    state = load_checkpoint_to_cpu(checkpoint_path)

    state["cfg"]
    state_dict = state["model"]

    state_dict_keys = list(state_dict.keys())

    config = Kosmos2Config()
    model = Kosmos2ForConditionalGeneration(config)

    # convert (by renaming keys)
    converted_state_dict = {}
    for key in state_dict_keys:
        if key in KEYS_TO_IGNORE:
            continue
        renamed_key = rename_key(key)
        converted_state_dict[renamed_key] = state_dict[key]

    # all HF model keys should be in the renamed keys from the original checkpoint
    assert set(model.state_dict().keys()) == set(converted_state_dict.keys())

    # check weight loading
    model.load_state_dict(converted_state_dict, strict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--kosmos2_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_kosmos2_checkpoint_to_pytorch(args.kosmos2_checkpoint_path, args.pytorch_dump_folder_path)
