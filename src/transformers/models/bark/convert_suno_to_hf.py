"""Convert Bark checkpoint."""

import argparse
import os
from pathlib import Path

import torch
from bark.generation import _load_model as _bark_load_model
from huggingface_hub import hf_hub_download

from transformers import EncodecConfig, EncodecModel, set_seed
from transformers.models.bark.configuration_bark import (
    BarkCoarseConfig,
    BarkConfig,
    BarkFineConfig,
    BarkSemanticConfig,
)
from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkGenerationConfig,
    BarkSemanticGenerationConfig,
)
from transformers.models.bark.modeling_bark import BarkCoarseModel, BarkFineModel, BarkModel, BarkSemanticModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

set_seed(770)


new_layer_name_dict = {
    "c_attn": "att_proj",
    "c_proj": "out_proj",
    "c_fc": "in_proj",
    "transformer.": "",
    "h.": "layers.",
    "ln_1": "layernorm_1",
    "ln_2": "layernorm_2",
    "ln_f": "layernorm_final",
    "wpe": "position_embeds_layer",
    "wte": "input_embeds_layer",
}


REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small:
        key += "_small"
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]["file_name"])


def _download(from_hf_path, file_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    if model_type == "text":
        ModelClass = BarkSemanticModel
        ConfigClass = BarkSemanticConfig
        GenerationConfigClass = BarkSemanticGenerationConfig
    elif model_type == "coarse":
        ModelClass = BarkCoarseModel
        ConfigClass = BarkCoarseConfig
        GenerationConfigClass = BarkCoarseGenerationConfig
    elif model_type == "fine":
        ModelClass = BarkFineModel
        ConfigClass = BarkFineConfig
        GenerationConfigClass = BarkFineGenerationConfig
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        _download(model_info["repo_id"], model_info["file_name"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]

    # convert Bark model arguments to HF Bark model arguments
    model_args["num_heads"] = model_args.pop("n_head")
    model_args["hidden_size"] = model_args.pop("n_embd")
    model_args["num_layers"] = model_args.pop("n_layer")

    model_config = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(config=model_config)
    model_generation_config = GenerationConfigClass()

    model.generation_config = model_generation_config
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            # replace part of the key with corresponding layer name in HF implementation
            new_k = k[len(unwanted_prefix) :]
            for old_layer_name in new_layer_name_dict:
                new_k = new_k.replace(old_layer_name, new_layer_name_dict[old_layer_name])

            state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.num_parameters(exclude_embeddings=True)
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    model.eval()
    model.to(device)
    del checkpoint, state_dict

    return model


def load_model(pytorch_dump_folder_path, use_small=False, model_type="text"):
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()

    device = "cpu"  # do conversion on cpu

    ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
    model = _load_model(ckpt_path, device, model_type=model_type, use_small=use_small)

    # load bark initial model
    bark_model = _bark_load_model(ckpt_path, "cpu", model_type=model_type, use_small=use_small)

    if model_type == "text":
        bark_model = bark_model["model"]

    if model.num_parameters(exclude_embeddings=True) != bark_model.get_num_params():
        raise ValueError("initial and new models don't have the same number of parameters")

    # check if same output as the bark model
    batch_size = 5
    sequence_length = 10

    if model_type in ["text", "coarse"]:
        vec = torch.randint(256, (batch_size, sequence_length), dtype=torch.int)
        output_old_model = bark_model(vec)[0]

        output_new_model_total = model(vec)

        # take last logits
        output_new_model = output_new_model_total.logits[:, [-1], :]

    else:
        prediction_codeboook_channel = 3
        n_codes_total = 8
        vec = torch.randint(256, (batch_size, sequence_length, n_codes_total), dtype=torch.int)

        output_new_model_total = model(prediction_codeboook_channel, vec)
        output_old_model = bark_model(prediction_codeboook_channel, vec)

        output_new_model = output_new_model_total.logits

    # output difference should come from the difference of self-attention implementation design
    if output_new_model.shape != output_old_model.shape:
        raise ValueError("initial and new outputs don't have the same shape")
    if (output_new_model - output_old_model).abs().max().item() > 1e-3:
        raise ValueError("initial and new outputs are not equal")

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


def load_whole_bark_model(
    semantic_path,
    coarse_path,
    fine_path,
    append_text,
    hub_path,
    folder_path,
):
    pytorch_dump_folder_path = os.path.join(folder_path, append_text)

    semanticConfig = BarkSemanticConfig.from_pretrained(os.path.join(semantic_path, "config.json"))
    coarseAcousticConfig = BarkCoarseConfig.from_pretrained(os.path.join(coarse_path, "config.json"))
    fineAcousticConfig = BarkFineConfig.from_pretrained(os.path.join(fine_path, "config.json"))
    codecConfig = EncodecConfig.from_pretrained("facebook/encodec_24khz")

    semantic = BarkSemanticModel.from_pretrained(semantic_path)
    coarseAcoustic = BarkCoarseModel.from_pretrained(coarse_path)
    fineAcoustic = BarkFineModel.from_pretrained(fine_path)
    codec = EncodecModel.from_pretrained("facebook/encodec_24khz")

    bark_config = BarkConfig.from_sub_model_configs(
        semanticConfig, coarseAcousticConfig, fineAcousticConfig, codecConfig
    )

    bark_generation_config = BarkGenerationConfig.from_sub_model_configs(
        semantic.generation_config, coarseAcoustic.generation_config, fineAcoustic.generation_config
    )

    bark = BarkModel(bark_config)

    bark.semantic = semantic
    bark.coarse_acoustics = coarseAcoustic
    bark.fine_acoustics = fineAcoustic
    bark.codec_model = codec

    bark.generation_config = bark_generation_config

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    bark.save_pretrained(pytorch_dump_folder_path, repo_id=hub_path, push_to_hub=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("model_type", type=str, help="text, coarse or fine.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--is_small", action="store_true", help="convert the small version instead of the large.")

    args = parser.parse_args()

    load_model(args.pytorch_dump_folder_path, model_type=args.model_type, use_small=args.is_small)
