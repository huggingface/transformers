import argparse

import torch
import fnmatch

from transformers import (
    DacConfig,
    DacModel,
    logging,
)

# checkpoints downloaded using:
# pip install descript-audio-codec
# python3 -m dac download # downloads the default 44kHz variant
# python3 -m dac download --model_type 44khz # downloads the 44kHz variant
# python3 -m dac download --model_type 24khz # downloads the 24kHz variant
# python3 -m dac download --model_type 16khz # downloads the 16kHz variant
# More informations: https://github.com/descriptinc/descript-audio-codec/tree/main 

logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.dac")


def match_pattern(string, pattern ):
    # Split the pattern into parts
    pattern_parts = pattern.split('.')
    string_parts = string.split('.')

    pattern_block_count = string_block_count = 0

    for part in pattern_parts:
        if part.startswith('block'):
            pattern_block_count += 1

    for part in string_parts:
        if part.startswith('block'):
            string_block_count += 1

    return fnmatch.fnmatch(string, pattern) and string_block_count == pattern_block_count


TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "alpha":
        hf_pointer.alpha.data = value
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")


def should_ignore(name, ignore_keys):
    for key in ignore_keys:
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False


def recursively_load_weights(orig_dict, hf_model, model_name):
    unused_weights = []

    if model_name in ["dac_16khz", "dac_24khz","dac_44khz"] :
        print('supported model')
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for name, value in orig_dict.items():
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        mapped_key = '.'.join(name.split('.')[:-1])
        if "weight_g" in name:
            weight_type = "weight_g"
        elif "weight_v" in name:
            weight_type = "weight_v"
        elif "bias" in name:
            weight_type = "bias"
        elif "alpha" in name:
            weight_type = "alpha"
        elif "weight" in name:
            weight_type = "weight"
        set_recursively(hf_model, mapped_key, value, name, weight_type)

    logger.warning(f"Unused weights: {unused_weights}")



@torch.no_grad()
def convert_checkpoint(
    model_name,
    checkpoint_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
        
        
    model_dict = torch.load(checkpoint_path, "cpu")

    config = DacConfig()

    metadata = model_dict['metadata']['kwargs']
    config.encoder_dim = metadata['encoder_dim']
    config.encoder_rates = metadata['encoder_rates']
    config.codebook_size = metadata['codebook_size']
    config.n_codebooks = metadata['n_codebooks']
    config.codebook_dim = metadata['codebook_dim']
    config.decoder_dim = metadata['decoder_dim']
    config.decoder_rates = metadata['decoder_rates']
    config.quantizer_dropout = metadata['quantizer_dropout']
    config.sample_rate = metadata['sample_rate']

    model = DacModel(config)

    original_checkpoint = model_dict['state_dict']

    recursively_load_weights(original_checkpoint, model, model_name)
    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        # feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="dac_16khz",
        type=str,
        help="The model to convert. Should be one of 'encodec_24khz', 'encodec_32khz', 'encodec_48khz'.",
    )

    args = parser.parse_args()

    args.model="dac_44khz"

    if args.model == "dac_16khz": 
        checkpoint_path = "/home/kamil/.cache/descript/dac/weights_16khz_8kbps_0.0.5.pth"
    if args.model == "dac_24khz": 
        checkpoint_path = "/home/kamil/.cache/descript/dac/weights_24khz_8kbps_0.0.4.pth"
    if args.model == "dac_44khz": 
        checkpoint_path = "/home/kamil/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"

    pytorch_dump_folder_path = '/home/kamil/.cache/transformers_dac'
    convert_checkpoint(args.model, checkpoint_path, pytorch_dump_folder_path, "kamilakesbi/" + str(args.model))

