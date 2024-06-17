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


MAPPING_ENCODER = {
    'encoder.block.*': "", 
    'encoder.block.*.block.*': "", 
    'encoder.block.*.block.*.block.*': "", 
}

MAPPING_QUANTIZER = {
    'quantizer.quantizers.': '', 
}

MAPPING_DECODER = {
    'decoder.model.*': '', 
    'decoder.model.*.block.*': '', 
    'decoder.model.*.block.*.block.*': "", 
}


MAPPING_16K = {
    **MAPPING_ENCODER,
    **MAPPING_QUANTIZER,
    **MAPPING_DECODER,
}

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
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    elif weight_type == "weight_ih_l0":
        hf_pointer.weight_ih_l0.data = value
    elif weight_type == "weight_hh_l0":
        hf_pointer.weight_hh_l0.data = value
    elif weight_type == "bias_ih_l0":
        hf_pointer.bias_ih_l0.data = value
    elif weight_type == "bias_hh_l0":
        hf_pointer.bias_hh_l0.data = value
    elif weight_type == "weight_ih_l1":
        hf_pointer.weight_ih_l1.data = value
    elif weight_type == "weight_hh_l1":
        hf_pointer.weight_hh_l1.data = value
    elif weight_type == "bias_ih_l1":
        hf_pointer.bias_ih_l1.data = value
    elif weight_type == "bias_hh_l1":
        hf_pointer.bias_hh_l1.data = value
    else:
        hf_pointer.data = value

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

    if model_name == "dac_16khz":
        MAPPING = MAPPING_16K
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for name, value in orig_dict.items():
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        is_used = False
       
        for key, mapped_key in MAPPING.items():

            if "quantizers" in key: 
                ##mapped_key = ###
                key = key  + '.'.join(name.split('.')[-3:-1]) + '.*'
                pass

            if fnmatch.fnmatch(name, key): 
                is_used = True
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
                # set_recursively(hf_model, mapped_key, value, name, weight_type)
            continue
        
        if not is_used:
            last_point = name.rfind('.')
            name = name[:last_point]
            if name not in unused_weights: 
                unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


if __name__ == "__main__": 

    model_name = "dac_16khz"

    if model_name == "dac_16khz": 
        location = "/home/kamil/.cache/descript/dac/weights_16khz_8kbps_0.0.5.pth"
        
        model_dict = torch.load(location, "cpu")

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

        # for name, layer in model.named_children():
        #     print(name, layer)

        original_checkpoint = model_dict['state_dict']

        recursively_load_weights(original_checkpoint, model, model_name)








