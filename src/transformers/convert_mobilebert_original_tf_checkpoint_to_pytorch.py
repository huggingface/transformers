import argparse

import torch

from transformers import MobileBertConfig, MobileBertForPreTraining, load_tf_weights_in_mobilebert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, mobilebert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = MobileBertConfig.from_json_file(mobilebert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = MobileBertForPreTraining(config)
    # Load weights from tf checkpoint
    model = load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path)
    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--mobilebert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained MobileBERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.mobilebert_config_file, args.pytorch_dump_path)
