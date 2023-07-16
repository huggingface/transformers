import argparse

import torch

from transformers import UnivNetGan, UnivNetGanConfig, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.univnet")


def load_weights(checkpoint, hf_model, config):
    hf_model.apply_weight_norm()

    # Convert initial conv layer
    hf_model.conv_pre.weight_g.data = checkpoint["conv_pre.weight_g"]
    hf_model.conv_pre.weight_v.data = checkpoint["conv_pre.weight_v"]
    hf_model.conv_pre.bias.data = checkpoint["conv_pre.bias"]

    # Convert location variable convolution (LVC) blocks
    for i in range(len(config.resblock_stride_sizes)):
        # Convert LVCBlock initial convt layer
        hf_model.resblocks[i].convt_pre.weight_g.data = checkpoint[f"res_stack.{i}.convt_pre.1.weight_g"]
        hf_model.resblocks[i].convt_pre.weight_v.data = checkpoint[f"res_stack.{i}.convt_pre.1.weight_v"]
        hf_model.resblocks[i].convt_pre.bias.data = checkpoint[f"res_stack.{i}.convt_pre.1.bias"]

        # Convert kernel predictor
        hf_model.resblocks[i].kernel_predictor.input_conv.weight_g.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.input_conv.0.weight_g"
        ]
        hf_model.resblocks[i].kernel_predictor.input_conv.weight_v.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.input_conv.0.weight_v"
        ]
        hf_model.resblocks[i].kernel_predictor.input_conv.bias.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.input_conv.0.bias"
        ]

        for j in range(config.kernel_predictor_num_blocks):
            hf_model.resblocks[i].kernel_predictor.resblocks[j].conv1.weight_g.data = checkpoint[
                f"res_stack.{i}.kernel_predictor.residual_convs.{j}.1.weight_g"
            ]
            hf_model.resblocks[i].kernel_predictor.resblocks[j].conv1.weight_v.data = checkpoint[
                f"res_stack.{i}.kernel_predictor.residual_convs.{j}.1.weight_v"
            ]
            hf_model.resblocks[i].kernel_predictor.resblocks[j].conv1.bias.data = checkpoint[
                f"res_stack.{i}.kernel_predictor.residual_convs.{j}.1.bias"
            ]

            hf_model.resblocks[i].kernel_predictor.resblocks[j].conv2.weight_g.data = checkpoint[
                f"res_stack.{i}.kernel_predictor.residual_convs.{j}.3.weight_g"
            ]
            hf_model.resblocks[i].kernel_predictor.resblocks[j].conv2.weight_v.data = checkpoint[
                f"res_stack.{i}.kernel_predictor.residual_convs.{j}.3.weight_v"
            ]
            hf_model.resblocks[i].kernel_predictor.resblocks[j].conv2.bias.data = checkpoint[
                f"res_stack.{i}.kernel_predictor.residual_convs.{j}.3.bias"
            ]

        hf_model.resblocks[i].kernel_predictor.kernel_conv.weight_g.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.kernel_conv.weight_g"
        ]
        hf_model.resblocks[i].kernel_predictor.kernel_conv.weight_v.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.kernel_conv.weight_v"
        ]
        hf_model.resblocks[i].kernel_predictor.kernel_conv.bias.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.kernel_conv.bias"
        ]

        hf_model.resblocks[i].kernel_predictor.bias_conv.weight_g.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.bias_conv.weight_g"
        ]
        hf_model.resblocks[i].kernel_predictor.bias_conv.weight_v.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.bias_conv.weight_v"
        ]
        hf_model.resblocks[i].kernel_predictor.bias_conv.bias.data = checkpoint[
            f"res_stack.{i}.kernel_predictor.bias_conv.bias"
        ]

        # Convert LVC residual blocks
        for j in range(len(config.resblock_dilation_sizes[i])):
            hf_model.resblocks[i].resblocks[j].conv.weight_g.data = checkpoint[
                f"res_stack.{i}.conv_blocks.{j}.1.weight_g"
            ]
            hf_model.resblocks[i].resblocks[j].conv.weight_v.data = checkpoint[
                f"res_stack.{i}.conv_blocks.{j}.1.weight_v"
            ]
            hf_model.resblocks[i].resblocks[j].conv.bias.data = checkpoint[f"res_stack.{i}.conv_blocks.{j}.1.bias"]

    # Convert output conv layer
    hf_model.conv_post.weight_g.data = checkpoint["conv_post.1.weight_g"]
    hf_model.conv_post.weight_v.data = checkpoint["conv_post.1.weight_v"]
    hf_model.conv_post.bias.data = checkpoint["conv_post.1.bias"]

    hf_model.remove_weight_norm()


@torch.no_grad()
def convert_univnet_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    if config_path is not None:
        config = UnivNetGanConfig.from_pretrained(config_path)
    else:
        config = UnivNetGanConfig()

    model = UnivNetGan(config)

    orig_checkpoint = torch.load(checkpoint_path)
    load_weights(orig_checkpoint["model_g"], model, config)

    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_univnet_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )
