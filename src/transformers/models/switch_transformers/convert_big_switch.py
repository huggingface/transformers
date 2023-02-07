import argparse
import json
import os

import tensorstore as ts
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tensorflow.io import gfile

from transformers.modeling_utils import dtype_byte_size
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
    rename_keys,
)
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import convert_file_size_to_int


def rename_base_flax_keys(flax_key_tuple, flax_tensor):
    """
    Post renaming of basic JAX keys to pytorch.
    """
    if flax_key_tuple[-1] == "kernel" and flax_tensor.ndim == 3:
        # expert layer
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
        flax_tensor = torch.permute(flax_tensor, (0, 2, 1))
    elif flax_key_tuple[-1] == "kernel" and ".".join(flax_key_tuple):
        # linear layer
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
        flax_tensor = flax_tensor.T
    elif flax_key_tuple[-1] in ["scale", "embedding"]:
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

    return flax_key_tuple, flax_tensor


def get_key_and_tensorstore_dict(layer, checkpoint_info, switch_checkpoint_path):
    if "metadata" in layer:
        split_layer = layer.split("metadata")
        curr_real_layer_name = "".join(split_layer[0])[:-1]
        split_layer = [tuple(("metadata" + split_layer[1]).split("/"))]
    elif "kvstore" in layer:
        split_layer = layer.split("kvstore")
        curr_real_layer_name = "".join(split_layer[0])[:-1]
        split_layer = [tuple(("kvstore" + split_layer[1]).split("/"))]

    else:
        split_layer = layer.split("/")
        curr_real_layer_name = "/".join(split_layer[:-1])
        split_layer[-1] = (split_layer[-1],)

    if "kvstore/path" in layer:
        content = f"{switch_checkpoint_path}/{checkpoint_info[layer]}"
    elif "kvstore/driver" in layer:
        content = "file"
    else:
        content = checkpoint_info[layer]

    return curr_real_layer_name, split_layer, content


def rename_and_save_block(current_block, save_path):
    current_block = rename_keys(current_block)
    new_current_block = {}
    for k, v in current_block.items():
        new_current_block[k.replace("/", ".")] = v
    current_block = new_current_block
    torch.save(current_block, save_path)


def shard_on_the_fly(switch_checkpoint_path, dump_path, max_shard_size, dtype, weights_name: str = WEIGHTS_NAME):
    max_shard_size = convert_file_size_to_int(max_shard_size)
    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    os.makedirs(dump_path, exist_ok=True)
    with gfile.GFile(switch_checkpoint_path + "/checkpoint", "rb") as fp:
        checkpoint_info = serialization.msgpack_restore(fp.read())["optimizer"]["target"]
        checkpoint_info = flatten_dict(checkpoint_info, sep="/")

    all_layers = {}
    for layer in checkpoint_info.keys():
        curr_real_layer_name, split_layer, content = get_key_and_tensorstore_dict(
            layer, checkpoint_info, switch_checkpoint_path
        )
        if curr_real_layer_name in all_layers:
            all_layers[curr_real_layer_name][split_layer[-1]] = content
        else:
            all_layers[curr_real_layer_name] = {split_layer[-1]: content}

    for key in all_layers.keys():
        # open tensorstore file
        raw_weights = ts.open(unflatten_dict(all_layers[key])).result().read().result()
        raw_weights = torch.tensor(raw_weights)
        weight_size = raw_weights.numel() * dtype_byte_size(raw_weights.dtype)

        # use the renaming pattern from the small conversion scripts
        key, raw_weights = rename_base_flax_keys(tuple(key.split("/")), raw_weights)
        key = "/".join(key)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            save_path = os.path.join(
                dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
            )
            rename_and_save_block(current_block, save_path)
            sharded_state_dicts.append(current_block.keys())
            del current_block
            current_block = {}
            current_block_size = 0

        current_block[key] = raw_weights.to(getattr(torch, dtype))
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    rename_and_save_block(current_block, save_path)
    sharded_state_dicts.append(current_block.keys())

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(
            ".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        )  # len(sharded_state_dicts):05d}
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        shards[shard_file] = shard
        for key in shard:
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}

    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    return metadata, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128/checkpoint_634600",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    parser.add_argument("--max_shard_size", default="10GB", required=False, help="Max shard size")
    parser.add_argument("--dtype", default="bfloat16", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128-converted",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    args = parser.parse_args()
    shard_on_the_fly(
        args.switch_t5x_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.max_shard_size,
        args.dtype,
    )


def sanity_check():
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, T5Tokenizer

    config = SwitchTransformersConfig.from_pretrained("google/switch-base-8")
    config.save_pretrained("/home/arthur_huggingface_co/transformers/switch_converted")
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        "/home/arthur_huggingface_co/transformers/switch_converted", device_map="auto"
    )

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    out = model.generate(input_ids, decoder_start_token_id=0)
    print(tokenizer.decode(out[0]))
