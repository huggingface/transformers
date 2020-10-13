#!/usr/bin/env bash

path=$(realpath ${1})
python ./convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py --prophetnet_checkpoint_path "${path}_old" --pytorch_dump_folder_path "${path}"
