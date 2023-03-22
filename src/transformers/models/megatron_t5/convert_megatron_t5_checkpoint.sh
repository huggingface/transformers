#!/bin/bash
# This script was created by eagle705.
# Copyright (C) 2023 eagle705. All rights reserved.

python convert_megatron_t5_checkpoint.py --hparams_yaml_file NeMo_hparams.yaml --print-checkpoint-structure --path_to_checkpoint=model.ckpt --output_dir=./t5 --config_file=./config.json
