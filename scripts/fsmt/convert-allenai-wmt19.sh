#!/usr/bin/env bash
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this script acquires data and converts it to fsmt model
# it covers:
# - allenai/wmt19-de-en-6-6-base
# - allenai/wmt19-de-en-6-6-big

# this script needs to be run from the top level of the transformers repo
if [ ! -d "src/transformers" ]; then
    echo "Error: This script needs to be run from the top of the transformers repo"
    exit 1
fi

mkdir data

# get data (run once)

cd data
gdown 'https://drive.google.com/uc?id=1j6z9fYdlUyOYsh7KJoumRlr1yHczxR5T'
gdown 'https://drive.google.com/uc?id=1yT7ZjqfvUYOBXvMjeY8uGRHQFWoSo8Q5'
gdown 'https://drive.google.com/uc?id=15gAzHeRUCs-QV8vHeTReMPEh1j8excNE'
tar -xvzf wmt19.de-en.tar.gz
tar -xvzf wmt19_deen_base_dr0.1_1.tar.gz
tar -xvzf wmt19_deen_big_dr0.1_2.tar.gz
cp wmt19.de-en/data-bin/dict.*.txt wmt19_deen_base_dr0.1_1
cp wmt19.de-en/data-bin/dict.*.txt wmt19_deen_big_dr0.1_2
cd -

# run conversions and uploads

PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19_deen_base_dr0.1_1/checkpoint_last3_avg.pt --pytorch_dump_folder_path data/wmt19-de-en-6-6-base

PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19_deen_big_dr0.1_2/checkpoint_last3_avg.pt --pytorch_dump_folder_path data/wmt19-de-en-6-6-big


# upload
cd data
transformers-cli upload -y wmt19-de-en-6-6-base
transformers-cli upload -y wmt19-de-en-6-6-big
cd -


# if updating just small files and not the large models, here is a script to generate the right commands:
perl -le 'for $f (@ARGV) { print qq[transformers-cli upload -y $_/$f --filename $_/$f] for ("wmt19-de-en-6-6-base", "wmt19-de-en-6-6-big")}' vocab-src.json vocab-tgt.json tokenizer_config.json config.json
# add/remove files as needed

