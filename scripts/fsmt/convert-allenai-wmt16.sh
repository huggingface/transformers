#!/usr/bin/env bash

# this script acquires data and converts it to fsmt model
# it covers:
# - allenai/wmt16-en-de-dist-12-1
# - allenai/wmt16-en-de-dist-6-1
# - allenai/wmt16-en-de-12-1

# this script needs to be run from the top level of the transformers repo
if [ ! -d "src/transformers" ]; then
    echo "Error: This script needs to be run from the top of the transformers repo"
    exit 1
fi

mkdir data

# get data (run once)

cd data
gdown 'https://drive.google.com/uc?id=1x_G2cjvM1nW5hjAB8-vWxRqtQTlmIaQU'
gdown 'https://drive.google.com/uc?id=1oA2aqZlVNj5FarxBlNXEHpBS4lRetTzU'
gdown 'https://drive.google.com/uc?id=1Wup2D318QYBFPW_NKI1mfP_hXOfmUI9r'
tar -xvzf trans_ende_12-1_0.2.tar.gz
tar -xvzf trans_ende-dist_12-1_0.2.tar.gz
tar -xvzf trans_ende-dist_6-1_0.2.tar.gz
gdown 'https://drive.google.com/uc?id=1mNufoynJ9-Zy1kJh2TA_lHm2squji0i9'
gdown 'https://drive.google.com/uc?id=1iO7um-HWoNoRKDtw27YUSgyeubn9uXqj'
tar -xvzf wmt16.en-de.deep-shallow.dist.tar.gz
tar -xvzf wmt16.en-de.deep-shallow.tar.gz
cp wmt16.en-de.deep-shallow/data-bin/dict.*.txt trans_ende_12-1_0.2
cp wmt16.en-de.deep-shallow.dist/data-bin/dict.*.txt trans_ende-dist_12-1_0.2
cp wmt16.en-de.deep-shallow.dist/data-bin/dict.*.txt trans_ende-dist_6-1_0.2
cp wmt16.en-de.deep-shallow/bpecodes trans_ende_12-1_0.2
cp wmt16.en-de.deep-shallow.dist/bpecodes trans_ende-dist_12-1_0.2
cp wmt16.en-de.deep-shallow.dist/bpecodes trans_ende-dist_6-1_0.2
cd -

# run conversions and uploads

PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/trans_ende-dist_12-1_0.2/checkpoint_top5_average.pt --pytorch_dump_folder_path data/wmt16-en-de-dist-12-1

PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/trans_ende-dist_6-1_0.2/checkpoint_top5_average.pt --pytorch_dump_folder_path data/wmt16-en-de-dist-6-1

PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/trans_ende_12-1_0.2/checkpoint_top5_average.pt --pytorch_dump_folder_path data/wmt16-en-de-12-1


# upload
cd data
transformers-cli upload -y wmt16-en-de-dist-12-1
transformers-cli upload -y wmt16-en-de-dist-6-1
transformers-cli upload -y wmt16-en-de-12-1
cd -


# if updating just small files and not the large models, here is a script to generate the right commands:
perl -le 'for $f (@ARGV) { print qq[transformers-cli upload -y $_/$f --filename $_/$f] for ("wmt16-en-de-dist-12-1", "wmt16-en-de-dist-6-1", "wmt16-en-de-12-1")}' vocab-src.json vocab-tgt.json tokenizer_config.json config.json
# add/remove files as needed

# Caching note: Unfortunately due to CDN caching the uploaded model may be unavailable for up to 24hs after upload
# So the only way to start using the new model sooner is either:
# 1. download it to a local path and use that path as model_name
# 2. make sure you use: from_pretrained(..., use_cdn=False) everywhere
