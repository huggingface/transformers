#!/bin/bash

rm -rf /fs/nexus-projects/JSALT_workshop/lasha/Dev/audiovisualflamingo-hf
python src/transformers/models/audiovisualflamingo/convert_audiovisualflamingo_to_hf.py \
    --model_dir /fs/nexus-projects/JSALT_workshop/lasha/Dev/audiovisualflamingo \
    --output_dir /fs/nexus-projects/JSALT_workshop/lasha/Dev/audiovisualflamingo-hf \
    --push_to_hub SreyanG-NVIDIA/audiovisualflamingo-hf
