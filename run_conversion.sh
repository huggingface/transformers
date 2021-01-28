#!/usr/bin/env bash
eval "python src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder_path ../add_wav2vec/hf/wav2vec2 --checkpoint_path ../add_wav2vec/data/wav2vec_big_960h.pt --dict_path ../add_wav2vec/data"
