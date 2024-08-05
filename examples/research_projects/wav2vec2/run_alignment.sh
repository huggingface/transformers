#!/usr/bin/env bash
python alignment.py  \
--model_name="arijitx/wav2vec2-xls-r-300m-bengali" \
--wav_dir="./wavs" \
--text_file="script.txt" \
--input_wavs_sr=48000 \
--output_dir="./out_alignment" \
--cuda
