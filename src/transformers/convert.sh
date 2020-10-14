#!/usr/bin/env bash

path=$(realpath ${1})

names=(
	"prophetnet-large-uncased"
	"prophetnet-large-uncased-cnndm"
	"xprophetnet-large-wiki100-cased"
	"xprophetnet-large-wiki100-cased-xglue-ntg"
	"xprophetnet-large-wiki100-cased-xglue-qg"
)
for name in "${names[@]}"; do
	python convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py --prophetnet_checkpoint_path "${path}/${name}_old" --pytorch_dump_folder_path "${path}/${name}"
#	eval "python ./save_tokenizer.py ${name}"
done
