#!/usr/bin/env bash
python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
	--beit_checkpoint /home/patrick/add_data2vec_beit/pretrained_base.pt \
	--hf_checkpoint_name "/home/patrick/add_data2vec_beit/data2vec-vision-base-patch16-224"
python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
	--beit_checkpoint /home/patrick/add_data2vec_beit/finetuned_base.pt \
	--hf_checkpoint_name "/home/patrick/add_data2vec_beit/data2vec-vision-base-patch16-224-ft1k"
python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
	--beit_checkpoint /home/patrick/add_data2vec_beit/pretrained_large.pt \
	--hf_checkpoint_name "/home/patrick/add_data2vec_beit/data2vec-vision-large-patch16-224"
python ./convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py \
	--beit_checkpoint /home/patrick/add_data2vec_beit/finetuned_large.pt \
	--hf_checkpoint_name "/home/patrick/add_data2vec_beit/data2vec-vision-large-patch16-224-ft1k"
