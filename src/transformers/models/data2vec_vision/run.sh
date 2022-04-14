#!/usr/bin/env bash
python ./convert_data2vec_vision_unilm_to_pytorch_new.py \
	--beit_checkpoint /home/patrick/add_data2vec_beit/pretrained_base.pt \
	--hf_checkpoint_name "/home/patrick/add_data2vec_beit/data2vec-vision-base-patch16-224"
python ./convert_data2vec_vision_unilm_to_pytorch_new.py \
	--beit_checkpoint /home/patrick/add_data2vec_beit/finetuned_base.pt \
	--hf_checkpoint_name "/home/patrick/add_data2vec_beit/data2vec-vision-base-patch16-224-ft1k"
