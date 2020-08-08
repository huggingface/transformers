#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

# From appendix C of paper https://arxiv.org/abs/1912.08777
# Set --gradient_accumulation_steps  so that effective batch size is 256 (2*128, 4*64, 8*32, 16*16)
python finetune.py \
    --learning_rate=1e-4 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 56 \
    --freeze_embeds --max_target_length 56 --label_smoothing 0.1 \
    $@

export WANDB_PROJECT=pegasus_ft_v0
export BS=16
export GAS=16
./finetune_pegasus_xsum.sh --data_dir $XSUM_DIR --output_dir peg_large_xsum_ft \
  --num_train_epochs 10  --train_batch_size $BS --eval_batch_size $BS  \
  --gradient_accumulation_steps $GAS \
  --logger_name wandb --gpus 1
