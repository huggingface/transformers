export WANDB_PROJECT=distilbart-trainer
export BS=32
export m=sshleifer/student_cnn_12_6
export tok=facebook/bart-large
export MAX_TGT_LEN=142

python finetune_trainer.py \
    --model_name_or_path $m --tokenizer_name $tok \ 
    --data_dir cnn_dm \
    --output_dir distilbart-cnn-12-6 --overwrite_output_dir \
    --learning_rate=3e-5 \
    --warmup_steps 500 --sortish_sampler \
    --fp16 \
    --n_val 500 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_encoder --freeze_embeds \
    --num_train_epochs=2 \
    --save_steps 3000 --eval_steps 3000 \
    --logging_first_step \
    --max_target_length 56 --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN \
    --do_train --do_eval --do_predict --evaluate_during_training \
    --predict_with_generate --sortish_sampler \
    "$@"
