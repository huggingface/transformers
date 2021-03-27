accelerate launch run_swag_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name swag \
  --output_dir /tmp/test-swag-no-trainer \
  --pad_to_max_length \
  --debug
