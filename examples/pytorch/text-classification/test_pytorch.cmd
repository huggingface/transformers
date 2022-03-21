@echo off

REM ============================================================================
REM Script used to test GLUE task with pytorch backend
REM ============================================================================

SET PYTHON_EXE=C:\\Users\\v-mankang\\anaconda3\\envs\\huggingface\\python.exe

SET ENCODER_BACKEND=pytorch
SET TASK_NAME=wnli
SET SEED=2022

SET OUTPUT_DIR=C:\\work\\HuggingFace\\transformers\\examples\\pytorch\\text-classification\\%TASK_NAME%\\pytorch

%PYTHON_EXE% run_glue_no_trainer.py --model_name_or_path bert-base-cased --encoder_backend %ENCODER_BACKEND% --task_name %TASK_NAME% --max_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1 --output_dir %OUTPUT_DIR% --seed %SEED%
