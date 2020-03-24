# Install newest ptl.
pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/

# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python run_bart_sum.py \
--data_dir=./cnn-dailymail/cnn_dm \
--model_type=bart \
--model_name_or_path=bart-large \
--learning_rate=3e-5 \
--train_batch_size=4 \
--eval_batch_size=4 \
--do_train