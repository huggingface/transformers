# Install newest ptl.
pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/

# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../../":"${PYTHONPATH}"

python run_bart_sum.py \
--data_dir=./cnn-dailymail/cnn_dm \
--output_dir=./results \
--do_train \
--train_batch_size=4