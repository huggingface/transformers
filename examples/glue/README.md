# GLUE Benchmark

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py).

#### Run PyTorch version using PyTorch-Lightning

Run `bash run_pl.sh` from the `glue` directory. This will also install `pytorch-lightning` and the requirements in `examples/requirements.txt`. It is a shell pipeline that will automatically download, pre-process the data and run the specified models. Logs are saved in `lightning_logs` directory.

Pass `--n_gpu` flag to change the number of GPUs. Default uses 1. At the end, the expected results are: `TEST RESULTS {'val_loss': tensor(0.0707), 'precision': 0.852427800698191, 'recall': 0.869537067011978, 'f1': 0.8608974358974358}`