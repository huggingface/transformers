# Benchmarks

You might want to add new benchmarks.

You will need to define a python function named `run_benchmark` in your python file and the file must be located in this `benchmark/` directory.

The expected function signature is the following:

```py
def run_benchmark(logger: Logger, branch: str, commit_id: str, commit_msg: str, num_tokens_to_generate=100):
```

## Writing metrics to the database

TODO
