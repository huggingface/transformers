# TPU experiments

This folder is home to the TPU experiments done by the Hugging Face. At the time of writing we're focusing on running 
**GLUE** experiments on a **CLOUD TPU**.
We take several approaches that are detailed below. If we're blocked in our path by an error, the error will be 
displayed in this README.

## Running GLUE on a Cloud TPU

### Using TPUStrategy

This refers to the script `run_tpu_glue.py`. It currently crashes with the following error:

```
Number of accelerators:  8
TPUStrategy obtained.
2019-09-29 15:54:22.498885: E tensorflow/core/framework/dataset.cc:76] The Encode() method is not implemented for DatasetVariantWrapper objects.
Traceback (most recent call last):
  File "/home/lysandre/transformers/examples/TPU/run_tpu_glue.py", line 32, in <module>
    train_distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/distribute/distribute_lib.py", line 674, in experimental_distribute_dataset
    return self._extended._experimental_distribute_dataset(dataset)  # pylint: disable=protected-access
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/distribute/tpu_strategy.py", line 256, in _experimental_distribute_dataset
    split_batch_by=self._num_replicas_in_sync)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/distribute/input_lib.py", line 81, in get_distributed_dataset
    input_context=input_context)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/distribute/input_lib.py", line 558, in __init__
    input_context=input_context)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/distribute/input_lib.py", line 484, in __init__
    dataset = distribute._RebatchDataset(dataset, split_batch_by)  # pylint: disable=protected-access
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/experimental/ops/distribute.py", line 112, in __init__
    **self._flat_structure)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/ops/gen_experimental_dataset_ops.py", line 6468, in rebatch_dataset
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:worker/replica:0/task:0/device:CPU:0 in order to run RebatchDataset: Unable to parse tensor proto
Additional GRPC error information:
{"created":"@1569772462.499287349","description":"Error received from peer","file":"external/grpc/src/core/lib/surface/call.cc","file_line":1039,"grpc_message":"Unable to parse tensor proto","grpc_status":3} [Op:RebatchDataset]
```