# TPU experiments

This folder is home to the TPU experiments done by the Hugging Face. At the time of writing we're focusing on running 
**GLUE** experiments on a **CLOUD TPU**.
We take several approaches that are detailed below. If we're blocked in our path by an error, the error will be 
displayed in this README.

## Running GLUE on a Cloud TPU

### Using TPUStrategy with a custom loop

This refers to the script `run_tpu_glue.py`. It doesn't crash anymore. Advancing on the script.

### Using TPUStrategy with keras' fit method

This refers to the script `run_tpu_glue_fit.py`. It currently runs fine on a CPU/GPU, but crashes on TPU with the 
following error message:

```
Running on TPU  ['192.168.31.2:8470']
Number of accelerators:  8
2019-09-29 17:09:03.135000: E tensorflow/core/framework/dataset.cc:76] The Encode() method is not implemented for DatasetVariantWrapper objects.
Traceback (most recent call last):
  File "/home/lysandre/transformers/examples/TPU/run_tpu_glue_fit.py", line 42, in <module>
    validation_data=valid_dataset, validation_steps=7)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/keras/engine/training.py", line 728, in fit
    use_multiprocessing=use_multiprocessing)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/keras/engine/training_distributed.py", line 628, in fit
    shuffle=shuffle)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/keras/engine/training.py", line 2419, in _standardize_user_data
    all_inputs, y_input, dict_inputs = self._build_model_with_inputs(x, y)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/keras/engine/training.py", line 2578, in _build_model_with_inputs
    inputs, targets, _ = training_utils.extract_tensors_from_dataset(inputs)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/keras/engine/training_utils.py", line 1581, in extract_tensors_from_dataset
    iterator = get_iterator(dataset)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/keras/engine/training_utils.py", line 1559, in get_iterator
    iterator = dataset_ops.make_one_shot_iterator(dataset)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 2070, in make_one_shot_iterator
    return dataset._make_one_shot_iterator()  # pylint: disable=protected-access
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 1655, in _make_one_shot_iterator
    return iterator_ops.IteratorV2(self)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/iterator_ops.py", line 595, in __init__
    self._create_iterator(dataset)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/iterator_ops.py", line 599, in _create_iterator
    dataset = dataset._apply_options()
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 295, in _apply_options
    static_optimization_configs)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 3741, in __init__
    **self._flat_structure)
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/ops/gen_dataset_ops.py", line 3960, in optimize_dataset
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:worker/replica:0/task:0/device:CPU:0 in order to run OptimizeDataset: Unable to parse tensor proto
Additional GRPC error information:
{"created":"@1569776943.135494508","description":"Error received from peer","file":"external/grpc/src/core/lib/surface/call.cc","file_line":1039,"grpc_message":"Unable to parse tensor proto","grpc_status":3} [Op:OptimizeDataset]
Exception ignored in: <bound method _RandomSeedGeneratorDeleter.__del__ of <tensorflow.python.data.ops.dataset_ops._RandomSeedGeneratorDeleter object at 0x7f6f795884e0>>
Traceback (most recent call last):
  File "/home/lysandre/transformers/venv/lib/python3.5/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 3009, in __del__
TypeError: 'NoneType' object is not callable
```
