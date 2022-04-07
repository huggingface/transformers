import os
import time

import numpy as np

import onnxruntime as ort


os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"
os.environ["ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE"] = "0"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"

sess_opt = ort.SessionOptions()
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
print("Create inference session...")
execution_provider = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
sess = ort.InferenceSession("model.onnx", sess_options=sess_opt, providers=execution_provider)
run_opt = ort.RunOptions()

sequence = 128
batch = 1
input_ids = np.ones((batch, sequence), dtype=np.int64)
attention_mask = np.ones((batch, sequence), dtype=np.int64)
token_type_ids = np.ones((batch, sequence), dtype=np.int64)

print("Warm up phase...")
sess.run(
    None,
    {
        sess.get_inputs()[0].name: input_ids,
        sess.get_inputs()[1].name: attention_mask,
        sess.get_inputs()[2].name: token_type_ids,
    },
    run_options=run_opt,
)

print("Start inference...")
start_time = time.time()
max_iters = 2000
predict = {}
for iter in range(max_iters):
    predict = sess.run(
        None,
        {
            sess.get_inputs()[0].name: input_ids,
            sess.get_inputs()[1].name: attention_mask,
            sess.get_inputs()[2].name: token_type_ids,
        },
        run_options=run_opt,
    )
print("Average Inference Time = {:.3f} ms".format((time.time() - start_time) * 1000 / max_iters))
