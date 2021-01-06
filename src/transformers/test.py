import datetime

import numpy as np

import onnxruntime as rt


sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = rt.InferenceSession("onnx/gpt2-optimized-quantized.onnx", sess_options)
input_name = sess.get_inputs()[0].name
X_test = np.zeros((1, 50))
start = datetime.datetime.now()
pred_onx = sess.run(None, {input_name: X_test.astype(np.long)})[0]
print(datetime.datetime.now() - start)
print(pred_onx)
