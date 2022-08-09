import numpy as np
import tensorflow as tf
import torch
from datasets import load_dataset

from transformers import (
    AutoFeatureExtractor,
    CvtForImageClassification,
    CvtModel,
    TFCvtForImageClassification,
    TFCvtModel,
)


dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/cvt-13")
pt_inputs = feature_extractor(image, return_tensors="pt")
tf_inputs = feature_extractor(image, return_tensors="tf")


print("\n--------- CVT  Classification ------------\n")
# PYTORCH:
pt_model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")
with torch.no_grad():
    pt_logits = pt_model(**pt_inputs).logits
pt_predicted_label = pt_logits.argmax(-1).item()

# TENSORFLOW:
tf_model = TFCvtForImageClassification.from_pretrained("microsoft/cvt-13", from_pt=True)
tf_logits = tf_model(**tf_inputs).logits
tf_predicted_label = int(tf.math.argmax(tf_logits, axis=-1))

print(f"TF input shape: {tf_inputs['pixel_values'].shape}")
print(f"PT input shape: {pt_inputs['pixel_values'].shape}")
print(f"TF Predicted label: {tf_model.config.id2label[tf_predicted_label]}")
print(f"PT Predicted label: {pt_model.config.id2label[pt_predicted_label]}")


print("\n--------- Model ------------\n")
# PYTORCH:
model = CvtModel.from_pretrained("microsoft/cvt-13")
with torch.no_grad():
    outputs = model(**pt_inputs)
last_hidden_states = outputs.last_hidden_state
np_pt = last_hidden_states.numpy()

# TENSORFLOW:
tf_model = TFCvtModel.from_pretrained("microsoft/cvt-13", from_pt=True)
tfo = tf_model(**tf_inputs, training=False)
np_tf = tfo.last_hidden_state.numpy()

assert np_pt.shape == np_tf.shape
diff = np.amax(np.abs(np_pt - np_tf))
print(f"\nMax absolute difference between models outputs {diff}\n")
