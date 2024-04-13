


# def find_pt_tf_differences(pt_outputs, tf_outputs):
#     """
#     Compares the TensorFlow and PyTorch outputs, returning a dictionary with all tensor differences.
#     """
#     # 1. All output attributes must be the same
#     pt_out_attrs = set(pt_outputs)
#     tf_out_attrs = set(tf_outputs)
#     if pt_out_attrs != tf_out_attrs:
#         raise ValueError(
#             f"The model outputs have different attributes, aborting. (Pytorch: {pt_out_attrs}, TensorFlow:"
#             f" {tf_out_attrs})"
#         )

#     # 2. For each output attribute, computes the difference
#     def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=""):
#         # If the current attribute is a tensor, it is a leaf and we make the comparison. Otherwise, we will dig in
#         # recursivelly, keeping the name of the attribute.
#         if isinstance(pt_out, torch.Tensor):
#             tensor_difference = np.max(np.abs(pt_out.numpy() - tf_out.numpy()))
#             differences[attr_name] = tensor_difference
#         else:
#             root_name = attr_name
#             for i, pt_item in enumerate(pt_out):
#                 # If it is a named attribute, we keep the name. Otherwise, just its index.
#                 if isinstance(pt_item, str):
#                     branch_name = root_name + pt_item
#                     tf_item = tf_out[pt_item]
#                     pt_item = pt_out[pt_item]
#                 else:
#                     branch_name = root_name + f"[{i}]"
#                     tf_item = tf_out[i]
#                 differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)

#         return differences

#     return _find_pt_tf_differences(pt_outputs, tf_outputs, {})



# import numpy as np
# import tensorflow as tf
# import torch
# from transformers.file_utils import ModelOutput

# def check_pt_tf_outputs(tf_outputs, pt_outputs, model_class=TFBeitModelOutputWithPooling, tol=2e-4, name="outputs", attributes=None):
#     """Check the outputs from PyTorch and TensorFlow models are close enough. Checks are done in a recursive way.

#     Args:
#         tf_outputs: TensorFlow outputs.
#         pt_outputs: PyTorch outputs.
#         model_class: The class of the model that is currently testing. For example, `TFBertModel`,
#             `TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Mainly used for providing more informative
#             error messages.
#         tol (float): Tolerance level for comparison.
#         name (str): The name of the output. For example, `output.hidden_states`, `output.attentions`, etc.
#         attributes (Tuple[str]): The names of the output's element if the output is a tuple/list with each element
#             being a named field in the output.
#     """

#     assert isinstance(name, str), "Name must be a string"
#     if attributes is not None:
#         assert isinstance(attributes, tuple), f"{name}: The argument `attributes` should be a `tuple`"

#     # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
#     if isinstance(tf_outputs, ModelOutput):
#         assert isinstance(pt_outputs, ModelOutput), f"{name}: `pt_outputs` should an instance of `ModelOutput` when `tf_outputs` is"

#         tf_keys = [k for k, v in tf_outputs.items() if v is not None]
#         pt_keys = [k for k, v in pt_outputs.items() if v is not None]

#         assert tf_keys == pt_keys, f"{name}: Output keys differ between TF and PyTorch"

#         # convert to the case of `tuple`
#         attributes = tuple([f"{name}.{k}" for k in tf_keys])
#         check_pt_tf_outputs(
#             tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
#         )

#     elif isinstance(tf_outputs, (tuple, list)):
#         assert isinstance(tf_outputs, type(pt_outputs)), f"{name}: Output types differ between TF and PyTorch"
#         assert len(tf_outputs) == len(pt_outputs), f"{name}: Output lengths differ between TF and PyTorch"

#         if attributes is not None:
#             assert len(attributes) == len(tf_outputs), f"{name}: The tuple `names` should have the same length as `tf_outputs`"
#         else:
#             attributes = tuple([f"{name}_{idx}" for idx in range(len(tf_outputs))])

#         for tf_output, pt_output, attr in zip(tf_outputs, pt_outputs, attributes):
#             print(attr)
#             check_pt_tf_outputs(tf_output, pt_output, model_class, tol=tol, name=attr)

#     elif isinstance(tf_outputs, tf.Tensor):
#         assert isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `tf_outputs` is"

#         tf_outputs = tf_outputs.numpy()
#         pt_outputs = pt_outputs.detach().cpu().numpy()

#         assert tf_outputs.shape == pt_outputs.shape, f"{name}: Output shapes differ between TF and PyTorch"

#         tf_nans = np.isnan(tf_outputs)
#         pt_nans = np.isnan(pt_outputs)

#         # Replacing NaNs with zeros for comparison
#         pt_outputs[tf_nans] = 0
#         tf_outputs[tf_nans] = 0
#         pt_outputs[pt_nans] = 0
#         tf_outputs[pt_nans] = 0

#         max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
#         assert max_diff <= tol, f"{name}: Difference between torch and tf is {max_diff} (>= {tol})."
#     else:
#         raise ValueError(
#             f"`tf_outputs` should be an instance of `tf.Tensor`, a `tuple`, or a `list`. Got {type(tf_outputs)} instead."
#         )

# conversion_differences = find_pt_tf_differences(op1, op2)
# output_differences = {k: v for k, v in conversion_differences.items() if "hidden" not in k}
# hidden_differences = {k: v for k, v in conversion_differences.items() if "hidden" in k}

# check_pt_tf_outputs(self, op2, op1, model_class, tol=1e-5, name="outputs", attributes=None)

# print(type(op1))

# print("\n")

# print(type(op2))
# # max_diff = np.amax(np.abs(op1.numpy() - op2.numpy()))
# print(max_diff)


# check_pt_tf_outputs(op2, op1, tol=2e-4, name="outputs", attributes=None)
# max_diff = np.amax(np.abs(op1.hidden_states[-1].numpy() - op2.hidden_states[-1].numpy()))
# print(max_diff)


# import torch
# import numpy as np
# from transformers import BeitModel, BeitConfig

# # Load the BEiT model and config
# config = BeitConfig.from_pretrained("microsoft/beit-base-patch16-224")
# model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")


# hidden_states = op1.hidden_states
# layerwise_outputs = {}
# layerwise_outputs["embeddings"] = hidden_states[0].numpy()
# for i, layer_output in enumerate(hidden_states[1:]):
#     layer_name = f"encoder.layer_.{i}"
#     layerwise_outputs[layer_name] = layer_output.numpy()

# layerwise_outputs["layernorm"] = op1.last_hidden_state.numpy()
# if op1.pooler_output is not None:
#     layerwise_outputs["pooler"] = op1.pooler_output.numpy()

# # Compare with the TF layerwise outputs
# # tf_layerwise_outputs = {...}  # Replace with the TF layerwise outputs

# for layer_name, pt_output in layerwise_outputs.items():
#     tf_output = tf_layerwise_outputs[layer_name]
#     abs_diff = np.amax(np.abs(pt_output - tf_output))
#     print(f"{layer_name}: Max Absolute Difference = {abs_diff}")
    
    # if abs_diff > 2e-4:
    #     print(f"The difference for layer '{layer_name}' exceeds the tolerance level of 2e-4.")



import numpy as np
import torch
from PIL import Image

from transformers import BeitImageProcessor
from transformers.models.beit.configuration_beit import BeitConfig
from transformers.models.beit.modeling_beit import BeitForMaskedImageModeling as BeitModel
from transformers.models.beit.tf_test import TFBeitForMaskedImageModeling as TFBeitModel
from transformers.models.beit.modeling_tf_beit import TFBeitModelOutputWithPooling

img = Image.open("/home/madelf1337/Projects/transformers/tests/fixtures/tests_samples/COCO/000000039769.png")

img_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")

image1 = img_processor(images=img, return_tensors="pt")
image2 = img_processor(images=img, return_tensors="tf")

config = BeitConfig.from_pretrained(
    "microsoft/beit-base-patch16-224-pt22k", output_hidden_states=True, output_attentions=True
)

pt_outputs = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k", config=config)
tf_outputs = TFBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k", config=config, from_pt=True)


# pt_outputs.eval()

with torch.no_grad():
    op1, pt_layerwise_outputs = pt_outputs(**image1)
op2, tf_layerwise_outputs = tf_outputs(image2)



max_absolute_differences = {}

for layer_name in tf_layerwise_outputs:
    if layer_name in pt_layerwise_outputs:
        # Calculate the maximum absolute difference between the two layers' outputs
        max_diff = np.max(np.abs(tf_layerwise_outputs[layer_name] - pt_layerwise_outputs[layer_name]))
        max_absolute_differences[layer_name] = max_diff

# Print the maximum absolute differences
for layer_name, max_diff in max_absolute_differences.items():
    print(f"Max absolute difference for layer {layer_name}: {max_diff}")

