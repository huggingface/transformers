from test_config import BeitConfig
from transformers import BeitImageProcessor
from transformers.models.beit.modeling_tf_beit import TFBeitModel
from transformers.models.beit.modeling_beit import BeitModel
from PIL import Image
import numpy as np
import torch
    
img = Image.open('/home/madelf1337/Projects/transformers/tests/fixtures/tests_samples/COCO/000000039769.png')

img_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")

image1 = img_processor(images = img, return_tensors='pt')
image2 = img_processor(images = img, return_tensors='tf')

config = BeitConfig.from_pretrained('microsoft/beit-base-patch16-224-pt22k', output_hidden_states=True, output_attentions=True)

pt_outputs = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k", config=config)
tf_outputs = TFBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k",config=config, from_pt=True)

# pt_outputs.eval()

with torch.no_grad():
    op1 = pt_outputs(**image1)
op2 = tf_outputs(image2)


def find_pt_tf_differences(pt_outputs, tf_outputs):
        """
        Compares the TensorFlow and PyTorch outputs, returning a dictionary with all tensor differences.
        """
        # 1. All output attributes must be the same
        pt_out_attrs = set(pt_outputs)
        tf_out_attrs = set(tf_outputs)
        if pt_out_attrs != tf_out_attrs:
            raise ValueError(
                f"The model outputs have different attributes, aborting. (Pytorch: {pt_out_attrs}, TensorFlow:"
                f" {tf_out_attrs})"
            )

        # 2. For each output attribute, computes the difference
        def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=""):
            # If the current attribute is a tensor, it is a leaf and we make the comparison. Otherwise, we will dig in
            # recursivelly, keeping the name of the attribute.
            if isinstance(pt_out, torch.Tensor):
                tensor_difference = np.max(np.abs(pt_out.numpy() - tf_out.numpy()))
                differences[attr_name] = tensor_difference
            else:
                root_name = attr_name
                for i, pt_item in enumerate(pt_out):
                    # If it is a named attribute, we keep the name. Otherwise, just its index.
                    if isinstance(pt_item, str):
                        branch_name = root_name + pt_item
                        tf_item = tf_out[pt_item]
                        pt_item = pt_out[pt_item]
                    else:
                        branch_name = root_name + f"[{i}]"
                        tf_item = tf_out[i]
                    differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)

            return differences

        return _find_pt_tf_differences(pt_outputs, tf_outputs, {})
conversion_differences = find_pt_tf_differences(op1, op2)
output_differences = {k: v for k, v in conversion_differences.items() if "hidden" not in k}
hidden_differences = {k: v for k, v in conversion_differences.items() if "hidden" in k}

print(output_differences)
print(hidden_differences)