import numpy as np
import torch
from PIL import Image

from transformers import BeitFeatureExtractor, Data2VecVisionForImageClassification, Data2VecVisionForSemanticSegmentation


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


feature_extractor = BeitFeatureExtractor.from_pretrained("facebook/data2vec-vision-base")
# model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base-ft1k")
model = Data2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base")


# for k in model.state_dict():
#     if "relative_position_bias_table" in k:
#         np.save(f"{k}.npy", model.state_dict()[k].numpy())

# mae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
# print(mae_model.state_dict().keys())

image = prepare_img()
inputs = feature_extractor(images=image, return_tensors="pt")


with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits.size())
# logits = outputs.logits

# # verify the logits
# expected_shape = torch.Size((1, 1000))
# assert logits.shape == expected_shape

# expected_slice = torch.tensor([0.3277, -0.1395, 0.0911]).numpy()

# np.testing.assert_allclose(logits[0, :3].numpy(), expected_slice, atol=1e-4)

# expected_top2 = [model.config.label2id[i] for i in ["remote control, remote", "tabby, tabby cat"]]
# np.testing.assert_equal(logits[0].topk(2).indices.cpu().numpy().tolist(), expected_top2)
