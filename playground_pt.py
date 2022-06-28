
from PIL import Image
from src.transformers.models.segformer import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch 

def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


feature_extractor = SegformerFeatureExtractor(
    image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
)
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
print(all([x.momentum is not None for x in model.modules() if isinstance(x, torch.nn.BatchNorm2d)]))

image = prepare_img()
encoded_inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = encoded_inputs.pixel_values
print(pixel_values.size())

with torch.no_grad():
    outputs = model(pixel_values)

expected_shape = torch.Size((1, model.config.num_labels, 128, 128))
print(outputs.logits.shape == expected_shape)

expected_slice = torch.tensor(
    [
        [[-4.6310, -5.5232, -6.2356], [-5.1921, -6.1444, -6.5996], [-5.4424, -6.2790, -6.7574]],
        [[-12.1391, -13.3122, -13.9554], [-12.8732, -13.9352, -14.3563], [-12.9438, -13.8226, -14.2513]],
        [[-12.5134, -13.4686, -14.4915], [-12.8669, -14.4343, -14.7758], [-13.2523, -14.5819, -15.0694]],
    ]
)
print(torch.allclose(outputs.logits[0, :3, :3, :3], expected_slice, atol=1e-4))