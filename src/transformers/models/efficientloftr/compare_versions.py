# uv pip install kornia einops hydra-core opencv-python-headless pillow requests matplotlib
import pathlib

import cv2
import numpy as np
import torch
from torch import Tensor


torch.manual_seed(42)

def read_image(image_path: pathlib.Path) -> np.ndarray:
    return cv2.imread(str(image_path))


def preprocess_image(image: np.ndarray, w: int, h: int, device) -> Tensor:
    image = cv2.resize(image, (w, h)).astype(np.float32)
    image /= 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = torch.from_numpy(image)
    return image.reshape((1, 1, h, w)).to(device)

def plot_pair(outputs, image1, image2, name="default.png"):
    import matplotlib.pyplot as plt
    # Create side by side image
    merged_image = np.zeros((max(image1.height, image2.height), image1.width + image2.width, 3))
    merged_image[: image1.height, : image1.width] = np.array(image1) / 255.0
    merged_image[: image2.height, image1.width:] = np.array(image2) / 255.0
    plt.imshow(merged_image)
    plt.axis("off")

    # Retrieve the keypoints and matches
    output = outputs[0]
    keypoints0 = output["keypoints0"]
    keypoints1 = output["keypoints1"]
    matching_scores = output["matching_scores"]
    keypoints0_x, keypoints0_y = keypoints0[:, 0].cpu().numpy(), keypoints0[:, 1].cpu().numpy()
    keypoints1_x, keypoints1_y = keypoints1[:, 0].cpu().numpy(), keypoints1[:, 1].cpu().numpy()

    # Plot the matches
    for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
            keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
    ):
        plt.plot(
            [keypoint0_x, keypoint1_x + image1.width],
            [keypoint0_y, keypoint1_y],
            color=plt.get_cmap("RdYlGn")(matching_score.item()),
            alpha=0.9,
            linewidth=0.5,
        )
        plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
        plt.scatter(keypoint1_x + image1.width, keypoint1_y, c="black", s=2)

    # Save the plot
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()

# device = "cuda"
# url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
# image1 = Image.open(requests.get(url_image1, stream=True).raw)
# url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
# image2 = Image.open(requests.get(url_image2, stream=True).raw)
#
# images = [image1, image2]
#
# image_processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
# pixel_values = image_processor(images, return_tensors="pt").to(device)
#
# print(pixel_values)
# print(pixel_values["pixel_values"].shape)
#
# with torch.no_grad():
#     eloftr_config = OmegaConf.load("original_config.yaml")
#     eloftr_weights = "eloftr.pth"
#     original_model = hydra.utils.instantiate(eloftr_config)
#     original_model.to(device)
#     original_model.eval()
#     torch.manual_seed(42)
#     original_outputs = original_model(**pixel_values)
#     print(original_outputs)
#
#     image_sizes = [[(image.height, image.width) for image in images]]
#     outputs = image_processor.post_process_keypoint_matching(original_outputs, image_sizes)
#     print(outputs)
#     plot_pair(outputs, image1, image2, "original")
#
#     modified_eloftr_config = OmegaConf.load("modified_config.yaml")
#     modified_model = hydra.utils.instantiate(modified_eloftr_config)
#     modified_model.to(device)
#     modified_model.eval()
#     torch.manual_seed(42)
#     modified_outputs = modified_model(**pixel_values)
#     print(modified_outputs)
#
#     assert torch.allclose(original_outputs.keypoints, modified_outputs.keypoints)
#     assert torch.allclose(original_outputs.matches, modified_outputs.matches)
#     assert torch.allclose(original_outputs.matching_scores, modified_outputs.matching_scores)
#
