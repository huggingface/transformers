from typing import List, Tuple, Union, Optional

import torch
from torch import nn, Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import ImagePointDescriptionOutput
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig


class SuperPoint(nn.Module):
    def __init__(
            self,
            conv_layers_sizes: List[int] = [64, 64, 128, 128, 256],
            descriptor_dim: int = 256,
            keypoint_threshold: float = 0.005,
            max_keypoints: int = -1,
            nms_radius: int = 4,
            border_removal_distance: int = 4,
    ):
        super().__init__()

        self.conv_layers_sizes = conv_layers_sizes
        self.descriptor_dim = descriptor_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(1, self.conv_layers_sizes[0],
                                kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(self.conv_layers_sizes[0], self.conv_layers_sizes[0],
                                kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(self.conv_layers_sizes[0], self.conv_layers_sizes[1],
                                kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(self.conv_layers_sizes[1], self.conv_layers_sizes[1],
                                kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(self.conv_layers_sizes[1], self.conv_layers_sizes[2],
                                kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(self.conv_layers_sizes[2], self.conv_layers_sizes[2],
                                kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(self.conv_layers_sizes[2], self.conv_layers_sizes[3],
                                kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(self.conv_layers_sizes[3], self.conv_layers_sizes[3],
                                kernel_size=3, stride=1, padding=1)

        self.convSa = nn.Conv2d(self.conv_layers_sizes[3], self.conv_layers_sizes[4],
                                kernel_size=3, stride=1, padding=1)
        self.convSb = nn.Conv2d(self.conv_layers_sizes[4], 65,
                                kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(self.conv_layers_sizes[3], self.conv_layers_sizes[4],
                                kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(self.conv_layers_sizes[4], self.descriptor_dim,
                                kernel_size=1, stride=1, padding=0)

    def encode(self, input):
        """ Run the CNN to encode the image. """
        input = self.relu(self.conv1a(input))
        input = self.relu(self.conv1b(input))
        input = self.pool(input)
        input = self.relu(self.conv2a(input))
        input = self.relu(self.conv2b(input))
        input = self.pool(input)
        input = self.relu(self.conv3a(input))
        input = self.relu(self.conv3b(input))
        input = self.pool(input)
        input = self.relu(self.conv4a(input))
        output = self.relu(self.conv4b(input))
        return output

    def get_scores(self, encoded):
        """ Compute the dense keypoint scores """
        scores = self.relu(self.convSa(encoded))
        scores = self.convSb(scores)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = self.simple_nms(scores, self.nms_radius)
        return scores

    def extract_keypoints(self, scores):
        b, _, h, w = scores.shape

        # Threshold keypoints by score value
        keypoints = [
            torch.nonzero(s > self.keypoint_threshold)
            for s in scores
        ]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            self.remove_borders(k, s, self.remove_borders, h * 8, w * 8)
            for k, s in zip(keypoints, scores)
        ]))

        # Keep the k keypoints with highest score
        if self.max_keypoints >= 0:
            keypoints, scores = list(zip(*[
                self.top_k_keypoints(k, s, self.max_keypoints)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        return keypoints, scores

    def get_descriptors(self, encoded, keypoints):
        """ Compute the dense descriptors """
        descriptors = self.convDb(self.relu(self.convDa(encoded)))
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [self.sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return descriptors

    def forward(self, image) -> Tuple[Tensor, Tensor, Tensor]:
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        encoded = self.encode(image)

        # Compute the dense keypoint scores
        scores = self.get_scores(encoded)

        # Extract keypoints
        keypoints, scores = self.extract_keypoints(scores)

        # Compute the descriptors
        descriptors = self.get_descriptors(encoded, keypoints)

        return keypoints, scores, descriptors

    @staticmethod
    def simple_nms(scores, nms_radius: int):
        assert (nms_radius >= 0)

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    @staticmethod
    def remove_borders(keypoints, scores, border: int, height: int, width: int):
        """ Removes keypoints too close to the border """
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]

    @staticmethod
    def top_k_keypoints(keypoints, scores, k: int):
        if k >= len(keypoints):
            return keypoints, scores
        scores, indices = torch.topk(scores, k, dim=0)
        return keypoints[indices], scores

    @staticmethod
    def sample_descriptors(keypoints, descriptors, s: int = 8):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5
        keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                                  ).to(keypoints)[None]
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors


class SuperPointModelForInterestPointDescription(PreTrainedModel):
    config_class = SuperPointConfig
    base_model_prefix = "superpoint"

    def __init__(self, config):
        super().__init__(config)
        self.superpoint = SuperPoint(
            conv_layers_sizes=config.conv_layers_sizes,
            descriptor_dim=config.descriptor_dim,
            keypoint_threshold=config.keypoint_threshold,
            max_keypoints=config.max_keypoints,
            nms_radius=config.nms_radius,
            border_removal_distance=config.border_removal_distance,
        )

    def forward(
            self,
            image: Tensor = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImagePointDescriptionOutput]:
        keypoints, scores, descriptors = self.superpoint(image)
        if not return_dict:
            return keypoints, scores, descriptors

        return ImagePointDescriptionOutput(
            keypoints=keypoints,
            scores=scores,
            descriptors=descriptors,
        )
