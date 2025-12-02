from typing import Union

import torch

from ...utils import TensorType
from ..superglue.image_processing_superglue_fast import SuperGlueImageProcessorFast
from .modeling_efficientloftr import EfficientLoFTRKeypointMatchingOutput


class EfficientLoFTRImageProcessorFast(SuperGlueImageProcessorFast):
    def post_process_keypoint_matching(
        self,
        outputs: "EfficientLoFTRKeypointMatchingOutput",
        target_sizes: Union[TensorType, list[tuple]],
        threshold: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the raw output of [`EfficientLoFTRKeypointMatchingOutput`] into lists of keypoints, scores and descriptors
        with coordinates absolute to the original image sizes.
        Args:
            outputs ([`EfficientLoFTRKeypointMatchingOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` or `List[Tuple[Tuple[int, int]]]`, *optional*):
                Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`Tuple[int, int]`) containing the
                target size `(height, width)` of each image in the batch. This must be the original image size (before
                any processing).
            threshold (`float`, *optional*, defaults to 0.0):
                Threshold to filter out the matches with low scores.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the keypoints in the first and second image
            of the pair, the matching scores and the matching indices.
        """
        if outputs.matches.shape[0] != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")
        if not all(len(target_size) == 2 for target_size in target_sizes):
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if isinstance(target_sizes, list):
            image_pair_sizes = torch.tensor(target_sizes, device=outputs.matches.device)
        else:
            if target_sizes.shape[1] != 2 or target_sizes.shape[2] != 2:
                raise ValueError(
                    "Each element of target_sizes must contain the size (h, w) of each image of the batch"
                )
            image_pair_sizes = target_sizes

        keypoints = outputs.keypoints.clone()
        keypoints = keypoints * image_pair_sizes.flip(-1).reshape(-1, 2, 1, 2)
        keypoints = keypoints.to(torch.int32)

        results = []
        for keypoints_pair, matches, scores in zip(keypoints, outputs.matches, outputs.matching_scores):
            # Filter out matches with low scores
            valid_matches = torch.logical_and(scores > threshold, matches > -1)

            matched_keypoints0 = keypoints_pair[0][valid_matches[0]]
            matched_keypoints1 = keypoints_pair[1][valid_matches[1]]
            matching_scores = scores[0][valid_matches[0]]

            results.append(
                {
                    "keypoints0": matched_keypoints0,
                    "keypoints1": matched_keypoints1,
                    "matching_scores": matching_scores,
                }
            )

        return results


__all__ = ["EfficientLoFTRImageProcessorFast"]
