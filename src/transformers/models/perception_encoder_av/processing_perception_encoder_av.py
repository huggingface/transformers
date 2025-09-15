from typing import Optional, Union

import torch

from ...processing_utils import ProcessorMixin
from ...utils import is_vision_available
from ...video_processing_utils import BaseVideoProcessor, VideoMetadata


if is_vision_available():
    from ...image_utils import PILImageResampling


class PerceptionEncoderAVVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        if num_frames:
            total_frames = metadata.total_num_frames
            num_frames = num_frames if num_frames is not None else self.num_frames
            assert num_frames is not None, "`num_frames` must be specified if `fixed_len_video == True`"
            frame_idxs = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
            return torch.tensor(frame_idxs)
        else:
            return super().sample_frames(metadata, num_frames, fps, **kwargs)


class PerceptionEncoderAVProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer", "feature_extractor"]
    video_processor_class = "PerceptionEncoderAVVideoProcessor"
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "PerceptionEncoderAVFeatureExtractor"


__all__ = ["PerceptionEncoderAVProcessor", "PerceptionEncoderAVVideoProcessor"]
