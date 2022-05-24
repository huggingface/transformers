import numpy as np
import torch

from transformers import VideoMAEConfig, VideoMAEForPreTraining


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                np.ones(self.num_masks_per_frame),
            ]
        )
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


num_frames = 16
input_size = 224
patch_size = (16, 16)
window_size = (num_frames // 2, input_size // patch_size[0], input_size // patch_size[1])

masked_position_generator = TubeMaskingGenerator(input_size=window_size, mask_ratio=0.9)


# test model

model = VideoMAEForPreTraining(VideoMAEConfig(norm_pix_loss=True))

pixel_values = torch.randn(1, 16, 3, 224, 224)

bool_masked_pos = masked_position_generator()
print("Shape of bool masked pos:", bool_masked_pos.shape)
print("Number of masked frames:", np.sum(bool_masked_pos))

bool_masked_pos = torch.from_numpy(bool_masked_pos)
bool_masked_pos = bool_masked_pos.unsqueeze(0)
bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

print(outputs.logits.shape)
