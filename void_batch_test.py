import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
import random
from typing import List, Dict, Any, Tuple

from transformers import Mask2FormerConfig, Mask2FormerModel
from src.adaptive_batch_trainer import AdaptiveBatchSizeTrainer



class DummySegmentationDataset(Dataset):
    """
    Dummy segmentation dataset backed by NumPy, with a few fully-void masks.

    Args:
        num_samples:       total number of samples in the dataset (>0)
        image_size:        (H, W) of each image, both >0
        num_classes:       number of segmentation classes (>0)
        void_count:        number of samples whose mask is entirely ignore_index
        ignore_index:      label value in the mask to indicate “void” pixels
        seed:              random seed for reproducibility
    """
    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int] = (4, 4),
        num_classes: int = 5,
        void_count: int = 2,
        ignore_index: int = 255,
        seed: int = 42,
    ):
        self.num_samples   = num_samples
        self.image_size    = image_size  # (H, W)
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index

        # -- Decide which indices are fully void --
        random.seed(seed)
        self.void_indices = set(random.sample(range(num_samples), void_count))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Let Python naturally raises IndexError if idx is out of bounds
        H, W = self.image_size

        # 1) Generate a normalized image in NumPy, then convert to torch (C,H,W)
        image_norm = np.random.rand(H, W, 3).astype(np.float32)
        pixel_values = torch.from_numpy(image_norm).permute(2, 0, 1)

        # 2) Generate either a random mask or an all-void mask
        if idx in self.void_indices:
            semantic_mask = np.full((H, W), fill_value=self.ignore_index, dtype=np.uint8)
        else:
            semantic_mask = np.random.randint(
                low=0,
                high=self.num_classes,
                size=(H, W),
                dtype=np.uint8
            )
        labels = torch.from_numpy(semantic_mask).long()

        return {
            "pixel_values": pixel_values,  # (3, H, W)
            "labels": labels,              # (H, W)
            "index": idx
        }



def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of items into a batch dict:
      - 'pixel_values': (B, C, H, W) float32
      - 'labels':       (B, H, W)    int64
      - 'indices':      (B,)         int64

    How it works on high-level
    1. Converts any NumPy arrays into PyTorch tensors (so it works if your dataset returns raw NumPy)
    2. Leaves tensors alone
    3. Stacks images and masks into batched tensors as 'pixel_values', `labels` respectively
    """
    pixel_values, labels, indices = [], [], []

    for item in batch:
        for key in ("pixel_values", "labels", "index"):
            if key not in item:
                raise KeyError(f"Batch item missing '{key}' field")

        img, msk, idx = item["pixel_values"], item["labels"], item["index"]

        # -- convert numpy image H×W×3 → torch C×H×W
        if isinstance(img, np.ndarray):
            # cast to float32 if not already
            img = img.astype(np.float32, copy=False)
            img = torch.from_numpy(img).permute(2, 0, 1)

        # -- convert numpy mask H×W → torch long
        if isinstance(msk, np.ndarray):
            # assume msk is uint8; cast to long for losses
            msk = torch.from_numpy(msk).long()

        pixel_values.append(img)
        labels.append(msk)
        indices.append(idx)

    # stack into batch dims
    try:
        batch_images = torch.stack(pixel_values, dim=0)          # (B, C, H, W)
        batch_masks  = torch.stack(labels,      dim=0)           # (B, H, W)
        batch_idxs   = torch.tensor(indices,    dtype=torch.long)
    except Exception as e:
        raise RuntimeError(f"Error stacking batch tensors: {e}")

    return {
        "pixel_values": batch_images,
        "labels":       batch_masks,
        "indices":      batch_idxs,
    }



def custom_batch_size_scheduler(step: int,
                                batch_size: int,
                                interval=5,
                                increment=1):
    """
        step: current optimization step to be provided by the trainer.
        batch_size: current optimization step  to be provided by the trainer.
    """
    if step % interval == 0 and step > 0:
        return batch_size + increment
    return batch_size



# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
# put your training_args here
config = Mask2FormerConfig()
model  = Mask2FormerModel(config)
dummy_data = DummySegmentationDataset(
    num_samples=100,
    image_size=(3, 256, 256),
    num_classes=5
)
scheduler = partial(
    custom_batch_size_scheduler,
    interval=5,
    increment=1,
)
trainer = AdaptiveBatchSizeTrainer(
    batch_size_scheduler=scheduler,
    model=model,
    args=training_args,      
    train_dataset=dummy_data,
    data_collator=collate_fn
)

# Start training
trainer.train()
