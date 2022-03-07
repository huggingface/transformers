import tempfile
import unittest

from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    import torch.nn as nn
    from torch.utils.data.dataset import Dataset

    # nb: we don't want to inherit from IterableDataset to hit the right code path
    class DummyDataset(Dataset):
        def __init__(self, length: int = 101):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, i):
            if (i < 0) or (i >= self.length):
                raise IndexError
            return {"input_ids": [i]}

    class DummyModel(nn.Module):
        def __init__(self, num_params: int):
            super().__init__()
            # Add some (unused) params. the point here is that randomness in model_init shouldn't influence
            # data loader order.
            self.params = nn.Parameter(torch.randn(num_params))

        def forward(self, input_ids, labels=None):
            if labels is not None:
                return torch.tensor(0.0, device=input_ids.device), input_ids
            else:
                return input_ids


@require_torch
class TrainerSamplerSeedTest(unittest.TestCase):
    def test_trainer_sampler_seed(self):
        from transformers import Trainer, TrainingArguments

        def _get_first_data_sample(num_params, seed, **kwargs):
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = Trainer(
                    model_init=lambda: DummyModel(num_params),
                    args=TrainingArguments(
                        output_dir=tmpdir,
                        **kwargs,
                        seed=seed,
                        local_rank=-1,
                    ),
                    train_dataset=DummyDataset(),
                )

                return next(iter(trainer.get_train_dataloader()))

        # test that the seed is passed to the sampler
        # the codepath we want to hit is world_size <= 1, and both group_by_length
        for group_by_length in [True, False]:
            sample42_1 = _get_first_data_sample(num_params=10, seed=42, group_by_length=group_by_length)
            sample42_2 = _get_first_data_sample(num_params=11, seed=42, group_by_length=group_by_length)
            self.assertTrue(torch.equal(sample42_1["input_ids"], sample42_2["input_ids"]))

            # make sure we have some randomness
            others = [_get_first_data_sample(num_params=i, seed=i, group_by_length=group_by_length) for i in range(10)]
            self.assertTrue(any(not torch.equal(sample42_1["input_ids"], sample["input_ids"]) for sample in others))
